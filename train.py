import torch
import numpy as np
import os
import time
from torch.utils.data import DataLoader, Subset
from dataset.dataset_load import GetDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from escnet import ESCNet
from utils import FeatureConverter
from tqdm import tqdm

def update_confusion_matrix(conf_matrix, preds, labels):
    """
    更新混淆矩阵
    Args:
        conf_matrix:    当前的混淆矩阵
        preds:          模型预测值，形状为 (batch_size, height, width)
        labels:         真实标签，形状为 (batch_size, height, width)

    Returns:
        conf_matrix: 更新后的混淆矩阵
    """
    preds_flat = preds.view(-1).long().cpu().numpy()
    labels_flat = labels.view(-1).long().cpu().numpy()

    for p, t in zip(preds_flat, labels_flat):
        conf_matrix[t, p] += 1

    return conf_matrix

def compute_metrics(conf_matrix):
    # 计算损伤分类指标
    num_classes = conf_matrix.shape[0]
    ious = []
    f1s = []
    for c in range(num_classes):
        tp = conf_matrix[c, c]
        fp = conf_matrix[:, c].sum() - tp
        fn = conf_matrix[c, :].sum() - tp
        denom = tp + fp + fn
        ious.append(tp / denom if denom else 0)
        f1c = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0
        f1s.append(f1c)
    # 计算miou和F1 score (damage)
    ious = np.array(ious)
    f1s_np = np.array(f1s[1:])
    miou = np.mean(ious)
    f1_damage = 4 / np.sum(1.0 / (f1s_np + 1e-6))

    # 计算建筑定位性能指标, 合并类别 1, 2, 3, 4 为一类
    binary_conf_matrix = np.zeros((2, 2), dtype=np.int64)
    
    # 类别 0：背景类, 类别 1：建筑类
    binary_conf_matrix[0, 0] = conf_matrix[0, 0]  # 真实 0 -> 预测 0
    binary_conf_matrix[0, 1] = conf_matrix[0, 1:].sum()  # 真实 0 -> 预测 1, 2, 3, 4
    binary_conf_matrix[1, 0] = conf_matrix[1:, 0].sum()  # 真实 1, 2, 3, 4 -> 预测 0
    binary_conf_matrix[1, 1] = conf_matrix[1:, 1:].sum()  # 真实 1, 2, 3, 4 -> 预测 1, 2, 3, 4
    
    # 计算 F1
    tp_loc = binary_conf_matrix[1, 1]
    fp_loc = binary_conf_matrix[0, 1]
    fn_loc = binary_conf_matrix[1, 0]

    f1_location = 2 * tp_loc / (2 * tp_loc + fp_loc + fn_loc) if (2 * tp_loc + fp_loc + fn_loc) else 0

    return miou, f1_location, f1_damage, f1s[1], f1s[2], f1s[3], f1s[4]

def main():
    # 初始化参数
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = 5
    LR = 0.001
    NUM_EPOCH = 50
    BATCH_SIZE = 8
    ETA_POS = 2
    GAMMA_CLR = 0.1
    OFFSETS = (0.0, 0.0, 0.0, 0.0, 0.0)
    train_set = GetDataset("C:\\Users\\why\\Desktop\\xView2dataset\\x256_lite\\train")
    val_set = GetDataset("C:\\Users\\why\\Desktop\\xView2dataset\\x256_lite\\val")

    if torch.cuda.is_available():
        print("Running on cuda: epoch={}, batchsize={}".format(NUM_EPOCH, BATCH_SIZE))
    else:
        raise Exception("No cuda available.")
    
    train_loader = DataLoader(train_set, BATCH_SIZE, pin_memory=True, drop_last=True, num_workers=8)
    val_loader = DataLoader(val_set, BATCH_SIZE, pin_memory=True, num_workers=8)
    model = ESCNet(
        FeatureConverter(ETA_POS, GAMMA_CLR, OFFSETS), 
        n_iters=5, 
        n_spixels=256,
        n_filters=64, 
        in_ch=5, 
        out_ch=20
    )
    model.to(DEVICE)

    # 定义损失函数、优化器、学习率调度器
    creterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-5)
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCH, eta_min=1e-5)

    # 训练循环
    start_time = time.time()
    last_time = start_time
    best_loss = 100
    for i in range(NUM_EPOCH):
        model.train()
        train_loss = 0
        conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

        for pre_image, post_image, damage_target in tqdm(train_loader, desc='Epoch {} Training'.format(i + 1)):
            # 加载图像
            pre_image = pre_image.to(DEVICE)
            post_image = post_image.to(DEVICE)
            target = damage_target.to(DEVICE).long()

            # 前向传播
            prob, prob_ds, (Q1, Q2), (ops1, ops2), (f1, f2) = model(pre_image, post_image, merge=False)
            loss = creterion(prob, target) + 0.5 * creterion(prob_ds, target)
            train_loss += loss

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 更新混淆矩阵
            preds = torch.argmax(prob, dim=1)
            conf_matrix = update_confusion_matrix(conf_matrix, preds, target)

        # 更新学习率
        scheduler.step()

        # 计算loss
        train_loss /= len(train_loader)

        # 计算评价指标
        miou_train, f1_loc_train, f1_dam_train, f1_c1, f1_c2, f1_c3, f1_c4 = compute_metrics(conf_matrix)
        print('Epoch: {} \t Training Loss: {:.6f}, mIoU: {:.4f}, f1_loc: {:.4f}, f1_dam: {:.4f}, no: {:.4f}, minor: {:.4f}, major: {:.4f}, destroyed: {:.4f}, LR: {}'.format(i + 1, train_loss, miou_train, f1_loc_train, f1_dam_train, f1_c1, f1_c2, f1_c3, f1_c4, scheduler.get_last_lr()))

        # 验证循环
        model.eval()
        val_loss = 0
        conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        for pre_image, post_image, damage_target in tqdm(val_loader, desc='Epoch {} Validating'.format(i + 1)):
            # 加载图像
            pre_image = pre_image.to(DEVICE)
            post_image = post_image.to(DEVICE)
            target = damage_target.to(DEVICE).long()

            with torch.no_grad():
                prob, prob_ds, (Q1, Q2), (ops1, ops2), (f1, f2) = model(pre_image, post_image, merge=False)
                loss = creterion(prob, target) + 0.5 * creterion(prob_ds, target)
                val_loss += loss.item()
                preds = torch.argmax(prob, dim=1)
                conf_matrix = update_confusion_matrix(conf_matrix, preds, target)

        # 计算评价指标
        val_loss = val_loss / len(val_loader)
        miou_val, f1_loc_val, f1_dam_val, f1_c1_val, f1_c2_val, f1_c3_val, f1_c4_val = compute_metrics(conf_matrix)
        print('Epoch: {} \t Validation Loss: {:.6f}, mIoU: {:.4f}, f1_loc: {:.4f}, f1_dam: {:.4f}, no: {:.4f}, minor: {:.4f}, major: {:.4f}, destroyed: {:.4f}'.format(i + 1, val_loss, miou_val, f1_loc_val, f1_dam_val, f1_c1_val, f1_c2_val, f1_c3_val, f1_c4_val))

        now = time.time()
        print("第{}轮训练的时长为{:.1f}秒".format(i + 1, now - last_time))
        last_time = now

        #模型保存
        if val_loss < best_loss:
            save_path = './checkpoints/escnet_250106_without_merge.pth'
            torch.save(model.state_dict(), save_path)
            best_loss = val_loss
            # print('save')
    end_time = time.time()
    print('{}轮训练的总时长为：{:.1f}秒'.format(NUM_EPOCH, end_time - start_time))

if __name__ == "__main__":
    if not os.path.exists("./checkpoints"):
        os.mkdir("./checkpoints")
    main()