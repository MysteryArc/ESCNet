import os

import numpy as np
import torch
from PIL import Image

from escnet import ESCNet
from utils import FeatureConverter, rgb_to_xylab
from tqdm import tqdm

def save_result(pred, target, path):
    # 定义颜色映射表，比如0类映射到黑色，1类映射到白色
    color_map = {
        0: [0, 0, 0],  # black
        1: [50, 205, 50],  # limegreen
        2: [255, 165, 0],  # orange
        3: [123, 104, 238],  # mediumslateblue
        4: [199, 21, 133],  # mediumvioletred
        5: [211, 211, 211]  # lightgray
    }
    pred = pred.squeeze(0).to('cpu').numpy()
    target = target.squeeze(0).to('cpu').numpy()
    H, W = pred.shape

    # 创建一个空的图像数组用于存储结果
    segmented_image = np.zeros((H, W, 3), dtype=np.uint8)
    target_image = np.zeros((H, W, 3), dtype=np.uint8)

    # 将类别映射到颜色
    for i in range(H):
        for j in range(W):
            segmented_image[i, j] = color_map.get(pred[i, j])
            target_image[i, j] = color_map.get(target[i, j])

    # 转换为 RGB 图像并保存
    target_path = path.replace('.png', '_gt.png')
    img = Image.fromarray(segmented_image, mode='RGB')
    tgt_img = Image.fromarray(target_image, mode='RGB')
    img.save(path)
    tgt_img.save(target_path)

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

def to_target(target_image):
    # 转换为 numpy 数组
    target_array = np.array(target_image)

    # 将像素值为5的部分设置为0
    target_array[target_array == 5] = 1

    # 转换为 PyTorch 张量
    target_tensor = torch.tensor(target_array, dtype=torch.int64)

    return target_tensor

def to_tensor(image):
    '''
    将图像转换为xylab格式再转换为张量

    Args:
        image: 图像

    Returns:
        image_tensor: 转换为张量的图像
    '''

    image_xylab = rgb_to_xylab(image)
    image_tensor = torch.tensor(image_xylab).permute(2, 0, 1).float()
    return image_tensor

def main():
    PATH = 'C:\\Users\\why\\Desktop\\result'
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    NUM_CLASSES = 5
    ETA_POS = 2
    GAMMA_CLR = 0.1
    OFFSETS = (0.0, 0.0, 0.0, 0.0, 0.0)
    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    test_dir = "C:\\Users\\why\\Desktop\\xView2dataset\\x256\\val"
    model = ESCNet(
        FeatureConverter(ETA_POS, GAMMA_CLR, OFFSETS),
        n_iters=5,
        n_spixels=256,
        n_filters=64,
        in_ch=5,
        out_ch=20
    )
    model.load_state_dict(torch.load('./checkpoints/escnet_250116.pth', map_location=DEVICE))
    model.to(DEVICE)

    model.eval()
    conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    # 进入数据集目录
    pre_img_dir = os.path.join(test_dir, 'pre')
    post_img_dir = os.path.join(test_dir, 'post')
    target_dir = os.path.join(test_dir, 'targets')

    # 获取目录下文件名列表
    pre_img_list = os.listdir(pre_img_dir)

    for image_name in tqdm(pre_img_list, desc='Inferring'):
        # 读取图像
        pre_img_name = image_name
        post_img_name = pre_img_name.replace('pre_disaster', 'post_disaster')
        target_name = post_img_name.replace('disaster', 'disaster_target')
        pre_img_path = os.path.join(pre_img_dir, pre_img_name)
        post_img_path = os.path.join(post_img_dir, post_img_name)
        target_path = os.path.join(target_dir, target_name)
        pre_img = Image.open(pre_img_path)
        post_img = Image.open(post_img_path)
        target = Image.open(target_path)

        pre_img = to_tensor(pre_img)
        post_img = to_tensor(post_img)
        target = to_target(target)

        # 加载图像
        pre_image = pre_img.unsqueeze(0).to(DEVICE)
        post_image = post_img.unsqueeze(0).to(DEVICE)
        target = target.unsqueeze(0).to(DEVICE).long()

        with torch.no_grad():
            prob, prob_ds, (Q1, Q2), (ops1, ops2), (f1, f2) = model(pre_image, post_image, merge=True)
            pred = torch.argmax(prob, dim=1)
            conf_matrix = update_confusion_matrix(conf_matrix, pred, target)
            image_path = os.path.join(PATH, image_name.replace('_pre_disaster', '_result'))
            save_result(pred, target, image_path)

    miou_val, f1_loc_val, f1_dam_val, f1_c1_val, f1_c2_val, f1_c3_val, f1_c4_val = compute_metrics(conf_matrix)
    print('mIoU: {:.4f}, f1_loc: {:.4f}, f1_dam: {:.4f}, no: {:.4f}, minor: {:.4f}, major: {:.4f}, destroyed: {:.4f}'.format(miou_val, f1_loc_val, f1_dam_val, f1_c1_val, f1_c2_val, f1_c3_val, f1_c4_val))

if __name__ == '__main__':
    main()