import matplotlib.pyplot as plt
import re

# 读取日志文件并提取数据
with open('logs\\241227_with_merge.txt', 'r', encoding='utf-16') as file:
    log_data = file.readlines()

# 定义正则表达式模式
training_pattern = r"Training Loss: ([\d.]+), mIoU: ([\d.]+)"
validation_pattern = r"Validation Loss: ([\d.]+), mIoU: ([\d.]+)"

# 初始化列表用于存储提取的数据
training_loss_list = []
training_miou_list = []
validation_loss_list = []
validation_miou_list = []

# 逐行解析日志文件
for line in log_data:
    # 匹配训练损失和mIoU
    training_match = re.search(training_pattern, line)
    if training_match:
        training_loss = float(training_match.group(1))
        training_miou = float(training_match.group(2))
        training_loss_list.append(training_loss)
        training_miou_list.append(training_miou)
    
    # 匹配验证损失和mIoU
    validation_match = re.search(validation_pattern, line)
    if validation_match:
        validation_loss = float(validation_match.group(1))
        validation_miou = float(validation_match.group(2))
        validation_loss_list.append(validation_loss)
        validation_miou_list.append(validation_miou)

# 创建一个包含所有 epoch 的列表，用于 x 轴
epochs = list(range(1, len(training_loss_list) + 1))

# 绘制训练损失和验证损失曲线
plt.figure(figsize=(12, 6))

# 绘制训练和验证损失曲线
plt.subplot(1, 2, 1)
plt.plot(epochs, training_loss_list, label='Training Loss', marker='o')
plt.plot(epochs, validation_loss_list, label='Validation Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# 绘制训练和验证 mIoU 曲线
plt.subplot(1, 2, 2)
plt.plot(epochs, training_miou_list, label='Training mIoU', marker='o')
plt.plot(epochs, validation_miou_list, label='Validation mIoU', marker='o')
plt.xlabel('Epoch')
plt.ylabel('mIoU')
plt.title('Training and Validation mIoU')
plt.legend()

# 显示图表
plt.tight_layout()
plt.show()