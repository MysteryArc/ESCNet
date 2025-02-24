import os
import numpy as np
import cv2
from tqdm import tqdm

def calculate_class_frequencies(label_dir, num_classes):
    """
    统计数据集中各类别的像素频率。

    参数：
        label_dir (str): 标签文件的目录路径，标签是灰度图像。
        num_classes (int): 类别总数（包括背景）。
    
    返回：
        class_counts (np.ndarray): 每个类别的像素总数。
    """
    # 初始化类别计数数组
    class_counts = np.zeros(num_classes, dtype=np.int64)

    # 遍历目录中的每个标签文件
    for filename in tqdm(os.listdir(label_dir), desc="Processing labels"):
        file_path = os.path.join(label_dir, filename)

        # 确保文件是图像
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            continue

        # 读取标签图像
        label_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if label_img is None:
            print(f"Warning: Unable to read {file_path}, skipping.")
            continue

        # 统计当前图像中各类别的像素数
        unique, counts = np.unique(label_img, return_counts=True)
        for u, c in zip(unique, counts):
            if u < num_classes:  # 忽略超过类别范围的值
                class_counts[u] += c

    return class_counts

def main():
    # 配置参数
    label_dir = "C:\\Users\\why\\Desktop\\xView2dataset\\train\\post\\targets"  # 替换为你的标签路径
    num_classes = 5  # 替换为你的类别总数

    # 统计类别频率
    class_counts = calculate_class_frequencies(label_dir, num_classes)

    # 输出结果
    total_pixels = class_counts.sum()
    class_frequencies = class_counts / total_pixels
    print("类别像素总数:", class_counts)
    print("类别频率:", class_frequencies)

if __name__ == "__main__":
    main()
