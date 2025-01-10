import os
import numpy as np
import torch
from torchvision.transforms import Lambda
from torchvision.transforms import Compose
from utils import rgb_to_xylab
from torch.utils.data import Dataset
from PIL import Image

class PreloadDataset(Dataset):
    def __init__(self, root_dir, preload=False):
        """
        Args:
            root_dir (str): 数据集的根目录。
            preload (bool): 是否预加载到内存中。
        """
        super().__init__()
        self.pre_images = []
        self.post_images = []
        self.targets = []
        self.preload = preload

        # 获取路径列表
        self.pre_img_dir = os.path.join(root_dir, 'pre')
        self.post_img_dir = os.path.join(root_dir, 'post')
        self.target_dir = os.path.join(root_dir, 'targets')
        self.pre_img_list = sorted(os.listdir(self.pre_img_dir))
        self.post_img_list = sorted(os.listdir(self.post_img_dir))
        self.target_list = sorted(os.listdir(self.target_dir))

        if len(self.pre_img_list) != len(self.post_img_list) or len(self.pre_img_list) != len(self.target_list):
            raise ValueError("数据集文件数量不匹配！")

        if preload:
            # 预加载数据到内存中
            for pre_name, post_name, target_name in zip(self.pre_img_list, self.post_img_list, self.target_list):
                pre_path = os.path.join(self.pre_img_dir, pre_name)
                post_path = os.path.join(self.post_img_dir, post_name)
                target_path = os.path.join(self.target_dir, target_name)

                pre_image = Image.open(pre_path).copy()  # 灾前图像
                post_image = Image.open(post_path).copy()  # 灾后图像
                target = Image.open(target_path).copy()  # 标签图像

                self.pre_images.append(pre_image)
                self.post_images.append(post_image)
                self.targets.append(target)
        else:
            # 保存路径，延迟加载
            for pre_name, post_name, target_name in zip(self.pre_img_list, self.post_img_list, self.target_list):
                pre_path = os.path.join(self.pre_img_dir, pre_name)
                post_path = os.path.join(self.post_img_dir, post_name)
                target_path = os.path.join(self.target_dir, target_name)
                self.pre_images.append(pre_path)
                self.post_images.append(post_path)
                self.targets.append(target_path)

    def __getitem__(self, index):
        if self.preload:
            # 从内存中获取数据
            pre_image = self.image2tensor(self.pre_images[index])
            post_image = self.image2tensor(self.post_images[index])
            target = self.target2tensor(self.targets[index])
        else:
            # 从硬盘中加载数据
            pre_image = self.image2tensor(Image.open(self.pre_images[index]))
            post_image = self.image2tensor(Image.open(self.post_images[index]))
            target = self.target2tensor(Image.open(self.targets[index]))

        # print("Damage unique values:", target.unique())
        
        return pre_image, post_image, target

    def __len__(self):
        return len(self.pre_images)
    
    def target2tensor(self, target_image):
        '''
        由于target图像中的像素值只有0、1、2、3、4, 直接使用toTensor()函数会使像素值从[0, 255]归一化到[0, 1]区间, 此方法会将图像先转成numpy数组再转换成张量, 维持原来的像素值不变.

        Args:
            target_image: 标签图像
        
        Returns:
            target_tensor: 转换为tensor张量的标签图像
        '''

        # 转换为 numpy 数组
        target_array = np.array(target_image)
        
        # 将像素值为5的部分设置为0
        target_array[target_array == 5] = 1

        # 转换为 PyTorch 张量
        target_tensor = torch.tensor(target_array, dtype=torch.int64)
        
        return target_tensor
    
    def image2tensor(self, image):
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