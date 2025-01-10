'''
@File       :  dataset_load.py
@Time       :  2024/11/05 19:22:48
@Author     :  Wang Hengyu
@Description:  
'''

# here put the import lib
from torch.utils.data import Dataset
from PIL import Image
from skimage.io import imread, imsave
from utils import rgb_to_xylab
import os
import torchvision.transforms as transforms
import torch
import numpy as np
import skimage

class GetDataset(Dataset):
    def __init__(self, root_dir) -> None:
        '''
        Args:
            root_dir: 数据集的根目录, 输入时应包含数据集类型(train or test or value), 如'E://xView2dataset/train'

        Returns:
            pre_image:          灾前图像
            post_image:         灾后图像
            building_target:    建筑区域标签(包含二类: background and buildings)
            damage_target:      损坏区域标签(包含五类: 背景、完好建筑物、三类建筑物损坏)

        Damage type:
            0 for no building 
            1 for building found and classified no-damaged 
            2 for building found and classified minor-damage 
            3 for building found and classified major-damage
            4 for building found and classified destroyed

        Directory structure:
            /path/to/dataset
            |-train
            |   |-pre
            |   |_post
            |   |_targets
            |-val
            |_test
        '''

        super().__init__()
        self.pre_images = []
        self.post_images = []
        self.targets = []
        # 进入数据集目录
        self.pre_img_dir = os.path.join(root_dir, 'pre')
        self.post_img_dir = os.path.join(root_dir, 'post')
        self.target_dir = os.path.join(root_dir, 'targets')
        
        # 获取目录下文件名列表
        self.pre_img_list = os.listdir(self.pre_img_dir)
        self.post_img_list = os.listdir(self.post_img_dir)
        self.target_list = os.listdir(self.target_dir)
        if len(self.pre_img_list) != len(self.post_img_list) or len(self.pre_img_list) != len(self.target_list):
            raise ValueError('数据集数量不正确.')
        
        for image_name in self.pre_img_list:
            pre_img_name = image_name
            post_img_name = pre_img_name.replace('pre_disaster', 'post_disaster')
            target_name = post_img_name.replace('disaster', 'disaster_target')
            pre_img_path = os.path.join(self.pre_img_dir, pre_img_name)
            post_img_path = os.path.join(self.post_img_dir, post_img_name)
            target_path = os.path.join(self.target_dir, target_name)
            self.pre_images.append(pre_img_path)
            self.post_images.append(post_img_path)
            self.targets.append(target_path)

    def __getitem__(self, index):
        # 读取图像
        temp_pre_image = Image.open(self.pre_images[index]).copy()
        temp_post_image = Image.open(self.post_images[index]).copy()
        temp_damage_target = Image.open(self.targets[index]).copy()

        # 转换图像格式
        pre_image = self.image2tensor(temp_pre_image)
        post_image = self.image2tensor(temp_post_image)
        damage_target = self.target2tensor(temp_damage_target)

        # 调试：检查转换后的目标图像的唯一值
        # print("Damage unique values:", damage_target.unique())

        return pre_image, post_image, damage_target
    
    def __len__(self):
        return len(self.pre_img_list)
    
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