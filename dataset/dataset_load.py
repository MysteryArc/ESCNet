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
            |   |-images
            |   |  |-pre
            |   |  |_post
            |   |_targets
            |      |-pre
            |      |_post
            |-val
            |_test
        '''

        super().__init__()
        # 进入数据集目录
        self.pre_img_dir = os.path.join(root_dir, 'pre', 'images')
        self.pre_tgt_dir = os.path.join(root_dir, 'pre', 'targets')
        self.post_img_dir = os.path.join(root_dir, 'post', 'images')
        self.post_tgt_dir = os.path.join(root_dir, 'post', 'targets')
        
        # 获取目录下文件名列表
        self.pre_img_list = os.listdir(self.pre_img_dir)
        self.post_img_list = os.listdir(self.post_img_dir)
        self.pre_tgt_list = os.listdir(self.pre_tgt_dir)
        self.post_tgt_list = os.listdir(self.post_tgt_dir)

        self.transforms_target = transforms.Compose(
            [
                transforms.Lambda(self.to_target),  # 将图像转换为张量
            ]
        )

    def to_target(self, target_image):
        '''
        由于target图像中的像素值只有0、1、2、3、4, 直接使用toTensor()函数会使像素值从[0, 255]归一化到[0, 1]区间, 此方法会将图像先转成numpy数组再转换成张量, 维持原来的像素值不变.

        Args:
            target_image: 标签图像
        
        Returns:
            target_tensor: 转换为tensor张量的标签图像
        '''

        # Convert the image to a tensor
        target_tensor = torch.tensor(np.array(target_image))
        return target_tensor
    
    def to_tensor(self, image):
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

    def __getitem__(self, index):
        pre_img_name = self.pre_img_list[index]
        post_img_name = pre_img_name.replace('pre_disaster', 'post_disaster')
        pre_tgt_name = pre_img_name.replace('disaster', 'disaster_target')
        post_tgt_name = post_img_name.replace('disaster', 'disaster_target')
        pre_img_path = os.path.join(self.pre_img_dir, pre_img_name)
        post_img_path = os.path.join(self.post_img_dir, post_img_name)
        pre_tgt_path = os.path.join(self.pre_tgt_dir, pre_tgt_name)
        post_tgt_path = os.path.join(self.post_tgt_dir, post_tgt_name)

        # 读取图像
        temp_pre_image = imread(pre_img_path)
        temp_post_image = imread(post_img_path)
        temp_building_target = Image.open(pre_tgt_path)
        temp_damage_target = Image.open(post_tgt_path)

        # 转换图像格式
        pre_image = self.to_tensor(temp_pre_image)
        post_image = self.to_tensor(temp_post_image)
        building_target = self.transforms_target(temp_building_target)
        damage_target = self.transforms_target(temp_damage_target)

        # 调试：检查转换后的目标图像的唯一值
        # print("Building unique values:", building_target.unique())
        # print("Damage unique values:", damage_target.unique())

        return pre_image, post_image, building_target, damage_target
    
    def __len__(self):
        return len(self.pre_img_list)