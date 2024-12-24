import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from blocks import Conv3x3, MaxPool2x2, DoubleConv
from lib import (CalcAssoc, CalcPixelFeats, CalcSpixelFeats, InitSpixelFeats, RelToAbsIndex, Smear)
from utils import init_grid


class FeatureExtactor(nn.Module):
    def __init__(self, n_filters=64, in_ch=5, out_ch=20):
        super().__init__()
        self.conv1 = DoubleConv(in_ch, n_filters)
        self.pool1 = MaxPool2x2()
        self.conv2 = DoubleConv(n_filters, n_filters)
        self.pool2 = MaxPool2x2()
        self.conv3 = DoubleConv(n_filters, n_filters)
        self.conv4 = Conv3x3(3*n_filters+in_ch, out_ch-in_ch, act=True)
    
    def forward(self, x):
        f1 = self.conv1(x) # channel n_filters
        p1 = self.pool1(f1) 
        f2 = self.conv2(p1) # channel n_filters
        p2 = self.pool2(f2)
        f3 = self.conv3(p2) # channel n_filters

        # Resize feature maps
        f2_rsz = F.interpolate(f2, size=f1.shape[2:], mode='bilinear', align_corners=True)
        f3_rsz = F.interpolate(f3, size=f1.shape[2:], mode='bilinear', align_corners=True)

        # Concatenate multi-level features and fuse them
        f_cat = torch.cat([x, f1, f2_rsz, f3_rsz], dim=1) # channel in_ch + 3 × n_filters
        f_out = self.conv4(f_cat) # channel out_ch - in_ch

        y = torch.cat([x, f_out], dim=1) # channel out_ch

        return y


class SSN(nn.Module):
    def __init__(
        self, 
        feat_cvrter,
        n_iters=10, 
        n_spixels=100,
        n_filters=64, in_ch=5, out_ch=20,
        cnn=True
    ):
        super().__init__()

        # 超像素数量
        self.n_spixels = n_spixels
        # 迭代次数
        self.n_iters = n_iters
        # 特征转换器，用于调整输入特征的尺寸
        self.feat_cvrter = feat_cvrter

        # 是否使用CNN进行特征提取
        self.cnn = cnn
        if cnn:
            # 像素级特征提取器
            self.cnn_modules = FeatureExtactor(n_filters, in_ch, out_ch)
        else:
            self.cnn_modules = None

        # 缓存变量，用于加速计算
        self._cached = False
        self._ops = {}
        self._layout = (None, 1, 1)

    def forward(self, x):
        if self.training:
            # 训练模式，使用缓存的操作和布局,并且每次更新缓存
            ops, (_, nw_spixels, nh_spixels) = self.get_ops_and_layout(x, ofa=True)
        else:
            # 测试模式，每次更新操作和布局
            ops, (_, nw_spixels, nh_spixels) = self.get_ops_and_layout(x, ofa=False)

        # 调整输入特征的尺寸
        x = self.feat_cvrter(x, nw_spixels, nh_spixels)

        # 提取像素级特征
        pf = self.cnn_modules(x) if self.cnn else x

        # 初始化超像素特征
        spf = ops['init_spixels'](pf.detach())

        # 迭代优化超像素特征
        for itr in range(self.n_iters):
            # 计算像素与超像素之间的负距离
            Q = self.nd2Q(ops['calc_neg_dist'](pf, spf))
            # 更新超像素特征
            spf = ops['map_p2sp'](pf, Q)

        # 返回关联概率映射Q，操作字典ops，调整后的输入x，超像素特征spf，像素级特征pf
        return Q, ops, x, spf, pf

    @staticmethod
    def nd2Q(neg_dist):
        # 使用softmax计算像素与超像素的关联概率
        return F.softmax(neg_dist, dim=1)

    def get_ops_and_layout(self, x, ofa=False):
        if ofa and self._cached:
            # 返回缓存的操作和布局
            return self._ops, self._layout

        # 获取输入尺寸
        b, _, h, w = x.size()   

        # 初始化索引网格，获取超像素数量和尺寸
        init_idx_map, n_spixels, nw_spixels, nh_spixels = init_grid(self.n_spixels, w, h)
        init_idx_map = torch.IntTensor(init_idx_map).expand(b, 1, h, w).to(x.device)

        # 构建操作模块，用于后续计算
        init_spixels = InitSpixelFeats(n_spixels, init_idx_map)
        map_p2sp = CalcSpixelFeats(nw_spixels, nh_spixels, init_idx_map)
        map_sp2p = CalcPixelFeats(nw_spixels, nh_spixels, init_idx_map)
        calc_neg_dist = CalcAssoc(nw_spixels, nh_spixels, init_idx_map)
        map_idx = RelToAbsIndex(nw_spixels, nh_spixels, init_idx_map)
        smear = Smear(n_spixels)

        ops = {
            'init_spixels': init_spixels,
            'map_p2sp': map_p2sp,
            'map_sp2p': map_sp2p,
            'calc_neg_dist': calc_neg_dist,
            'map_idx': map_idx,
            'smear': smear
        }

        if ofa:
            # 缓存操作和布局
            self._ops = ops
            self._layout = (init_idx_map, nw_spixels, nh_spixels)
            self._cached = True

        # 返回操作字典ops和布局信息
        return ops, (init_idx_map, nw_spixels, nh_spixels)