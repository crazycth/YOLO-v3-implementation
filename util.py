from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2

def predict_transform(prediction,inp_dim,anchors,num_classes,CUDA=True):
    """
    :param prediction: 输出
    :param inp_dim: 输入图像尺寸
    :param anchors:
    :param num_classes:
    :param CUDA: 是否使用CUDA
    :return:
    """
    batch_size = prediction.size(0)
    #输入图像是检测图像的stride倍
    stride = inp_dim // prediction.size(2)

    #grid_size就是特征图的大小
    grid_size = inp_dim//stride

    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    prediction = prediction.view(batch_size,bbox_attrs*num_anchors,grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size,grid_size*grid_size*num_anchors,bbox_attrs)

    #当前anchor大小是根据最开始输入来定的，为此需要重新确定一下锚框大小
    anchors = [(a[0]/stride,a[1]/stride) for a in anchors]

    #对x,y坐标(在此只用一个值表示）以及目标分数进行sigmoid变换
    #0:x 1:y 2:w 3:h 4:confidence 5-classes:分类置信度
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])

    #将网格偏移添加到预测的中心坐标
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid,grid)

    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset,y_offset),1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)

    prediction[:,:,:2] += x_y_offset

    #将锚应用于边界框的尺寸
    anchors = torch.FloatTensor(anchors)
    if CUDA:
        anchors = anchors.cuda()
    anchors = anchors.repeat(grid_size*grid_size,1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors

    #将sigmoid激活函数用于类别分类
    prediction[:,:,5:5+num_classes] = torch.sigmoid((prediction[:,:,5:5+num_classes]))

    prediction[:,:,:4] *= stride

    return prediction






