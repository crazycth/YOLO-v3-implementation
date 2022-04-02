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


def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res


def bbox_iou(box1,box2):
    """
    计算两个box之间的IOU
    """
    b1_x1 , b1_y1 , b1_x2 , b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1 , b2_y1 , b2_x2 , b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    inter_rect_x1 = torch.max(b1_x1,b2_x1)
    inter_rect_y1 = torch.max(b1_y1,b2_y1)
    inter_rect_x2 = torch.min(b1_x2,b2_x2)
    inter_rect_y2 = torch.min(b1_y2,b2_y2)

    inter_area = torch.clamp(inter_rect_x2-inter_rect_x1+1,min=0) * torch.clamp(inter_rect_y2-inter_rect_y1+1,min=0)

    b1_area = (b1_x2-b2_x1+1)*(b1_y2-b2_y1+1)
    b2_area = (b2_x2-b2_x1+1)*(b2_y2-b2_y1+1)

    iou = inter_area/(b1_area+b2_area-inter_area)

    return iou




def write_results(prediction,confidence,num_classes,nms_conf=0.4):
    """
    :param prediction: [Batch_size*10647*85] 每张图像一共建立了10647个图像，85是边界框的属性
    :param confidence: 目标分数阀值
    :param num_classes: 种类数，在此以80为例子 所以85 = 80 + 5
    :param nms_conf: NMS IOU阀值

    prediction : 0:x 1:y 2:h 3:w 4:conf
    """

    conf_mask = (prediction[:,:,4]>confidence).float().unsqueeze(2)
    prediction = prediction * conf_mask
    box_corner = prediction.new(prediction.shape)

    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2)
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_corner[:,:,:4]

    batch_size = prediction.size(0)
    write = False

    for ind in range(batch_size):
        #拿出特定的一张图片
        image_pred = prediction[ind]
        max_conf , max_conf_score = torch.max(image_pred[:,5:5+num_classes],1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)

        #80个类别只关心最大的那个的置信度和分数
        seq = (image_pred[:,:5],max_conf,max_conf_score)
        image_pred = torch.cat(seq,1)

        #左上角X，左上角Y，右下角X，右下角Y，conf, 分类最大置信度分数，最大置信度坐标
        non_zero_ind = (torch.nonzero(image_pred[:,4]))
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        except:
            continue

        if image_pred_.shape[0] == 0:
            continue

        img_classes = torch.unique(image_pred_[:,-1])

        # 在这里按类别做NMS
        for cls in img_classes:

            cls_mask = image_pred_*(image_pred_[:,-1]==cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1,7)

            conf_sort_index = torch.sort(image_pred_class[:,4],descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)

            for i in range(idx):
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0),image_pred_class[i+1:])
                except ValueError:
                    break
                except IndexError:
                    break

                iou_mask = (ious<nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask

                non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1,7)


            #指明是这个batch中的那一张图片
            batch_ind = image_pred_class.new(image_pred_class.size(0),1).fill_(ind)
            seq = batch_ind , image_pred_class

            # print("a",batch_ind.shape)
            # print(image_pred_class.shape)

            #哪一张图片 四个角坐标 conf 最大置信度类别的分数 最大置信度类别索引 共8个 叠加在一起

            if not write:
                output = torch.cat(seq,1)
                write = True
            else:
                out = torch.cat(seq,1)
                output = torch.cat((output,out))
    try:
        return output

    except:
        return 0


def load_classes(namesfile):
    fp = open(namesfile,"r")
    names = fp.read().split("\n")[:-1]
    return names


def letterbox_image(img,inp_dim):
    img_w , img_h = img.shape[1],img.shape[0]
    w , h = inp_dim
    new_w = int(img_w*min(w/img_w,h/img_h))
    new_h = int(img_h*min(w/img_w,h/img_h))
    resized_image = cv2.resize(img,(new_w,new_h),interpolation=cv2.INTER_CUBIC)
    canvas = np.full((inp_dim[1],inp_dim[0],3),128)
    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image
    return canvas


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """

    img = letterbox_image(img, (inp_dim, inp_dim))
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img

