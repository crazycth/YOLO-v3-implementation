from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from pprint import pprint

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer,self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self,anchors):
        super(DetectionLayer,self).__init__()
        self.anchors = anchors

class Darknet(nn.Module):
    def __init__(self):

def parse_cfg(cfgfile):
    """
    解析CFG
    """
    file = open(cfgfile,'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if len(x) > 0]
    lines = [x for x in lines if x[0] != '#']
    lines = [x.rstrip().lstrip() for x in lines]

    # pprint(lines)

    block = {}
    blocks = []

    for line in lines:
        if line[0] == '[':
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()
        else:
            key,value = line.split('=')
            block[key.rstrip()] = value.lstrip()

    blocks.append(block)
    # pprint(blocks)
    return blocks


def create_modules(blocks):
    """
    一共有五种层，在此需要自己构建
    convolutional
    shortcut
    route
    yolo
    net : 网络描述层
    """
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []
    # print(net_info)

    for index,x in enumerate(blocks[1:]):
        module = nn.Sequential()
        if (x["type"]=="convolutional"):
            activation = x["activation"]
            try:
                batch_nomalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_nomalize = 0
                bias = True

            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1 )//2
            else:
                pad = 0

            conv = nn.Conv2d(prev_filters,filters,kernel_size,stride,pad,bias=bias)
            module.add_module("conv_{0}".format(index),conv)

            if batch_nomalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index),bn)

            if activation == "leaky":
                activn = nn.LeakyReLU(0.1,inplace=True)
                module.add_module("leaky_{0}".format(index),activn)
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor=2,mode="bilinear")
            module.add_module("upsample_{}".format(index),upsample)
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')
            start = int(x["layers"][0])
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()

            if end < 0:
                filters = output_filters[index+start] + output_filters[index+end]
            else:
                filters = output_filters[index+start]
        elif (x["type"] == "shortcut"):
            shorcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index),shorcut)
        elif (x["type"] == "yolo"):
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i],anchors[i+1]) for i in range(0,len(anchors),2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index),detection)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return (net_info,module_list)


if __name__ == '__main__':
    blocks = parse_cfg('cfg/yolov3.cfg')
    print(create_modules(blocks))