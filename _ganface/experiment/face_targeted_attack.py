#!/user/bin/env python    
#-*- coding:utf-8 -*- 

"""@@@
用于评估人脸身份识别对抗攻击算法的攻击效果

需要整合成可以迭代计算预先准备好的对抗算法的模式

"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from attack.attack_fgsm import *
from get_dataset import *


"""获取数据"""
batch_size = 20
root = r'D:\workspace\dataset\myfaces\all'
_, ds_loader = getDataLoader('face', root=root, batch_size=batch_size, shuffle_flag=True)


"""获取预训练模型"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = models.inception_v3(pretrained=True).to(device)
model_save_path = '../saveModel/mynet_0.pkl'
model = torch.load(model_save_path)
# 直接切换到推理模式
model.eval()


"""未攻击正确率"""
# print("True Image & Predicted Label")
#
# correct = 0
# total = 0
#
# for images, labels in ds_loader:
#     images, labels = images.to(device), labels.to(device)
#     # 前向传播
#     outputs = model(images)
#     # max 返回(最大值,下标)， pre size: [b]
#     _, pre = torch.max(outputs.data, dim=1)
#     # 迭代次数+1
#     total += 1
#     # 累积正确个数
#     correct += (pre == labels).sum()
#     # 展示batch_size张图片 和 其预测的类别
#     # showBatchImages(torchvision.utils.make_grid(images.cpu().data, normalize=True), [classes[i] for i in pre], False)
#
# # 预测的平均正确率
# print('Accuracy of test text: %f %%' % (100 * float(correct) / total))




"""攻击后正确率"""


