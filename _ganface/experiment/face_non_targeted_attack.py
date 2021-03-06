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

from attack.attack_bim import BIM
from get_dataset import *


"""获取数据"""
batch_size = 20
root = r'D:\workspace\dataset\myfaces\test'
_, ds_loader = getDataLoader('face', root=root, batch_size=batch_size, shuffle_flag=True)


"""获取预训练模型"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = models.inception_v3(pretrained=True).to(device)
model_save_path = '../saveModel/myfacenet_1_1108.pkl'
model = torch.load(model_save_path).to(device)
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
#     total += batch_size
#     # 累积正确个数
#     correct += (pre == labels).sum()
#     # 展示batch_size张图片 和 其预测的类别
#     # showBatchImages(torchvision.utils.make_grid(images.cpu().data, normalize=True), [classes[i] for i in pre], False)
#
# # 预测的平均正确率
# print('Accuracy of test text: %f %%' % (100 * float(correct) / total))


epss = np.linspace(0, 0.99, 100) #0.01

# y = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]
y = []

"""攻击后正确率"""
# 攻击参数
# eps = 0.7
# eps = torch.ra
a = BIM
for eps in epss:
    attack = a(model, eps=eps)

    loss = nn.CrossEntropyLoss()

    print("Attack Image & Predicted Label")

    correct = 0
    total = 0

    for images, labels in ds_loader:
        # 获取对抗样本
        images = attack(images, labels)
        labels = labels.to(device)
        # 推理对抗样本
        outputs = model(images)
        # 获取预测值
        _, pre = torch.max(outputs.data, dim=1)
        # 计算对抗后的正确率
        total += batch_size
        correct += (pre == labels).sum()
        # 展示batch_size张图片 和 其预测的类别
        # showBatchImages(torchvision.utils.make_grid(images.cpu().data, normalize=True), [classes[i] for i in pre])

    print('Accuracy of test text: %f %%' % (100 * float(correct) / total))
    y.append(100 * float(correct) / total)


plt.plot(epss, y)
plt.show()