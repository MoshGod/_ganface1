#!/user/bin/env python    
#-*- coding:utf-8 -*- 

"""@@@
用于获取 dataset、dataset_loader、showImage
"""

import numpy as np
import json

import torch
import torch.utils.data as Data
import torchvision.utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import sys

def get_mnist(batch_size = 32, is_train=True):
    if is_train:
        train_ds = torchvision.datasets.MNIST(r'D:\workspace\dataset', train=True, download=True,
                                              transform=torchvision.transforms.Compose([
                                                  torchvision.transforms.ToTensor(),
                                                  # torchvision.transforms.Normalize(
                                                  #     (0.1307,), (0.3081,))
                                                  # 通道上的均值 和 方差
                                              ]))
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        return train_ds, train_loader
    else:
        test_ds = torchvision.datasets.MNIST(r'D:\workspace\dataset', train=False, download=True,
                                             transform=torchvision.transforms.Compose([
                                                 torchvision.transforms.ToTensor(),
                                                 # torchvision.transforms.Normalize(
                                                 #     (0.1307,), (0.3081,))
                                             ]))


        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        return test_ds, test_loader


def custom_label(data_mode, json_data_path):
    # 获取下标 → 类别字典 {"0": "QC",  "1": "LH"}
    class_idx = json.load(open(json_data_path))[data_mode]
    # 自定义的类别列表 ["QC", "LH"]
    idx2label = list(class_idx.values())
    return idx2label


# 图像文件夹自定义标签，得到的数据是Tensor对象
def getData(data_mode, root, json_data_path=r'../data/class_index.json',
                  transform = transforms.Compose([
                                transforms.Resize((299, 299)),
                                transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
                                ])):

    # 从分类文件夹 root 中读取数据，假设有两个文件夹是 LH、QC
    old_data = dsets.ImageFolder(root=root, transform=transform)
    # 初始的：会自动根据分的文件夹的名字按顺序分类 ["LH", "QC"]
    old_classes = old_data.classes
    # 自定义的 ["QC", "LH"]
    idx2label = custom_label(data_mode, json_data_path)
    label2idx = {}
    for i, item in enumerate(idx2label):
        label2idx[item] = i
    # 重新读取一次，类别重新映射
    new_data = dsets.ImageFolder(root=root, transform=transform,
                                 target_transform=lambda x: idx2label.index(old_classes[x]))
    # 新的类别
    new_data.classes = idx2label
    # 新的 类别 → 下标 映射
    new_data.class_to_idx = label2idx
    # 返回自定义标签后的ImageFolder数据

    return new_data


def getDataLoader(data_mode, root, json_data_path=r'../data/class_index.json', transform = transforms.Compose([
                                transforms.Resize((299, 299)),
                                transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
                                ]), batch_size=1, shuffle_flag=False):

    ds = getData(data_mode, root, json_data_path, transform)
    return ds, Data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle_flag,
                           num_workers=(0 if sys.platform.startswith('win32') else 4))


# 展示图片的，传进来的是img 是 Tensor，title是类别
def showImage(img, title, title_flag=True):
    npimg = img.numpy()
    fig = plt.figure() # figsize = (5, 5)
    plt.imshow(np.transpose(npimg,(1,2,0)))
    if(title_flag):
        plt.title(str(title[0])+str(title[1])+str(title[2]))
    plt.show()


def showBatchImages(ds, ds_loader, title_flag=True):
    normal_iter = iter(ds_loader)
    # 取出一张图片和其标签
    images, labels = normal_iter.next()

    print("True Image & True Label")
    # 会显示batch_size张图片
    img = torchvision.utils.make_grid(images, normalize=True)
    showImage(torchvision.utils.make_grid(images, normalize=True), [ds.classes[i] for i in labels], title_flag)

    return img

def showBatchImages2(images, labels, classes, title_flag=True):
    print("True Image & True Label")
    # 会显示batch_size张图片
    img = torchvision.utils.make_grid(images, normalize=True)
    showImage(torchvision.utils.make_grid(images, normalize=True), [classes[i] for i in labels], title_flag)

    return img

if __name__ == '__main__':
    train_ds = torchvision.datasets.MNIST(r'D:\workspace\dataset', train=True, download=True,
                                          transform=torchvision.transforms.Compose([
                                              transforms.Scale(299),
                                              torchvision.transforms.ToTensor(),
                                              torchvision.transforms.Normalize(
                                                  (0.1307,), (0.3081,))
                                              # 通道上的均值 和 方差
                                          ]))

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)

    classes = train_ds.classes
    class_to_idx = train_ds.class_to_idx
    idx_to_class = list(class_to_idx.keys())
    showBatchImages(train_ds, train_loader, False)



