#!/user/bin/env python    
#-*- coding:utf-8 -*- 


import torch


def batch_per_image_standardization(imgs):
    # replicate tf.image.per_image_standardization, but in batch
    assert imgs.ndimension() == 4
    mean = imgs.view(imgs.shape[0], -1).mean(dim=1).view(
        imgs.shape[0], 1, 1, 1)
    return (imgs - mean) / batch_adjusted_stddev(imgs)


def batch_adjusted_stddev(imgs):
    # for batch_per_image_standardization
    std = imgs.view(imgs.shape[0], -1).std(dim=1).view(imgs.shape[0], 1, 1, 1)
    std_min = 1. / imgs.new_tensor(imgs.shape[1:]).prod().float().sqrt()
    return torch.max(std, std_min)

