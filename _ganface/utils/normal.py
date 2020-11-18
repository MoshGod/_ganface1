#!/user/bin/env python    
#-*- coding:utf-8 -*- 

import os
import sys
import pathlib

from torch import nn
from torch.utils.data.dataset import Subset

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def mkdir(directory):
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)


def bchw2bhwc(x):
    if isinstance(x, np.ndarray):
        pass
    else:
        raise

    if x.ndim == 3:
        return np.moveaxis(x, 0, 2)
    if x.ndim == 4:
        return np.moveaxis(x, 1, 3)


def bhwc2bchw(x):
    if isinstance(x, np.ndarray):
        pass
    else:
        raise

    if x.ndim == 3:
        return np.moveaxis(x, 2, 0)
    if x.ndim == 4:
        return np.moveaxis(x, 3, 1)


class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)


def normalize_fn(tensor, mean, std):
    """Differentiable version of torchvision.functional.normalize"""
    # here we assume the color channel is in at dim=1
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)


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


class PerImageStandardize(nn.Module):
    def __init__(self):
        super(PerImageStandardize, self).__init__()

    def forward(self, tensor):
        return batch_per_image_standardization(tensor)


def predict_from_logits(logits, dim=1):
    return logits.max(dim=dim, keepdim=False)[1]


def get_accuracy(pred, target):
    return pred.eq(target).float().mean().item()


def set_torch_deterministic():
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True


def set_seed(seed=None):
    import torch
    import numpy as np
    import random
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True