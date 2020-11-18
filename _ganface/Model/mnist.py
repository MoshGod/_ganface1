#!/user/bin/env python    
#-*- coding:utf-8 -*-

import math

import  torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18

from model_utils import Flatten


from util import *


# 默认参数，不是很重要，但可以学习
DIM_INPUT = 15
NUM_CLASS = 10
BATCH_SIZE = 16

IMAGE_SIZE = 16
COLOR_CHANNEL = 3


class SimpleModel(nn.Module):
    def __init__(self, dim_input=DIM_INPUT, num_classes=NUM_CLASS):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(dim_input, 10)
        self.fc2 = nn.Linear(10, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class SimpleImageModel(nn.Module):

    def __init__(self, num_classes=NUM_CLASS):
        super(SimpleImageModel, self).__init__()
        self.num_classes = NUM_CLASS
        self.conv1 = nn.Conv2d(
            COLOR_CHANNEL, 8, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(4)
        self.linear1 = nn.Linear(4 * 4 * 8, self.num_classes)

    def forward(self, x):
        out = self.maxpool1(self.relu1(self.conv1(x)))
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        return out


class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(2)
        self.linear1 = nn.Linear(7 * 7 * 64, 200)
        self.relu3 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(200, 10)

    def forward(self, x):
        out = self.maxpool1(self.relu1(self.conv1(x)))
        out = self.maxpool2(self.relu2(self.conv2(out)))
        out = out.view(out.size(0), -1)
        out = self.relu3(self.linear1(out))
        out = self.linear2(out)
        return out


class MLP(nn.Module):
    # MLP-300-100

    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(28 * 28, 300)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(300, 100)
        self.relu2 = nn.ReLU(inplace=True)
        self.linear3 = nn.Linear(100, 10)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.linear1(out)
        out = self.relu1(out)
        out = self.linear2(out)
        out = self.relu2(out)
        out = self.linear3(out)
        return out


class ConvModel(nn.Module):

    def __init__(self, image_size=IMAGE_SIZE, dim_input=DIM_INPUT, num_classes=NUM_CLASS):
        super(ConvModel, self).__init__()
        self.stride = 2
        if image_size == 28:
            self.fl_size = 64
            self.stride = 1
        elif image_size == 128:
            self.fl_size = 3136
        elif image_size == 299:
            self.fl_size = 18496
        elif image_size == 224:
            self.fl_size = 10816
        else:
            self.fl_size = 1

        self.conv_unit = nn.Sequential(
            nn.Conv2d(dim_input, 16, kernel_size=3, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=self.stride, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
        )

        self.fc_unit = nn.Sequential(
            nn.Linear(self.fl_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        batchsz = x.size(0)
        x = self.conv_unit(x)
        # print(x.shape)
        x = x.view(batchsz, self.fl_size)
        logits = self.fc_unit(x)

        return logits


class ConvWidthModel(nn.Module):
    def __init__(self):
        super(ConvWidthModel, self).__init__()

        self.conv_unit1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(96),

            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(256),

            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            Flatten()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ReLU(),
            nn.Dropout()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 3)
        )

    def forward(self, x):
        x = self.conv_unit1(x)
        # print(x.size())
        x = self.fc1(x)
        logits = self.fc2(x)

        return logits


class LeNet5Madry(nn.Module):

    def __init__(
            self, nb_filters=(1, 32, 64), kernel_sizes=(5, 5),
            paddings=(2, 2), strides=(1, 1), pool_sizes=(2, 2),
            nb_hiddens=(7 * 7 * 64, 1024), nb_classes=10):
        super(LeNet5Madry, self).__init__()
        self.conv1 = nn.Conv2d(
            nb_filters[0], nb_filters[1], kernel_size=kernel_sizes[0],
            padding=paddings[0], stride=strides[0])
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(pool_sizes[0])
        self.conv2 = nn.Conv2d(
            nb_filters[1], nb_filters[2], kernel_size=kernel_sizes[1],
            padding=paddings[0], stride=strides[0])
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(pool_sizes[1])
        self.linear1 = nn.Linear(nb_hiddens[0], nb_hiddens[1])
        self.relu3 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(nb_hiddens[1], nb_classes)

    def forward(self, x):
        out = self.maxpool1(self.relu1(self.conv1(x)))
        out = self.maxpool2(self.relu2(self.conv2(out)))
        out = out.view(out.size(0), -1)
        out = self.relu3(self.linear1(out))
        out = self.linear2(out)
        return out


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_planes, out_planes, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.drop_rate = drop_rate
        self.in_out_equal = (in_planes == out_planes)

        if not self.in_out_equal:
            self.conv_shortcut = nn.Conv2d(
                in_planes, out_planes, kernel_size=1, stride=stride,
                padding=0, bias=False)

    def forward(self, x):
        out = self.relu1(self.bn1(x))
        if not self.in_out_equal:
            x = self.conv_shortcut(out)
        out = self.relu2(self.bn2(self.conv1(out)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        out += x
        return out


class ConvGroup(nn.Module):
    def __init__(
            self, num_blocks, in_planes, out_planes, block, stride,
            drop_rate=0.0):
        super(ConvGroup, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, num_blocks, stride, drop_rate)

    def _make_layer(
            self, block, in_planes, out_planes, num_blocks, stride, drop_rate):
        layers = []
        for i in range(int(num_blocks)):
            layers.append(
                block(in_planes=in_planes if i == 0 else out_planes,
                      out_planes=out_planes,
                      stride=stride if i == 0 else 1,
                      drop_rate=drop_rate)
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, drop_rate=0.0,
                 color_channels=3, block=BasicBlock):
        super(WideResNet, self).__init__()
        num_channels = [
            16, int(16 * widen_factor),
            int(32 * widen_factor), int(64 * widen_factor)]
        assert((depth - 4) % 6 == 0)
        num_blocks = (depth - 4) / 6

        self.conv1 = nn.Conv2d(
            color_channels, num_channels[0], kernel_size=3, stride=1,
            padding=1, bias=False)
        self.convgroup1 = ConvGroup(
            num_blocks, num_channels[0], num_channels[1], block, 1, drop_rate)
        self.convgroup2 = ConvGroup(
            num_blocks, num_channels[1], num_channels[2], block, 2, drop_rate)
        self.convgroup3 = ConvGroup(
            num_blocks, num_channels[2], num_channels[3], block, 2, drop_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(num_channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(num_channels[3], num_classes)
        self.num_channels = num_channels[3]

        for mod in self.modules():
            if isinstance(mod, nn.Conv2d):
                n = mod.kernel_size[0] * mod.kernel_size[1] * mod.out_channels
                mod.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(mod, nn.BatchNorm2d):
                mod.weight.data.fill_(1)
                mod.bias.data.zero_()
            elif isinstance(mod, nn.Linear):
                mod.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.convgroup1(out)
        out = self.convgroup2(out)
        out = self.convgroup3(out)
        out = self.relu(self.bn1(out))
        out = out.mean(dim=-1).mean(dim=-1)
        out = self.fc(out)
        return out


class PerImageStandardize(nn.Module):
    def __init__(self):
        super(PerImageStandardize, self).__init__()

    def forward(self, tensor):
        return batch_per_image_standardization(tensor)


def get_lenet5madry_with_width(widen_factor):
    return LeNet5Madry(
        nb_filters=(1, int(widen_factor * 32), int(widen_factor * 64)),
        nb_hiddens=(7 * 7 * int(widen_factor * 64), int(widen_factor * 1024)))

def get_cifar10_wrn28_widen_factor(widen_factor):
    model = WideResNet(28, 10, widen_factor)
    model = nn.Sequential(PerImageStandardize(), model)
    return model





if __name__ == '__main__':
    x = torch.rand((1,1,28,28))

    # trained_model = resnet18(pretrained=True)  # 设置True，表明使用训练好的参数  224*224
    # model = nn.Sequential(*list(trained_model.children())[:-1],  # [b, 512, 1, 1]
    #                       Flatten(),  # [b, 512, 1, 1] => [b, 512]
    #                       # nn.Linear(512, 5)
    #                       )

    # x = batch_per_image_standardization(x)
    # model = nn.Sequential(
    #     PerImageStandardize(),
    #     WideResNet(124, 10, 1)
    # )
    model = ConvModel(image_size=28, dim_input=1, num_classes=10) # 年龄
    # ConvModel(image_size=299, num_classes=12) # 人脸
    # ConvModel(image_size=224, num_classes=2)  # 性别
    print(model(x).size())
