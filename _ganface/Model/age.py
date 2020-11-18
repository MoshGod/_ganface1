#!/user/bin/env python    
#-*- coding:utf-8 -*- 

import  torch
from    torch import nn
from torch.nn import functional as F

from model_utils import Flatten


class MyAgeNet0(nn.Module):

    def __init__(self):
        super(MyAgeNet0, self).__init__()

        self.conv_unit = nn.Sequential(
            # [b, 1, 224, 224] => [b, 16, 60, 60] => [b, 16, 30, 30]
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # [b, 16, 30, 30] => [b, 32, 28, 28] => [b, 32, 14, 14]
            nn.ReLU(),

            # [b, 32, 14, 14] => [b, 64, 12, 12] => [b, 64, 6, 6]
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU(),

        )

        self.fc_unit = nn.Sequential(
            nn.Linear(64*14*14, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        batchsz = x.size(0)
        x = self.conv_unit(x)
        x = x.view(batchsz, 64*14*14)
        logits = self.fc_unit(x)

        return logits


class MyAgeNet1(nn.Module):
    def __init__(self):
        super(MyAgeNet1, self).__init__()

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
            nn.Linear(1536,512),
            nn.ReLU(),
            nn.Dropout()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512,3)
        )

    def forward(self, x):
        x = self.conv_unit1(x)
        # print(x.size())
        x = self.fc1(x)
        logits = self.fc2(x)

        return logits



if __name__ == '__main__':
    x = torch.rand((1,3,128,128))
    model = MyAgeNet0()
    model(x)
