#!/user/bin/env python    
#-*- coding:utf-8 -*- 

import  torch
from    torch import nn
from torch.nn import functional as F


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)



