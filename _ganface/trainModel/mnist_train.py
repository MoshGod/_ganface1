#!/user/bin/env python    
#-*- coding:utf-8 -*-

"""样本很小，模型随时都会过拟合"""
from Model.mnist import *

import torch
import torchvision
from torchvision import transforms
from torch import optim, nn
from torch.nn import functional as F

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import time
import random
import sys
sys.path.append("..")

import warnings
warnings.filterwarnings("ignore") # 忽略警告

from visdom import Visdom  # cmd输入 python -m visdom.server 启动

viz = Visdom() # 实例化

viz.line([0.], [0.],         win='train_loss', opts=dict(title='train loss'))
viz.line([[0.0, 0.0]], [0.], win='test',       opts=dict(title='test loss&acc.', legend=['loss', 'acc.']))
# 创建直线：y,x,窗口的id(environment),窗口其它配置信息
# 复试折线图

# Set the random seed manually for reproducibility.
seed = 6666
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 获取数据
epochs = 20
batch_size = 32

train_ds = torchvision.datasets.MNIST(r'D:\workspace\dataset', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   # torchvision.transforms.Normalize(
                                   #     (0.1307,), (0.3081,) )
                                   # 通道上的均值 和 方差
                               ]))
test_ds = torchvision.datasets.MNIST(r'D:\workspace\dataset', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   # torchvision.transforms.Normalize(
                                   #     (0.1307,), (0.3081,))
                               ]))
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
classes = train_ds.classes
class_to_idx = train_ds.class_to_idx
idx_to_class = list(class_to_idx.keys())


# MyANet0: 3*conv + 2*fc
desc = "3*conv + 2*fc"
model = ConvModel(image_size=28, dim_input=1, num_classes=10).to(device)
weight_save_path = '../saveModel/mnist_1_w.mdl'
model_save_path = '../saveModel/mnist_1_0.pkl'

# model = torch.load(model_save_path)
# model.load_state_dict(torch.load(weight_save_path))


# 优化器 评价指标
lr = 0.005
optimizer = optim.Adam(model.parameters(), weight_decay=1e-8, lr=lr)
criteon = nn.CrossEntropyLoss().to(device)

def eval():
    global global_step
    model.eval()
    correct = 0
    test_loss = 0
    total = len(test_loader.dataset)
    step = 0

    for x, y in test_loader:
        step += 1
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1) # [b]
            test_loss += criteon(logits, y).item()
            correct += torch.eq(pred, y).sum().float().item()

    print('Test Loss {:.4f} Acc: {}/{}'.format(test_loss, correct, total))
    viz.line([[test_loss/step, correct / total]], [global_step], win='test', update='append')
    viz.images(x.view(-1, 1, 28, 28), win='x')
    # 创建图片：图片的tensor，
    viz.text(str(pred.detach().cpu().numpy()), win='pred', opts=dict(title='pred'))
    return test_loss/step, (correct, total)

global_step = 0

def train():
    global global_step
    best_acc, best_epoch = 0, 0
    since = time.time()

    for epoch in range(epochs):
        print('\nEpoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 20)

        runing_loss, runing_corrects = 0.0, 0.0
        for batchidx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device) # x: [b, 3, 32, 32], y: [b]

            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            # y_pred = torch.argmax(logits.data, 1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            runing_loss += loss.item() * x.size(0)
            runing_corrects += torch.sum((torch.argmax(logits.data, 1))==y.data)

            global_step += 1

            if batchidx % 100 == 0:
                print('epoch {} batch {}'.format(epoch, batchidx), 'loss:', loss.item())
                viz.line([loss.item()], [global_step], win='train_loss', update='append')

        epoch_loss = runing_loss / len(train_loader.dataset)
        epoch_acc = runing_corrects.double() / len(train_loader.dataset)
        print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        if epoch % 1 == 0:
            test_loss, (a, b) = eval()
            val_acc = a/b
            if val_acc> best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), weight_save_path)
                torch.save(model, model_save_path)

    time_elapsed = time.time() - since
    print('\n\n','*'*20,'\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))
    print('best test acc:', best_acc, 'best epoch:', best_epoch)



if __name__ == '__main__':

    train()
    # eval()


# 保存
# torch.save(model, '\model.pkl')
# # 加载
# model = torch.load('\model.pkl')