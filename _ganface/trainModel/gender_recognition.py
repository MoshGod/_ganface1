#!/user/bin/env python    
#-*- coding:utf-8 -*-

"""样本很小，模型随时都会过拟合"""

from Model.mnist import *
from get_dataset import *

import torch
from torchvision import transforms
from torch import optim, nn
from torch.nn import functional as F
from torch.optim import lr_scheduler

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import time
import random
import sys

from normal import set_seed

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
set_seed(6666)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 数据预处理
transform = transforms.Compose([
    # transforms.Scale(size),
    transforms.Resize((224, 224)),
    # transforms.CenterCrop((size, size)),
    # transforms.RandomRotation(0.1),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  # PIL Image → Tensor
    # transforms.Gray()
])


# 获取数据
epochs = 22
batch_size = 32

root = r'D:\workspace\dataset\gender'
train_ds, train_loader = getDataLoader('gender', root=root+r'\train', transform=transform, batch_size=batch_size, shuffle_flag=True)
_, test_loader = getDataLoader('gender', root=root+r'\test', transform=transform, batch_size=batch_size)
classes = train_ds.classes
class_to_idx = train_ds.class_to_idx
idx_to_class = list(class_to_idx.keys())


"""最终确定模型后，把模型和参数用properties或xml联系起来"""
# MyGenderNet0: 3*conv + 2*fc 0.9265  acc: 0.9406  可以
desc = "3*conv + 2*fc"
model = ConvModel(image_size=224, dim_input=3, num_classes=2).to(device)
weight_save_path = '../saveModel/mygendernet_0_w.mdl'
model_save_path = '../saveModel/mygendernet_0.pkl'


# model = torch.load(model_save_path)
model.load_state_dict(torch.load(weight_save_path))



# 优化器 评价指标
lr = 0.005
optimizer = optim.Adam(model.parameters(), weight_decay=0, lr=lr)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
criteon = nn.CrossEntropyLoss().to(device)
# criteon = nn.BCEWithLogitsLoss().to(device)


def eval():
    global global_step
    model.eval()
    correct = 0
    test_loss = 0
    total = len(test_loader.dataset)
    step = 0
    for x, y in test_loader:
        step += 1
        # print(x.shape, y.shape) # torch.Size([1, 3, 32, 32]) torch.Size([1])
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            test_loss += criteon(logits, y).item()

            pred = logits.argmax(dim=1) # [b]
            correct += torch.eq(pred, y).sum().float().item()

    print('Test Loss {:.4f} Acc: {}/{}'.format(test_loss, correct, total))

    viz.line([[test_loss/step, correct / total]], [global_step], win='test', update='append')
    # viz.images(x.view(-1, 3, 224, 224), win='x')
    # 创建图片：图片的tensor，
    # viz.text(str(pred.detach().cpu().numpy()), win='pred', opts=dict(title='pred'))
    return test_loss/step, (correct, total)


global_step = 0

def train():
    global global_step
    best_acc, best_epoch = 0, 0
    since = time.time()

    for epoch in range(epochs):
        print('\nEpoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 20)
        # 学习率步进
        exp_lr_scheduler.step()
        model.train()
        runing_loss, runing_corrects = 0.0, 0.0
        for batchidx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device) # x: [b, 3, 32, 32], y: [b]

            logits = model(x)
            # y_pred = torch.argmax(logits.data, 1)
            loss = criteon(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            runing_loss += loss.item() * x.size(0)
            runing_corrects += torch.sum((torch.argmax(logits.data, 1))==y.data)

            global_step += 1

            if batchidx % 100 == 0:
                print('epoch {} batch {}'.format(epoch, batchidx), 'loss:', loss.item())
                viz.line([loss.item()], [global_step], win='train_loss', update='append')
                # 添加数据：添加y，添加x，添加在哪里？，更新方式是末尾添加

        epoch_loss = runing_loss / len(train_loader.dataset)
        epoch_acc = runing_corrects.double() / len(train_loader.dataset)
        print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        if epoch % 1 == 0:
            test_loss, (a, b) = eval()
            val_acc = a/b
            # print(a, b, a/b, val_acc)

            print('Test Loss {:.4f} Acc: {}/{} = {:.4f}'.format(test_loss, a, b, val_acc))
            # if val_acc> best_acc:
            #     best_epoch = epoch
            #     best_acc = val_acc
            #     torch.save(model.state_dict(), weight_save_path)
            #     torch.save(model, model_save_path)

    time_elapsed = time.time() - since
    print('\n\n','*'*20,'\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))
    print('best test acc:', best_acc, 'best epoch:', best_epoch)


def mytest():
    model = torch.load(model_save_path)
    print('loaded from ckpt!')
    print('Verifying... Waiting...')
    path = r'D:\workspace\dataset\myfaces\train\QC\111.jpg'
    img = Image.open(path)

    data = transform(img) # 3,32,32
    img_np = data.permute(1,2,0).numpy()
    plt.imshow(img_np)
    plt.show()
    # print(data.shape)
    data = torch.unsqueeze(data, 0).to(device)

    # data = data.permute(0, 2, 3, 1)
    with torch.no_grad():
        logits = model(data)
        probs = F.softmax(logits, dim=1)
        pred = logits.argmax(dim=1)
        print('the class to idx: {} → {}'.format(idx_to_class[pred.item()],pred.item()))
    print('The probs: ')
    for i, p in enumerate(idx_to_class):
        print(p,':\t','{:.6f}'.format(probs.cpu()[0][i].double().item()))
    print('\nThe Result: He is', idx_to_class[pred.item()])


if __name__ == '__main__':

    train()
    # eval()
    # mytest()





# 保存
# torch.save(model, '\model.pkl')
# # 加载
# model = torch.load('\model.pkl')