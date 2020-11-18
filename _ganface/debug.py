#!/user/bin/env python    
#-*- coding:utf-8 -*-

"""样本很小，模型随时都会过拟合"""

import torch
from torchvision import datasets, transforms
from torch import optim, nn
from torch.utils.data import DataLoader
from torch.nn import functional as F

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import time
import random
import sys

from lenet5 import Lenet5

sys.path.append("..")
import warnings
warnings.filterwarnings("ignore") # 忽略警告

from visdom import Visdom
# cmd输入 python -m visdom.server 启动

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


# 硬件参数
num_workers = 0 if sys.platform.startswith('win32') else 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""默认图像数据目录结构
root
.
├──dog
|   ├──001.png
|   ├──002.png
|   └──...
└──cat
|   ├──001.png
|   ├──002.png
|   └──...
└──...

拥有成员变量:
  · self.classes - 用一个list保存 类名
  · self.class_to_idx - 类名对应的 索引
  · self.imgs - 保存(img-path, class) tuple的list

ds = datasets.ImageFolder('./data/dogcat_2') #没有transform，先看看取得的原始图像数据
ds.classes # 根据分的文件夹的名字来确定的类别          ['cat', 'dog']
ds.class_to_idx # 按顺序为这些类别定义索引为0,1...     {'cat': 0, 'dog': 1}
ds.imgs # 返回从所有文件夹中得到的图片的路径以及其类别  [('./data/dogcat_2/cat/cat.12484.jpg', 0), ('./data/dogcat_2/cat/cat.12485.jpg', 0), ...]

ds[0]    # 第1个图片的元组
ds[0][0] # 图像Tensor数据
ds[0][1] # 得到的是类别0，即cat
"""


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
epochs = 25
batch_size = 32
num_classes = 2 # 类别
# input_channel = 1
root = r'D:\workspace\dataset\gender'
train_ds = datasets.ImageFolder(root=root+r'\train', transform=transform) # 得到的数据是Tensor对象
test_ds = datasets.ImageFolder(root=root+r'\test', transform=transform) # 得到的数据是Tensor对象
# print(train_ds.class_to_idx)
# print(test_ds.class_to_idx)
idx_to_class = {0: 'CC', 1: 'CLT', 2: 'DHH', 3: 'DJR', 4: 'DJX', 5: 'HZH', 6: 'PHS', 7: 'QC', 8: 'QOO', 9: 'ZY'}


# img = plt.imread(path)
# print(img.shape)
# img = np.resize(img, (224, 224, 3))
# img_plt = train_ds[50][0]
# img_np = np.array(img_plt)
# print(img_np.shape)
# # print(train_ds[0][0][0].shape)
# plt.imshow(img_np)
# plt.show()


train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_ds, num_workers=num_workers)


# resnet18_0: 224 rgb
# pretrain_model = resnet18(pretrained=True).to(device)
# model = nn.Sequential(*list(pretrain_model.children())[:-1],  # [b, 512, 1, 1]
#                       Flatten(),  # [b, 512, 1, 1] => [b, 512]
#                       nn.Linear(512, num_classes)).to(device)
# model_save_path = 'resnet18_0_w.mdl'
# model_save_path = 'resnet18_0.pkl'

from Model.gender import *
# lenet5_0: 32 rgb
desc = "lenet5 32*32"
model = MyGenderNet1().to(device)
weight_save_path = 'saveModel/mygendernet_1_w.mdl'
model_save_path = 'saveModel/mygendernet_1_w.pkl'


# model = torch.load(model_save_path)

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

    for x, y in test_loader:
        # print(x.shape, y.shape) # torch.Size([1, 3, 32, 32]) torch.Size([1])
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1) # [b]
            test_loss += criteon(logits, y).item()
            correct += torch.eq(pred, y).sum().float().item()

    viz.line([[test_loss/total, correct / total]], [global_step], win='test', update='append')
    # viz.images(x.view(-1, 3, 299, 299), win='x')
    # 创建图片：图片的tensor，
    # viz.text(str(pred.detach().cpu().numpy()), win='pred', opts=dict(title='pred'))
    return (correct, total)

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

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            runing_loss += loss.item() * x.size(0)
            runing_corrects += torch.sum((torch.argmax(logits.data, 1))==y.data)

            global_step += 1
            # 添加数据：添加y，添加x，添加在哪里？，更新方式是末尾添加

            if batchidx % 200 == 0:
                print('epoch {} batch {}'.format(epoch, batchidx), 'loss:', loss.item())
                viz.line([loss.item()], [global_step], win='train_loss', update='append')

        epoch_loss = runing_loss / len(train_loader.dataset)
        epoch_acc = runing_corrects.double() / len(train_loader.dataset)
        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        if epoch % 1 == 0:
            a, b = eval()
            val_acc = a/b
            print('Test: {} Acc: {}/{}'.format(epoch, a, b))
            if val_acc> best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), weight_save_path)
                torch.save(model, model_save_path)

    time_elapsed = time.time() - since
    print('\n\n','*'*20,'\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))
    print('best test acc:', best_acc, 'best epoch:', best_epoch)


def mytest():
    # model =
    # weight_load_path = weight_save_path
    # weight_load_path = 'best_lenet5_0.mdl'
    # model.load_state_dict(torch.load(weight_load_path))
    print('loaded from ckpt!')
    print('Verifying... Waiting...')
    path = r'D:\workspace\dataset\myfaces\hzh1.png'
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
    for i, p in enumerate(idx_to_class.values()):
        print(p,':\t','{:.6f}'.format(probs.cpu()[0][i].double().item()))
    print('\nThe Result: He is', idx_to_class[pred.item()])


if __name__ == '__main__':

    train()
    # mytest()





# 保存
# torch.save(model, '\model.pkl')
# # 加载
# model = torch.load('\model.pkl')