#!/user/bin/env python    
#-*- coding:utf-8 -*- 

from torchvision import transforms

from matplotlib import pyplot as plt
from PIL import Image
import sys
sys.path.append("..")
import warnings
warnings.filterwarnings("ignore") # 忽略警告


transform1 = transforms.Compose([
        transforms.Scale(32),
        transforms.ToTensor(),
    ])

path = r'D:\workspace\dataset\myfaces\test\QC\1.png'
img = Image.open(path)
img = transform1(img)
print(img.shape)
# R = img[0]
# G = img[1]
# B = img[2]
#
# img[0]=0.299*R+0.587*G+0.114*B
# img = img[0]
# img = img.view(1,224,224)
# img = torch.squeeze(img)
# print(img.shape)
img = img.permute(1,2,0)
img = img.data.numpy()

plt.imshow(img)
plt.show()
