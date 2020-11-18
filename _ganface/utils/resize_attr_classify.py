'''
按attractive（3）、gender（19）, heavy_makeup（21）,young（41）四个属性划分数据集
'''

from imageio import imread, imsave
import cv2
import glob, os
from tqdm import tqdm
import shutil

read_path = '../datasets/CelebA/cropdata1'
save_path = '../datasets/CelebA/cropdata_resize'
if not os.path.exists(save_path):
    os.mkdir(save_path)
# gender
data_dir_gender = '../datasets/CelebA/Img/Gender'  # 将整理好的数据放在这个文件夹下
male_dir_gender = '../datasets/CelebA/Img/Gender/Male'
female_dir_gender = '../datasets/CelebA/Img/Gender/Female'
if not os.path.exists(data_dir_gender):
    os.mkdir(data_dir_gender)
if not os.path.exists(male_dir_gender):
    os.mkdir(male_dir_gender)
if not os.path.exists(female_dir_gender):
    os.mkdir(female_dir_gender)


data_dir_Attractive = '../datasets/CelebA/Img/Attractive'  # 将整理好的数据放在这个文件夹下
dir_Attractive = '../datasets/CelebA/Img/Attractive/Attractive'
dir_unAttractive = '../datasets/CelebA/Img/Attractive/Unattractive'
if not os.path.exists(data_dir_Attractive):
    os.mkdir(data_dir_Attractive)
if not os.path.exists(dir_Attractive):
    os.mkdir(dir_Attractive)
if not os.path.exists(dir_unAttractive):
    os.mkdir(dir_unAttractive)

data_dir_makeup = '../datasets/CelebA/Img/Makeup'  # 将整理好的数据放在这个文件夹下
heavy_dir_makeup = '../datasets/CelebA/Img/Makeup/Heavy_Makeup'
light_dir_makeup = '../datasets/CelebA/Img/Makeup/Light_Makeup'
if not os.path.exists(data_dir_makeup):
    os.mkdir(data_dir_makeup)
if not os.path.exists(heavy_dir_makeup):
    os.mkdir(heavy_dir_makeup)
if not os.path.exists(light_dir_makeup):
    os.mkdir(light_dir_makeup)

data_dir_Age = '../datasets/CelebA/Img/Age'  # 将整理好的数据放在这个文件夹下
young_dir_Age = '../datasets/CelebA/Img/Age/Young'
nouong_dir_Age = '../datasets/CelebA/Img/Age/NotYong'
if not os.path.exists(data_dir_Age):
    os.mkdir(data_dir_Age)
if not os.path.exists(young_dir_Age):
    os.mkdir(young_dir_Age)
if not os.path.exists(nouong_dir_Age):
    os.mkdir(nouong_dir_Age)

Attr_type_Attractive = 3  # Male
Attr_type_Heavy_Makeup = 19 # Male
Attr_type_gender = 21  # Male
Attr_type_Age = 40  # Male



WIDTH = 224
HEIGHT = 224


def read_process_save(read_path, save_path):
    image = imread(read_path)
    h = image.shape[0]
    w = image.shape[1]
    if h > w:
        image = image[h // 2 - w // 2: h // 2 + w // 2, :, :]
    else:
        image = image[:, w // 2 - h // 2: w // 2 + h // 2, :]
    image = cv2.resize(image, (WIDTH, HEIGHT))
    imsave(save_path, image)

# read_process_save(read_path, save_path)
target_attractive = 'Attractive' #3
target_Heavy_Makeup = 'Heavy_Makeup' #19
target_gender = 'Male'   #21
target_Young = 'Young'  #40

with open('../datasets/CelebA/Anno/New_list_attr_celeba_test.txt', "r") as Attr_file:  #打开属性列表文档
    Attr_info = Attr_file.readlines()
    Attr_info = Attr_info[2:]
    index = 0
    for line in Attr_info:
        index += 1
        info = line.split()
        filename = info[0]
        filepath_old = os.path.join("../datasets/CelebA/cropdata_resize", filename)   #图片原始存储位置

        # filepath_old = os.path.join("D:/MissTang/EX/datasets/CelebA/Img/img_align_celeba", filename)   #图片原始存储位置
        if os.path.isfile(filepath_old):
            image = imread(filepath_old)
            ## _Attractive
            if int(info[Attr_type_Attractive]) == 1:
                save_path_attractive = os.path.join(dir_Attractive, filename)
                imsave(save_path_attractive, image)
                # read_process_save(os.path.join("D:/MissTang/EX/datasets/CelebA/cropdata1", filename),
                #                   os.path.join(dir_Attractive, filename))  # 有魅力
            else:
                save_path_unattractive = os.path.join(dir_unAttractive, filename)
                imsave(save_path_unattractive, image)
                # read_process_save(os.path.join("D:/MissTang/EX/datasets/CelebA/cropdata1", filename),
                #                   os.path.join(dir_unAttractive, filename))  #无魅力
            ## Heavy_Makeup
            if int(info[Attr_type_Heavy_Makeup]) == 1:
                save_path_makeup = os.path.join(heavy_dir_makeup, filename)
                imsave(save_path_makeup, image)
                # read_process_save(os.path.join("D:/MissTang/EX/datasets/CelebA/cropdata1", filename),
                #                   os.path.join(heavy_dir_makeup, filename))  # 化浓妆
            else:
                save_path_makeupli = os.path.join(light_dir_makeup, filename)
                imsave(save_path_makeupli, image)
                # read_process_save(os.path.join("D:/MissTang/EX/datasets/CelebA/cropdata1", filename),
                #                   os.path.join(light_dir_makeup, filename))  # 化淡妆

            ## gender
            if int(info[Attr_type_gender]) == 1:
                save_path_male = os.path.join(male_dir_gender, filename)
                imsave(save_path_male, image)
                # read_process_save(os.path.join("D:/MissTang/EX/datasets/CelebA/cropdata1", filename),
                #                   os.path.join(male_dir_gender, filename))  # 男
            else:
                save_path_female = os.path.join(female_dir_gender, filename)
                imsave(save_path_female, image)
                # read_process_save(os.path.join("D:/MissTang/EX/datasets/CelebA/cropdata1", filename),
                #                   os.path.join(female_dir_gender, filename))  # 女

            ## age
            if int(info[Attr_type_Age]) == 1:
                save_path_young = os.path.join(young_dir_Age, filename)
                imsave(save_path_young, image)
                # read_process_save(os.path.join("D:/MissTang/EX/datasets/CelebA/cropdata1", filename),
                #                   os.path.join(young_dir_Age, filename))  # young
            else:
                save_path_noyoung = os.path.join(nouong_dir_Age, filename)
                imsave(save_path_noyoung, image)
                # read_process_save(os.path.join("D:/MissTang/EX/datasets/CelebA/cropdata1", filename),
                #                   os.path.join(nouong_dir_Age, filename))  # not young
