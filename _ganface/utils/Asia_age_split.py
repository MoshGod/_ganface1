#!/user/bin/env python
#-*- coding:utf-8 -*-


'''
1. read the whole files under a certain folder
2. chose 10000 files randomly
3. copy them to another folder and save
'''

import shutil


def copyFile(fileDir, tarDir):
    # pathDir = os.listdir(fileDir)
    # 文件权限
    filename = open(r'D:\2020SIT\2020秋季\人脸年龄数据集\megaage_asian\megaage_asian\list\train_name.txt','r')
    labels = open(r'D:\2020SIT\2020秋季\人脸年龄数据集\megaage_asian\megaage_asian\list\train_age.txt','r')
    age = ''
    for file, label in zip(filename.readlines(), labels.readlines()):
        if 0 <= int(label) < 10:
            age = '10'
        elif 10 <= int(label) < 20:
            age = '20'
        elif 20 <= int(label) < 30:
            age = '30'
        elif 30 <= int(label) < 40:
            age = '40'
        elif 40 <= int(label) < 50:
            age = '50'
        elif 50 <= int(label) < 60:
            age = '60'
        else:
            age = '70'

        shutil.copyfile(fileDir + '\\' + file[:-1], tarDir + '\\{}\\'.format(age) + file[:-1])

    filename.close()
    labels.close()


if __name__ == '__main__':
    fileDir = r'D:\2020SIT\2020秋季\人脸年龄数据集\megaage_asian\megaage_asian\train'
    tarDir = r'D:\2020SIT\2020秋季\人脸年龄数据集\megaage_asian\Asia_age'
    copyFile(fileDir, tarDir)

