#!/user/bin/env python    
#-*- coding:utf-8 -*- 


'''
1. read the whole files under a certain folder
2. chose 10000 files randomly
3. copy them to another folder and save
'''
import os, random, shutil
from normal import mkdir


def copyFile(fileDir, tarDir):
    pathDir = os.listdir(fileDir)
    sample = random.sample(pathDir, 200)

    for name in sample:
        shutil.copyfile(fileDir + name, tarDir + name)
        os.remove(fileDir + name)


if __name__ == '__main__':
    fileDir = r'D:\workspace\dataset\non_Asia_age_4\train\age_55_94\\'
    tarDir = r'D:\workspace\dataset\non_Asia_age_4\test\age_55_94\\'
    # dir_list1 = ['10_14', '15_19', '20_24', '25_29']
    # dir_list2 = ['30_34', '35_39', '40_44', '45_49']
    # dir_list3 = ['50_54', '55_59', '60_94']
    # # dir_list = [dir_list1, dir_list2, dir_list3]
    # dir_list = dir_list1 + dir_list2 + dir_list3
    # # dir_dst = ['10-29', '30-49', '50-94']
    # for dst in dir_list:
    #     # mkdir(tarDir+'\\age_'+dst)
    #     copyFile(fileDir+'\\age_'+dst+'\\', tarDir+'\\age_'+dst+'\\')
    copyFile(fileDir, tarDir)
