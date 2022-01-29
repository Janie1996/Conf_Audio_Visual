# -*- coding: UTF-8 -*-
"""
@file:wj_2022_01_01_dataset.py
@author: Wei Jie
@date: 2022/1/1
@description: 继承dataset类，用于数据集读取；目前初始化只保存了数据路径
            dataload在训练过程中读取每个视频内容
            优： 不占用大量内存
            缺： 每一轮训练都要从硬盘读取数据，耗时
"""
import sys
import os

sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])


import cv2
from PIL import Image
import numpy as np
import torch
import torch.utils.data as data

try:
    import cPickle as pickle
except:
    import pickle

# 读取视频数据
def get_video(index,root,name_emotion,transform,length):

    #length=5

    names = list(name_emotion.keys())
    labels = list(name_emotion.values())
    index_name = names[index]   #视频name
    index_label = labels[index]  # label

    #index_name='Ses04M_script03_2_M020'
    vc = cv2.VideoCapture(root+ index_name + '.mp4')  # 读入视频文件
    video_length = int(vc.get(7))
    video = torch.ones(length, 3, 224, 128)

    num = 0
    # 均匀采样视频帧
    #print(index_name)
    if(video_length<length):
        for i in range(video_length):
            rval, frame = vc.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame.astype('uint8')).convert('RGB')
            frame = transform(frame).unsqueeze(0)
            video[i, :, :, :] = frame
            num +=1
        for j in range(num,length):
            video[j, :, :, :] = frame
    else:
        step = int(video_length/length)-1
        for i in range(length):
            #print(i)
            rval, frame = vc.read()

                        # cv2.imshow('OriginalPicture', frame)
                        # cv2.waitKey()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame.astype('uint8')).convert('RGB')
            frame = transform(frame).unsqueeze(0)
            video[i, :, :, :] = frame
            for j in range(step):
                rval, frame = vc.read()
    vc.release()

    # if(index==1):
    #     print(sum(sum(sum(sum(video)))))

    return video,index_label


# video数据集
class VideoDataset(data.Dataset):

    def __init__(self, video_root, video_list, name_emotin, length, transform=None):
        self.video_root = video_root
        self.video_list = video_list
        self.name_emotin = name_emotin
        self.transform = transform
        self.length = length

    def __getitem__(self, index):
       # print(self.video_list[index])
        video,label=get_video(self.video_list[index],self.video_root,self.name_emotin,self.transform,self.length)
        return video, label

    def __len__(self):
        return len(self.video_list)

if __name__=="__main__":

    print(1)
