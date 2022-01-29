# -*- coding: UTF-8 -*-
"""
@file:wj_2022_01_01_dataload.py
@author: Wei Jie
@date: 2022/1/1
@description: 数据集加载类
"""
from __future__ import print_function
import sys
import os

import cv2

sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])


import numpy as np

import torch
import torch.utils.data
import torchvision.transforms as transforms
from VideoDataset import wj_2022_01_01_dataset as dataset



def VideoLoadData(datadir, train_list, test_list, name_emotion,batchsize,length):

    train_dataset = dataset.VideoDataset(
        video_root=datadir,
        video_list=train_list,
        name_emotin=name_emotion,
        length= length,
        transform=transforms.Compose([transforms.Resize((224,128)),
                                      transforms.ToTensor()]),
    )

    test_dataset = dataset.VideoDataset(
        video_root=datadir,
        video_list=test_list,
        name_emotin=name_emotion,
        length=length,
        transform=transforms.Compose([transforms.Resize((224,128)), transforms.ToTensor()]),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batchsize, shuffle=True)


    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batchsize, shuffle=False)

    return train_loader,test_loader



if __name__ == '__main__':

    import pickle
    import random

    """audio"""

    # datadir='E:/Dataset/IEMOCAP_full_release/mfccfeatures/'
    # name_emotionLabel=pickle.load(open('../Data/name_emotionLabel_dict.pickle','rb'))
    #
    # test_list=random.sample(range(0,len(name_emotionLabel)),int(553*3))
    # train_list=list(set(np.arange(len(name_emotionLabel)))-set(test_list))
    #
    # batchsize=16
    #
    # train_loader,test_loader = AudioLoadData(datadir,train_list, test_list,name_emotionLabel, batchsize)
    #
    #
    # for i, (video, index) in enumerate(test_loader):
    #     audio=video.cuda()  # batch,10,3,224,224
    #     label=index.cuda()
    #     print(i)
    #     print()

    """video"""

    datadir = 'E:/Dataset/IEMOCAP_full_release/newallavi/'
    name_emotionLabel = pickle.load(open('../Data/name_emotionLabel_dict.pickle', 'rb'))

    test_list = random.sample(range(0, len(name_emotionLabel)), int(553 * 3))
    train_list = list(set(np.arange(len(name_emotionLabel))) - set(test_list))

    batchsize = 16

    train_loader, test_loader = VideoLoadData(datadir, train_list, test_list, name_emotionLabel, batchsize)

    for i, (video, index) in enumerate(test_loader):
        video = video.cuda()  # batch,10,3,224,224
        label = index.cuda()
        print(i)
        print()