# -*- coding: UTF-8 -*-
"""
@file:test.py
@author: Wei Jie
@date: 2022/1/1
@description:
"""
import pickle
import sys
import os

import cv2
import torch
from PIL import Image
import torchvision.transforms as transforms
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])


def get_video():
    transform = transforms.Compose([transforms.Resize((224, 128)),
                                    transforms.ToTensor()])


    length=5

    #index_name='Ses04M_script03_2_M020'
    vc = cv2.VideoCapture('E:/Dataset/IEMOCAP_full_release/faceallavi/Ses04M_script03_2_M020.mp4')  # 读入视频文件
    video_length = int(vc.get(7))
    video = torch.ones(length, 3, 224, 128)

    num = 0
    # 均匀采样视频帧
    #print(index_name)
    if(video_length<length):
        for i in range(video_length):
            rval, frame = vc.read()
            # cv2.imshow('OriginalPicture', frame)
            # cv2.waitKey()
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
            frame = transform(frame)#.unsqueeze(0)
            cv2.imshow('OriginalPicture', frame.numpy().swapaxes(0,1).swapaxes(1,2))
            # cv2.imshow('OriginalPicture', frame.numpy().transpose([1,2,0]))
            cv2.waitKey()

            video[i, :, :, :] = frame
            for j in range(step):
                rval, frame = vc.read()
    vc.release()

    # if(index==1):
    #     print(sum(sum(sum(sum(video)))))

    return video


if __name__ == "__main__":
    get_video()

    datadir='DATA/2021_12_28_name_melspectrogram_dict.pickle'
    allfeatures = list(pickle.load(open(datadir, 'rb')).values())

    datadir = 'DATA/name_melspectrogram_128_dict.pickle'
    allfeatures1 = list(pickle.load(open(datadir, 'rb')).values())

    print()