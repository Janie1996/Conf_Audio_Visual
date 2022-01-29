# -*- coding: UTF-8 -*-
"""
@file:wj_2022_01_11_proposed_featureSave.py
@author: Wei Jie
@date: 2022/1/11
@description:只有原始音频数据特征，特征直接加载到内存中，只使用DSCASA，训练完保存特征
"""
import random
import sys
import os
import warnings

from torch.backends import cudnn

sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])
warnings.filterwarnings('ignore')

import time
from torch.optim import lr_scheduler
import argparse
import torch
import torch.nn as nn
from AudioDataset import wj_2021_12_27_utils as utils
from AudioDataset import wj_2022_01_11_dataload as load_dataset
from AudioModels import wj_2022_01_08_DSCASA_Transformer as Model
import pickle
import numpy as np

# np.random.seed(12)
# torch.manual_seed(12)  #为CPU设置种子用于生成随机数，以使得结果是确定的   　　
# torch.cuda.manual_seed(12) #为当前GPU设置随机种子；  　　
# cudnn.deterministic = True

num=0

parser = argparse.ArgumentParser()

parser.add_argument('--audioFeatureRoot', default='DATA/2021_12_31_name_delta_melspectrogram_dict.pickle', help='root-path for audio')
parser.add_argument('--videoRoot', default='E:/Dataset/IEMOCAP_full_release/', help='root-path for video')
parser.add_argument('--testNum',type=int,default=1106,help='test dataset number')
parser.add_argument('--batchSize', type=int, default=128, help='train batch size')
parser.add_argument('--nClasses', type=int, default=4, help='# of classes in source domain')
parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.08, help='learning rate, default=0.0002')
parser.add_argument('--step_size', type=float, default=80, help='step of learning rate changes')
parser.add_argument('--gamma', type=float, default=0.5, help='weight decay of learning rate')
parser.add_argument('--cudaNum',  default='0', help='the cuda number to be used')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum　(default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--input_channel',  default=3, type=int,help='the channel number of input feature')
parser.add_argument('--length',  default=5, type=int,help='the clip number of input audio feature')
parser.add_argument('--frameWin',  default=224, type=int,help='the frame number of each input melspec feature map')


opt = parser.parse_args()
print(opt)

def train(train_loader,model,criterion,optimizer,scheduler):

    model.train(True)
    right,right1,right2=0,0,0
    all=0
    print(scheduler.get_lr())


    for i, (video, index) in enumerate(train_loader):
        video = video.float().cuda()  # batch,N,3,224,224
        label = index.cuda()

        pre,fea,score = model(video)

        loss = criterion(pre,label)

        _, preds = torch.max(pre.data, 1)
        right += float(torch.sum(preds == label.data))

        # _, preds = torch.max(pre1.data, 1)
        # right1 += float(torch.sum(preds == label.data))
        #
        # _, preds = torch.max(pre2.data, 1)
        # right2 += float(torch.sum(preds == label.data))

        all += label.size(0)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
    scheduler.step()
    print('Train *Prec@Audio {:.3f};  {:.3f};  {:.3f}  '.format(right/all,right1/all,right2/all))

from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
import numpy as np


def test_UA_WA(test_loader,model):

    model.eval()
    right, right1, right2 = 0, 0, 0
    all=0

    trace_y=[]
    trace_pre_y,trace_pre_y2=[],[]

    with torch.no_grad():
        for i, (video, index) in enumerate(test_loader):
            video = video.float().cuda()  # batch,10,3,224,224
            label = index.cuda()

            trace_y.append(label.cpu().detach().numpy())

            pre, fea,score = model(video)

            _, preds = torch.max(pre.data, 1)
            right += float(torch.sum(preds == label.data))
            trace_pre_y.append(preds.cpu().detach().numpy())

            # _, preds = torch.max(pre1.data, 1)
            # right1 += float(torch.sum(preds == label.data))
            #
            # _, preds = torch.max(pre2.data, 1)
            # right2 += float(torch.sum(preds == label.data))
            # trace_pre_y2.append(preds.cpu().detach().numpy())


            all += label.size(0)

        trace_y = np.concatenate(trace_y)
        trace_pre_y = np.concatenate(trace_pre_y)
        # trace_pre_y2 = np.concatenate(trace_pre_y2)

        weighted_accuracy = accuracy_score(trace_y,trace_pre_y)
        unweighted_accuracy = balanced_accuracy_score(trace_y,trace_pre_y)

        # weighted_accuracy2 = accuracy_score(trace_y,trace_pre_y2)
        # unweighted_accuracy2 = balanced_accuracy_score(trace_y,trace_pre_y2)

    print('Test *Prec@Audio {:.3f};  {:.3f};  {:.3f}  '.format(right/all,right1/all,right2/all))

    return max(right/all,right1/all,right2/all),unweighted_accuracy


def main():
    global num

    name_emotionLabel = pickle.load(open('DATA/name_emotionLabel_dict.pickle', 'rb'))

    session='Ses01'
    man='_F'
    num+=0

    #train_list, test_list = utils.leaveOneSession(name_emotionLabel, session)
    #train_list, test_list = utils.leaveOneSpeaker(name_emotionLabel,session,man)

    train_list, test_list = utils.randomSplit(length=len(name_emotionLabel),testNum=opt.testNum)
    train_loader, test_loader, feature_loader  = load_dataset.AudioLoadData(opt.audioFeatureRoot,train_list, test_list, name_emotionLabel,
                                                                           opt.batchSize,framelen=opt.length,winlen=opt.frameWin)



    model = Model.inno_model1(input_channel=opt.input_channel,classes=opt.nClasses,height=224,width=40)

    # checkpoint = torch.load('Checkpoint/Audio_seed_6099_0.6862567811934901_DSCASA.pth')
    # test_ss_0.484629294755877_DSCASA.pth
    # checkpoint = torch.load('Checkpoint/test_drop_DSCASA.pth')
    # model.load_state_dict(checkpoint)



    ''' Loss & Optimizer '''
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), opt.lr,momentum=opt.momentum, weight_decay=opt.weight_decay)
    scheduler = lr_scheduler.MultiStepLR(optimizer, [80,160,240], gamma=opt.gamma)

    # optimizer = torch.optim.Adam(params=model.parameters(),lr=opt.lr,weight_decay=opt.weight_decay)
    # optimizer = torch.optim.RMSprop(params=model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    model.cuda()

    best_wa = 0
    best_ua = 0

    acc = 0
    acc1 = 0

    for epoch in range(opt.niter):
        print("Epoch:",epoch)
        # start_time = time.time()
        train(train_loader,model,criterion,optimizer,scheduler)
        acc, acc1 = test_UA_WA(test_loader, model)
        # print("time:",time.time()-start_time)
        if (acc > best_wa):
            best_wa = acc
            model.eval()
            state_dict1=model.state_dict()
            torch.save(state_dict1, "Checkpoint/test_drop_DSCASA"+str(num)+".pth")
            # model.load_state_dict(state_dict1)
            # acc, acc1 = test_UA_WA(test_loader, model)
        if (acc1 > best_ua):
            best_ua = acc1
        print("best_wa:", best_wa, "best_ua:", best_ua)
        # checkpoint = torch.load('Checkpoint/test_drop_DSCASA.pth')
        # model.load_state_dict(checkpoint)
        # acc, acc1 = test_UA_WA(test_loader, model)

    print("end best_wa:", best_wa, "best_ua:", best_ua)
    # torch.save(state_dict1, "Checkpoint/test_drop_DSCASA.pth")
    # if(best_wa>0.68):
    #     torch.save(state_dict1, "Checkpoint/Audio_seed_6099_"+str(best_wa)+"_DSCASA.pth")#, _use_new_zipfile_serialization=False)

    file = open('DATA/log_audioOnly.txt', 'a')
    file.write(str(best_wa)+'  '+str(best_ua))
    file.write('\n')
    file.close()
    num+=1

if __name__=="__main__":

    main()
    file = open('DATA/log_audioOnly.txt', 'w')
    file.close()

    opt.cudaNum = list(map(int, opt.cudaNum.split(',')))
    for i in range(10):
        main()