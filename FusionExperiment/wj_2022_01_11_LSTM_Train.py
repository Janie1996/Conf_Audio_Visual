# -*- coding: UTF-8 -*-
"""
@file:wj_2022_01_11_LSTM_Train.py
@author: Wei Jie
@date: 2022/1/11
@description:
"""


import sys
import os

sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])

import warnings

import pretrainedmodels
warnings.filterwarnings('ignore')

import time
from torch.optim import lr_scheduler
import argparse
import torch
import torch.nn as nn
from AudioDataset import wj_2021_12_27_utils as utils
from AudioDataset import wj_2021_12_27_dataload as load_dataset_audio
from VideoDataset import wj_2022_01_02_dataload as load_dataset_video
# from DomainModels import wj_2022_01_08_ResNet_Domain as DomainModel
from DomainModels import wj_2022_01_17_proposed_Domain as DomainModel
from AudioModels import wj_2022_01_08_DSCASA_Transformer as AudioModel
from FeatureFusionModels import wj_2022_01_08_LSTM as FusionModel
import pickle

parser = argparse.ArgumentParser()

parser.add_argument('--audioFeatureRoot', default='DATA/2022_01_03_name_logfbank_dict_40d.pickle', help='root-path for audio')
parser.add_argument('--videoRoot', default='E:/Dataset/IEMOCAP_full_release/faceallavi/', help='root-path for video')
parser.add_argument('--testNum',type=int,default=1106,help='test dataset number')
parser.add_argument('--batchSize', type=int, default=16, help='train batch size')
parser.add_argument('--nClasses', type=int, default=4, help='# of classes in source domain')
parser.add_argument('--niter', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')
parser.add_argument('--step_size', type=float, default=10, help='step of learning rate changes')
parser.add_argument('--gamma', type=float, default=0.5, help='weight decay of learning rate')
parser.add_argument('--cudaNum',  default='0', help='the cuda number to be used')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum　(default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--input_channel',  default=1, type=int,help='the channel number of input feature')
parser.add_argument('--length',  default=5, type=int,help='the clip number of input audio feature')
parser.add_argument('--frameWin',  default=224, type=int,help='the frame number of each input melspec feature map')
parser.add_argument('--featureDim',  default=40, type=int,help='the feature dimension of audio input')


opt = parser.parse_args()
print(opt)

def train(train_loader_audio,train_loader_video,model_D,criterion_D_E,criterion_D_D,optimizer_D,scheduler_D,model_A,criterion_A,optimizer_A,scheduler_A,model_F,criterion_F,optimizer_F,scheduler_F):


    model_D.train(True)
    model_A.train(True)
    model_F.train(False)



    right_d = 0
    right, right1, right2, right3, right4 = 0, 0, 0,0 ,0

    allLoss = 0
    all_e=0
    all = 0
    print(scheduler_F.get_lr())


    for i, data in enumerate(zip(train_loader_audio, train_loader_video)):

        loss=0

        # Domain
        audio_video = np.concatenate((data[0][0], data[1][0]), axis=0)
        label_emo = np.append(data[0][1], data[1][1])

        a = np.ones(len(data[0][1]))
        b = np.zeros(len(data[1][1]))
        label_domain = np.append(a, b)

        state = np.random.get_state()
        np.random.shuffle(audio_video)

        np.random.set_state(state)
        np.random.shuffle(label_emo)

        np.random.set_state(state)
        np.random.shuffle(label_domain)

        audio_video = torch.from_numpy(audio_video).float().cuda()
        label_domain = torch.from_numpy(label_domain).float().cuda()
        label_emo = torch.from_numpy(label_emo).cuda()

        emo,dom,fea_d,emo_d = model_D(audio_video)

        _, preds = torch.max(dom.data, 1)
        right_d += float(torch.sum(preds == label_domain.data))

        lossD = criterion_D_D(dom, label_domain.unsqueeze(1))

        allLoss += lossD.item()

        all += label_emo.size(0)

        optimizer_D.zero_grad()
        lossD.backward()
        optimizer_D.step()

        # Domain Audio Fusion
        if(len(data[0][0])):

            # Domain
            audio = data[0][0].float().cuda()  # batch,10,3,224,224
            label = data[0][1].cuda()
            emo, dom, fea_d,emo_d = model_D(audio)
            emo_d = emo_d.unsqueeze(3)

            _, preds = torch.max(emo.data, 1)
            right += float(torch.sum(preds == label.data))

            lossE = criterion_D_E(emo, label)

            optimizer_D.zero_grad()
            lossE.backward()
            optimizer_D.step()

            all_e += label.size(0)

            # Audio
            emo, fea_a, emo_a = model_A(audio)
            emo_a = emo_a.unsqueeze(3)

            _, preds = torch.max(emo.data, 1)
            right1 += float(torch.sum(preds == label.data))

            lossE = criterion_A(emo, label)

            optimizer_A.zero_grad()
            lossE.backward()
            optimizer_A.step()

            # Fusion
            audio = np.concatenate((fea_a.cpu().detach().numpy(), fea_d.cpu().detach().numpy()), axis=2)
            audio = torch.from_numpy(audio).cuda()

            emo_audio = np.concatenate((emo_a.cpu().detach().numpy(), emo_d.cpu().detach().numpy()), axis=3)
            maxpool = torch.nn.MaxPool2d((1,2))
            emo_audio = torch.from_numpy(emo_audio)
            emo_audio = maxpool(emo_audio).squeeze(3).cuda()

            emo,emo1,emo2= model_F(audio,emo_audio)

            _, preds = torch.max(emo.data, 1)
            right2 += float(torch.sum(preds == label.data))

            _, preds = torch.max(emo1.data, 1)
            right3 += float(torch.sum(preds == label.data))

            _, preds = torch.max(emo2.data, 1)
            right4 += float(torch.sum(preds == label.data))

            lossE = criterion_F(emo, label)

            # optimizer_F.zero_grad()
            # lossE.backward()
            # optimizer_F.step()

    scheduler_D.step()
    scheduler_A.step()
    # scheduler_F.step()

    print('Train *Prec@Audio {:.3f};  {:.3f}; @Fusion {:.3f} ; {:.3f} ;{:.3f} ;Prec@Domain {:.3f}; Loss {:.3f}  '.format(right/all_e,right1/all_e,right2/all_e,right3/all_e,right4/all_e,right_d / all,allLoss / all))


from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
import numpy as np


def test_UA_WA(test_loader,model_D,model_A,model_F):

    model_D.eval()
    model_A.eval()
    model_F.eval()


    right, right1, right2, right3, right4 = 0, 0, 0,0,0
    all=0

    trace_y=[]
    trace_pre_y,trace_pre_y1,trace_pre_y2,trace_pre_y3,trace_pre_y4=[],[],[],[],[]

    with torch.no_grad():
        for i, (video, index) in enumerate(test_loader):
            video = video.float().cuda()  # batch,10,3,224,224
            label = index.cuda()

            trace_y.append(label.cpu().detach().numpy())

            pre,dom,fea_d,emo_d = model_D(video)
            pre1,fea_a,emo_a = model_A(video)

            emo_d = emo_d.unsqueeze(3)
            emo_a = emo_a.unsqueeze(3)

            video = np.concatenate((fea_a.cpu().detach().numpy(), fea_d.cpu().detach().numpy()), axis=2)
            video = torch.from_numpy(video).cuda()

            emo_audio = np.concatenate((emo_a.cpu().detach().numpy(), emo_d.cpu().detach().numpy()), axis=3)
            maxpool = torch.nn.MaxPool2d((1, 2))
            emo_audio = torch.from_numpy(emo_audio)
            emo_audio = maxpool(emo_audio).squeeze(3).cuda()

            pre2,pre3,pre4 = model_F(video,emo_audio)

            _, preds = torch.max(pre.data, 1)
            right += float(torch.sum(preds == label.data))
            trace_pre_y.append(preds.cpu().detach().numpy())

            _, preds = torch.max(pre1.data, 1)
            right1 += float(torch.sum(preds == label.data))
            trace_pre_y1.append(preds.cpu().detach().numpy())

            _, preds = torch.max(pre2.data, 1)
            right2 += float(torch.sum(preds == label.data))
            trace_pre_y2.append(preds.cpu().detach().numpy())

            _, preds = torch.max(pre3.data, 1)
            right3 += float(torch.sum(preds == label.data))
            trace_pre_y3.append(preds.cpu().detach().numpy())

            _, preds = torch.max(pre4.data, 1)
            right4 += float(torch.sum(preds == label.data))
            trace_pre_y4.append(preds.cpu().detach().numpy())

            all += label.size(0)

        trace_y = np.concatenate(trace_y)
        trace_pre_y = np.concatenate(trace_pre_y)
        trace_pre_y1 = np.concatenate(trace_pre_y1)
        trace_pre_y2 = np.concatenate(trace_pre_y2)
        trace_pre_y3 = np.concatenate(trace_pre_y3)
        trace_pre_y4 = np.concatenate(trace_pre_y4)


        weighted_accuracy = accuracy_score(trace_y,trace_pre_y)
        unweighted_accuracy = balanced_accuracy_score(trace_y,trace_pre_y)

        weighted_accuracy1 = accuracy_score(trace_y, trace_pre_y1)
        unweighted_accuracy1 = balanced_accuracy_score(trace_y, trace_pre_y1)

        weighted_accuracy2 = accuracy_score(trace_y,trace_pre_y2)
        unweighted_accuracy2 = balanced_accuracy_score(trace_y,trace_pre_y2)

        weighted_accuracy3 = accuracy_score(trace_y, trace_pre_y3)
        unweighted_accuracy3 = balanced_accuracy_score(trace_y, trace_pre_y3)

        weighted_accuracy4 = accuracy_score(trace_y, trace_pre_y4)
        unweighted_accuracy4 = balanced_accuracy_score(trace_y, trace_pre_y4)

    print('Test *Prec@Audio {:.3f};  {:.3f}; @Fusion {:.3f}; {:.3f};  {:.3f};  '.format(right/all,right1/all,right2/all,right3/all,right4/all))

    return max(right/all,right1/all,right2/all,right3/all,right4/all),max(unweighted_accuracy,unweighted_accuracy1,unweighted_accuracy2,unweighted_accuracy3,unweighted_accuracy4)

def trainAudio(train_loader,model,criterion,optimizer,scheduler):

    model.train(True)
    right,right1,right2=0,0,0
    all=0
    print(scheduler.get_lr())
    for i, (video, index) in enumerate(train_loader):
        video = video.float().cuda()  # batch,N,3,224,224
        label = index.cuda()

        pre,fea,emo_a = model(video)

        loss = criterion(pre,label)

        _, preds = torch.max(pre.data, 1)
        right += float(torch.sum(preds == label.data))

        all += label.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()
    print('Train *Prec@Audio {:.3f} '.format(right/all))


def trainLSTM(test_loader,model_D,model_A,model_F,criterion_F,optimizer_F,scheduler_F):

    # model_D.train(True)
    # model_A.train(True)
    model_D.eval()
    model_A.eval()
    model_F.train(True)


    right, right1, right2, right3, right4 = 0, 0, 0,0,0
    all=0

    trace_y=[]
    trace_pre_y,trace_pre_y1,trace_pre_y2,trace_pre_y3,trace_pre_y4=[],[],[],[],[]

    print(scheduler_F.get_lr())

    for i, (video, index) in enumerate(test_loader):
        video = video.float().cuda()  # batch,10,3,224,224
        label = index.cuda()

        trace_y.append(label.cpu().detach().numpy())

        pre, dom, fea_d, emo_d = model_D(video)
        pre1, fea_a, emo_a = model_A(video)

        emo_d = emo_d.unsqueeze(3)
        emo_a = emo_a.unsqueeze(3)

        video = np.concatenate((fea_a.cpu().detach().numpy(), fea_d.cpu().detach().numpy()), axis=2)
        video = torch.from_numpy(video).cuda()

        emo_audio = np.concatenate((emo_a.cpu().detach().numpy(), emo_d.cpu().detach().numpy()), axis=3)
        maxpool = torch.nn.MaxPool2d((1, 2))
        emo_audio = torch.from_numpy(emo_audio)
        emo_audio = maxpool(emo_audio).squeeze(3).cuda()

        pre2, pre3, pre4 = model_F(video, emo_audio)

        _, preds = torch.max(pre2.data, 1)
        right2 += float(torch.sum(preds == label.data))
        trace_pre_y2.append(preds.cpu().detach().numpy())

        _, preds = torch.max(pre3.data, 1)
        right3 += float(torch.sum(preds == label.data))
        trace_pre_y3.append(preds.cpu().detach().numpy())

        _, preds = torch.max(pre4.data, 1)
        right4 += float(torch.sum(preds == label.data))
        trace_pre_y4.append(preds.cpu().detach().numpy())

        lossE = criterion_F(pre3, label)+criterion_F(pre4, label)

        optimizer_F.zero_grad()
        lossE.backward()
        optimizer_F.step()

        all += label.size(0)

    trace_y = np.concatenate(trace_y)

    trace_pre_y2 = np.concatenate(trace_pre_y2)
    trace_pre_y3 = np.concatenate(trace_pre_y3)
    trace_pre_y4 = np.concatenate(trace_pre_y4)

    weighted_accuracy2 = accuracy_score(trace_y, trace_pre_y2)
    unweighted_accuracy2 = balanced_accuracy_score(trace_y, trace_pre_y2)

    weighted_accuracy3 = accuracy_score(trace_y, trace_pre_y3)
    unweighted_accuracy3 = balanced_accuracy_score(trace_y, trace_pre_y3)

    weighted_accuracy4 = accuracy_score(trace_y, trace_pre_y4)
    unweighted_accuracy4 = balanced_accuracy_score(trace_y, trace_pre_y4)

    scheduler_F.step()
    print('Train @Fusion {:.3f}; {:.3f};  {:.3f};  '.format(right2/all,right3/all,right4/all))

    return max(right/all,right1/all,right2/all,right3/all,right4/all),max(unweighted_accuracy2,unweighted_accuracy3,unweighted_accuracy4)

def test_UA_WA_(test_loader,model):

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

            pre, dom, fea, emo_all = model(video)

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
        #
        # weighted_accuracy2 = accuracy_score(trace_y,trace_pre_y2)
        # unweighted_accuracy2 = balanced_accuracy_score(trace_y,trace_pre_y2)

    print('Test *Prec@Audio {:.3f};  {:.3f};  {:.3f}  '.format(right/all,right1/all,right2/all))

    return max(right/all,right1/all,right2/all),unweighted_accuracy

def main():


    name_emotionLabel = pickle.load(open('DATA/name_emotionLabel_dict.pickle', 'rb'))
    name_emotionLabel_video = pickle.load(open('DATA/name_emotionLable_dict_noblack.pickle', 'rb'))

    session='Ses01'
    man='_F'

    #train_list, test_list = utils.leaveOneSession(name_emotionLabel, session)
    #train_list, test_list = utils.leaveOneSpeaker(name_emotionLabel,session,man)

    train_list, test_list = utils.randomSplit(length=len(name_emotionLabel),testNum=opt.testNum)
    train_loader_audio, test_loader_audio = load_dataset_audio.AudioLoadData(opt.audioFeatureRoot,train_list, test_list, name_emotionLabel,
                                                                           opt.batchSize,framelen=opt.length,winlen=opt.frameWin)


    train_loader_video, test_loader_video = load_dataset_video.VideoLoadData(opt.videoRoot, list(np.arange(len(name_emotionLabel_video))), [1],
                                                           name_emotionLabel_video, opt.batchSize,
                                                           length=opt.length,channel=1,feature=40)


    model_D = DomainModel.inno_model1(input_channel=opt.input_channel,classes=opt.nClasses,height=opt.frameWin,width=opt.featureDim)
    checkpoint = torch.load('Checkpoint/Domain_40_2_CASA_1.pth')
    # model_D = DomainModel.resnet18(classes=opt.nClasses,channel=1)
    # checkpoint = torch.load('Checkpoint/Domain_40_6_Resnet_1.pth')
    model_D.load_state_dict(checkpoint)
    # model_D.cuda()
    # a1,a2=test_UA_WA_(test_loader_audio,model_D)
    # 预训练参数加载
    # args = pretrainedmodels.__dict__['resnet18'](pretrained='imagenet').state_dict()
    # model_state_dict = model_D.state_dict()
    # for key in args:
    #     if key in model_state_dict:
    #         model_state_dict[key] = args[key]
    # model_D.load_state_dict(model_state_dict)


    model_A = AudioModel.inno_model1(input_channel=opt.input_channel,classes=opt.nClasses,height=opt.frameWin,width=opt.featureDim)
    checkpoint = torch.load('Checkpoint/test_drop_DSCASA7_1.pth')
    model_A.load_state_dict(checkpoint)

    model_F = FusionModel.EmotionRecog(classes=opt.nClasses,featureDim=768+768)

    ''' Loss & Optimizer '''


    criterion_D_E = nn.CrossEntropyLoss()
    criterion_D_D = nn.BCEWithLogitsLoss()
    optimizer_D = torch.optim.SGD(filter(lambda p: p.requires_grad, model_D.parameters()), 0.0005,momentum=opt.momentum, weight_decay=opt.weight_decay)
    scheduler_D = lr_scheduler.MultiStepLR(optimizer_D, [5,10,15], gamma=opt.gamma)

    criterion_A = nn.CrossEntropyLoss()
    optimizer_A = torch.optim.SGD(filter(lambda p: p.requires_grad, model_A.parameters()), opt.lr,
                                  momentum=opt.momentum, weight_decay=opt.weight_decay)
    scheduler_A = lr_scheduler.MultiStepLR(optimizer_A, [40,80,120], gamma=opt.gamma)

    criterion_F = nn.CrossEntropyLoss()
    optimizer_F = torch.optim.SGD(filter(lambda p: p.requires_grad, model_F.parameters()), opt.lr,
                                  momentum=opt.momentum, weight_decay=opt.weight_decay)
    scheduler_F = lr_scheduler.MultiStepLR(optimizer_F, [40,80,120], gamma=opt.gamma)
    #lr_scheduler.StepLR(optimizer_F, step_size=opt.step_size, gamma=opt.gamma)

    # optimizer = torch.optim.Adam(params=model.parameters(),lr=opt.lr,weight_decay=opt.weight_decay)
    # optimizer = torch.optim.RMSprop(params=model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    model_D.cuda()
    model_A.cuda()
    model_F.cuda()

    best_wa = 0
    best_ua = 0

    acc = 0
    acc1 = 0

    for epoch in range(opt.niter):
        print("Epoch:",epoch)
        # start_time = time.time()
        # if(epoch<5):
        #     for au in range(10):
        #         trainAudio(train_loader_audio,model_A,criterion_A,optimizer_A,scheduler_A)
        #
        # train(train_loader_audio,train_loader_video,model_D,criterion_D_E,criterion_D_D,optimizer_D,scheduler_D,model_A,criterion_A,optimizer_A,scheduler_A,model_F,criterion_F,optimizer_F,scheduler_F)
        acc, acc1 = test_UA_WA(test_loader_audio, model_D,model_A,model_F)
        if (epoch > -1):
            model_F = FusionModel.EmotionRecog(classes=opt.nClasses, featureDim=768 + 768)
            optimizer_F = torch.optim.SGD(filter(lambda p: p.requires_grad, model_F.parameters()), opt.lr,
                                          momentum=opt.momentum, weight_decay=opt.weight_decay)
            scheduler_F = lr_scheduler.MultiStepLR(optimizer_F, [10, 20, 30], gamma=opt.gamma)
            model_F.cuda()
            for au in range(200):

                trainLSTM(train_loader_audio, model_D, model_A, model_F, criterion_F, optimizer_F, scheduler_F)
                acc, acc1 = test_UA_WA(test_loader_audio, model_D, model_A, model_F)
                if (acc > best_wa):
                    best_wa = acc
                if (acc1 > best_ua):
                    best_ua = acc1

        # print("time:",time.time()-start_time)
        if (acc > best_wa):
            best_wa = acc
        if (acc1 > best_ua):
            best_ua = acc1
        print("best_wa:", best_wa, "best_ua:", best_ua)
    print("end best_wa:", best_wa, "best_ua:", best_ua)

    file = open('DATA/log_audioOnly.txt', 'a')
    file.write(str(best_wa)+'  '+str(best_ua))
    file.write('\n')
    file.close()

if __name__=="__main__":

    main()
    file = open('DATA/log_audioOnly.txt', 'w')
    file.close()

    opt.cudaNum = list(map(int, opt.cudaNum.split(',')))
    for i in range(10):
        main()