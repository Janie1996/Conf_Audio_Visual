# -*- coding: UTF-8 -*-
"""
@file:wj_2022_01_11_Resnet_featureSave.py
@author: Wei Jie
@date: 2022/1/11
@description:
"""

import sys
import os
import warnings

import pretrainedmodels

sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])
warnings.filterwarnings('ignore')

import time
from torch.optim import lr_scheduler
import argparse
import torch
import torch.nn as nn
from AudioDataset import wj_2021_12_27_utils as utils
from AudioDataset import wj_2021_12_27_dataload as load_dataset_audio
from VideoDataset import wj_2022_01_02_dataload as load_dataset_video
from DomainModels import wj_2022_01_08_ResNet_Domain as Model
# from DomainModels import wj_2022_01_17_proposed_Domain as Model
import pickle

num=0

parser = argparse.ArgumentParser()

parser.add_argument('--audioFeatureRoot', default='DATA/2022_01_03_name_logfbank_dict_40d.pickle', help='root-path for audio')
parser.add_argument('--videoRoot', default='E:/Dataset/IEMOCAP_full_release/faceallavi/', help='root-path for video')
parser.add_argument('--testNum',type=int,default=1106,help='test dataset number')
parser.add_argument('--batchSize', type=int, default=32, help='train batch size')
parser.add_argument('--nClasses', type=int, default=4, help='# of classes in source domain')
parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.005, help='learning rate, default=0.0002')
parser.add_argument('--step_size', type=float, default=80, help='step of learning rate changes')
parser.add_argument('--gamma', type=float, default=0.5, help='weight decay of learning rate')
parser.add_argument('--cudaNum',  default='0', help='the cuda number to be used')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum　(default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--input_channel',  default=1, type=int,help='the channel number of input feature')
parser.add_argument('--length',  default=5, type=int,help='the clip number of input audio feature')
parser.add_argument('--frameWin',  default=224, type=int,help='the frame number of each input melspec feature map')


opt = parser.parse_args()
print(opt)

def train(train_loader_audio,train_loader_video,model,criterion,criterion_D,optimizer,scheduler):
    model.train(True)
    right_d = 0
    right, right1, right2 = 0, 0, 0

    allLoss = 0
    all = 0
    print(scheduler.get_lr())
    for i, data in enumerate(zip(train_loader_audio, train_loader_video)):
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

        pre, dom, fea, emo_all = model(audio_video)

        _, preds = torch.max(pre.data, 1)
        right += float(torch.sum(preds == label_emo.data))

        # _, preds = torch.max(pre1.data, 1)
        # right1 += float(torch.sum(preds == label_emo.data))
        #
        # _, preds = torch.max(pre2.data, 1)
        # right2 += float(torch.sum(preds == label_emo.data))

        lossE = criterion(pre, label_emo)
        lossD = criterion_D(dom, label_domain.unsqueeze(1))

        # lossD = F.binary_cross_entropy_with_logits(preD.squeeze(),label_domain)

        allLoss += lossE.item()
        allLoss += lossD.item()

        loss = lossE + lossD


        preds = torch.round(torch.sigmoid(dom)).squeeze(1)
        right_d += float(torch.sum(preds == label_domain.data))

        all += label_emo.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()
    print('Train *Prec@Audio {:.3f};  {:.3f};  {:.3f} ; Prec@Domain {:.3f}; Loss {:.3f}  '.format(right/all,right1/all,right2/all,right_d / all,allLoss / all))

def trainAug(train_loader_audio,train_loader_video,model,criterion,criterion_D,optimizer,scheduler):
    model.train(True)
    right_d = 0
    right, right1, right2 = 0, 0, 0

    allLoss = 0
    all = 0
    # print(scheduler.get_lr())
    for i, data in enumerate(zip(train_loader_audio, train_loader_video)):
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

        pre, dom, fea, emo_all = model(audio_video)

        _, preds = torch.max(pre.data, 1)
        right += float(torch.sum(preds == label_emo.data))

        # _, preds = torch.max(pre1.data, 1)
        # right1 += float(torch.sum(preds == label_emo.data))
        #
        # _, preds = torch.max(pre2.data, 1)
        # right2 += float(torch.sum(preds == label_emo.data))

        lossE = criterion(pre, label_emo)
        lossD = criterion_D(dom, label_domain.unsqueeze(1))

        # lossD = F.binary_cross_entropy_with_logits(preD.squeeze(),label_domain)

        allLoss += lossE.item()
        allLoss += lossD.item()

        loss = lossE + lossD


        preds = torch.round(torch.sigmoid(dom)).squeeze(1)
        right_d += float(torch.sum(preds == label_domain.data))

        all += label_emo.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # scheduler.step()
    print('Train *Prec@Audio {:.3f};  {:.3f};  {:.3f} ; Prec@Domain {:.3f}; Loss {:.3f}  '.format(right/all,right1/all,right2/all,right_d / all,allLoss / all))


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
    global num

    name_emotionLabel = pickle.load(open('DATA/name_emotionLabel_dict.pickle', 'rb'))
    name_emotionLabel_video = pickle.load(open('DATA/name_emotionLable_dict_noblack.pickle', 'rb'))

    session='Ses01'
    man='_F'

    #train_list, test_list = utils.leaveOneSession(name_emotionLabel, session)
    #train_list, test_list = utils.leaveOneSpeaker(name_emotionLabel,session,man)

    train_list, test_list = utils.randomSplit(length=len(name_emotionLabel),testNum=opt.testNum)
    train_loader_audio, test_loader_audio = load_dataset_audio.AudioLoadData(opt.audioFeatureRoot,train_list, test_list, name_emotionLabel,
                                                                           opt.batchSize,framelen=opt.length,winlen=opt.frameWin)
    train_loader_audio1, test_loader_audio1 = load_dataset_audio.AudioLoadData(
        'DATA/2022_01_26_name_logfbank_dict_40d_1.pickle', train_list,
        test_list, name_emotionLabel,
        opt.batchSize, framelen=opt.length,
        winlen=opt.frameWin)

    train_loader_audio2, test_loader_audio2 = load_dataset_audio.AudioLoadData('DATA/2022_01_26_name_logfbank_dict_40d_2.pickle', train_list,
                                                                             test_list, name_emotionLabel,
                                                                             opt.batchSize, framelen=opt.length,
                                                                             winlen=opt.frameWin)
    train_loader_audio3, test_loader_audio3 = load_dataset_audio.AudioLoadData(
        'DATA/2022_01_26_name_logfbank_dict_40d_3.pickle', train_list,
        test_list, name_emotionLabel,
        opt.batchSize, framelen=opt.length,
        winlen=opt.frameWin)

    train_loader_audio4, test_loader_audio4 = load_dataset_audio.AudioLoadData(
        'DATA/2022_01_26_name_logfbank_dict_40d_4.pickle', train_list,
        test_list, name_emotionLabel,
        opt.batchSize, framelen=opt.length,
        winlen=opt.frameWin)


    train_loader_video, test_loader_video = load_dataset_video.VideoLoadData(opt.videoRoot, list(np.arange(len(name_emotionLabel_video))), [1],
                                                           name_emotionLabel_video, opt.batchSize,
                                                           length=opt.length,channel=opt.input_channel,feature=40)


    # model = Model.inno_model1(input_channel=opt.input_channel, classes=opt.nClasses, height=224, width=40)
    model=Model.resnet18(classes=opt.nClasses,channel=opt.input_channel)

    # 预训练参数加载
    args = pretrainedmodels.__dict__['resnet18'](pretrained='imagenet').state_dict()
    model_state_dict = model.state_dict()
    for key in args:
        if key in model_state_dict:
            model_state_dict[key] = args[key]
    model.load_state_dict(model_state_dict)

    ''' Loss & Optimizer '''
    criterion = nn.CrossEntropyLoss()
    criterion_D = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), opt.lr,momentum=opt.momentum, weight_decay=opt.weight_decay)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.step_size, gamma=opt.gamma)
    scheduler = lr_scheduler.MultiStepLR(optimizer, [10,20,30,40], gamma=opt.gamma)

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
        train(train_loader_audio,train_loader_video,model,criterion,criterion_D,optimizer,scheduler)
        train(train_loader_audio1, train_loader_video, model, criterion, criterion_D, optimizer, scheduler)
        train(train_loader_audio2, train_loader_video, model, criterion, criterion_D, optimizer, scheduler)
        train(train_loader_audio3, train_loader_video, model, criterion, criterion_D, optimizer, scheduler)
        train(train_loader_audio4, train_loader_video, model, criterion, criterion_D, optimizer, scheduler)
        acc, acc1 = test_UA_WA(test_loader_audio, model)
        # print("time:",time.time()-start_time)
        if (acc > best_wa):
            best_wa = acc
            model.eval()
            state_dict1 = model.state_dict()

            torch.save(state_dict1,
                           "Checkpoint/Domain_Resnet_" + str(num) + "_AV_6364.pth")
        if (acc1 > best_ua):
            best_ua = acc1
        print("best_wa:", best_wa, "best_ua:", best_ua)
    print("end best_wa:", best_wa, "best_ua:", best_ua)
    num+=1

    file = open('DATA/log_audioVisual_Domain_6364.txt', 'a')
    file.write(str(best_wa)+'  '+str(best_ua))
    file.write('\n')
    file.close()

if __name__=="__main__":


    file = open('DATA/log_audioVisual_Domain_6364.txt', 'w')
    file.close()

    opt.cudaNum = list(map(int, opt.cudaNum.split(',')))
    for i in range(5):
        main()