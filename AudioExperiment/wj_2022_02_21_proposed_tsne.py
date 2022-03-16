# -*- coding: UTF-8 -*-
"""
@file:wj_2022_02_21_proposed_tsne.py
@author: Wei Jie
@date: 2022/2/21
@description:
"""
import sys
import os

sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])

from time import time as t
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn import datasets
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import time



def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    # ax = plt.subplot(111)
    # ax = plt.figure()
    for c, i, target_name in zip('cmgy', range(label.shape[0]), ['ang', 'hap', 'neu', 'sad']):
        # c = plt.cm.rainbow(int(255 / 9 * label[i]))
        plt.scatter(data[label == i, 0], data[label == i, 1], c=c, s=5, label=target_name)
    # for i in range(data.shape[0]):
    #     c = plt.cm.rainbow(int(255 / 9 * label[i]))
    #     plt.text(data[i, 0], data[i, 1], str(label[i]),
    # color=plt.cm.Set1(label[i] / 10.),
    # color=c,
    # fontdict={'weight': 'bold', 'size': 9})
    plt.legend()
    plt.xticks([])
    plt.yticks([])
    # plt.xlabel('(a)')
    plt.xlabel(title, fontsize=18)
    return fig



def main1():

    data = np.random.randint(0,90000,(5,324))
    label = np.array([1,2,2,3,2])
    print(data.shape)
    # sc = StandardScaler()
    # data = sc.fit_transform(data)
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)

    result = tsne.fit_transform(data)

    fig = plot_embedding(result, label,
                         '(e)')
    plt.savefig('/tsne_.jpg')

    plt.show()


import argparse
import torch
import torch.nn as nn
from AudioDataset import wj_2021_12_27_utils as utils
from AudioDataset import wj_2022_01_26_dataload as load_dataset
from AudioModels import wj_2022_01_08_DSCASA_Transformer as Model
import pickle


num=0

parser = argparse.ArgumentParser()

parser.add_argument('--audioFeatureRoot', default='DATA/2022_01_03_name_logfbank_dict_40d.pickle', help='root-path for audio')
parser.add_argument('--videoRoot', default='E:/Dataset/IEMOCAP_full_release/', help='root-path for video')
parser.add_argument('--testNum',type=int,default=1106,help='test dataset number')
parser.add_argument('--batchSize', type=int, default=32, help='train batch size')
parser.add_argument('--nClasses', type=int, default=4, help='# of classes in source domain')
parser.add_argument('--niter', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.08, help='learning rate, default=0.0002')
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


from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import numpy as np


def test_UA_WA(test_loader,model):

    model.eval()
    right, right1, right2 = 0, 0, 0
    all=0

    trace_y=[]
    trace_pre_y,trace_pre_y2=[],[]
    features=[]

    with torch.no_grad():
        for i, (video, index) in enumerate(test_loader):
            video = video.float().cuda()  # batch,10,3,224,224
            label = index.cuda()

            trace_y.append(label.cpu().detach().numpy())

            pre, fea,score = model(video)

            _, preds = torch.max(pre.data, 1)
            right += float(torch.sum(preds == label.data))
            trace_pre_y.append(preds.cpu().detach().numpy())
            features.append(fea[:,2,:].cpu().detach().numpy())

            all += label.size(0)

        trace_y = np.concatenate(trace_y)
        trace_pre_y = np.concatenate(trace_pre_y)
        features = np.concatenate(features)


        weighted_accuracy = accuracy_score(trace_y,trace_pre_y)
        unweighted_accuracy = balanced_accuracy_score(trace_y,trace_pre_y)



    print('Test *Prec@Audio {:.3f};  {:.3f};  {:.3f}  '.format(right/all,right1/all,right2/all))

    return features,trace_y




def get_audio_split_withoutResize(allfeatures, framenum, winlen):
    """

    :param allfeatures:  特征list: audioNum,OriframeLenth,channel,feature_dim
    :param framenum:     分段的段数 length
    :param winlen:       分段后的 feature map 维度, 每段的帧数
    :return:    特征list: audioNum,length,channel,winlen,feature_dim
    """


    feature_dataset = []
    channel = allfeatures[0].shape[1]

    for index in range(len(allfeatures)):
        # print(index)

        mfcc = allfeatures[index]  # 当前音频的特征： frame,3,filter

        if (mfcc.shape[0] < framenum + winlen):
            # 特征前后分别加 winlen/2帧 特征
            padding_front = np.zeros((int(winlen / 2), channel, mfcc.shape[2]))
            padding_back = np.zeros((int(winlen / 2), channel, mfcc.shape[2]))
            front = np.vstack((padding_front, mfcc))
            mfcc = np.vstack((front, padding_back))

        # sequence_list=np.zeros((framenum,winlen,3,mfcc.shape[2]))

        sequence_list = np.zeros((framenum, channel, winlen, mfcc.shape[2]))
        sequence_idx = 0  # 特征分段数  10
        winstep = int(mfcc.shape[0] / framenum)
        idx = 0

        while (idx < mfcc.shape[0] - winlen):

            middle = mfcc[idx:idx + winlen, :, :].transpose((1, 0, 2))  # 3,win,filter
            sequence_list[sequence_idx] = middle
            # for j in range(3):
            #     sequence_list[sequence_idx][j] = cv2.resize(middle[j], (224, 224), interpolation=cv2.INTER_LINEAR)

            # sequence_list[sequence_idx]=mfcc[idx:idx+winlen,:,:]
            idx += winstep
            sequence_idx += 1
            if (sequence_idx == framenum):
                break
        features = torch.from_numpy(sequence_list)
        feature_dataset.append(features)  # 所有分段完的特征  5531,N,3,224,224

    return feature_dataset


def main():

    name_emotionLabel = pickle.load(open('DATA/name_emotionLabel_dict.pickle', 'rb'))

    train_list, test_list = utils.randomSplit(length=len(name_emotionLabel),testNum=opt.testNum)

    allfeatures = list(pickle.load(open(opt.audioFeatureRoot, 'rb')).values())
    feature_dataset = get_audio_split_withoutResize(allfeatures, opt.length, opt.frameWin)
    train_loader, test_loader, feature_loader  = load_dataset.AudioLoadData(feature_dataset,train_list, test_list, name_emotionLabel, opt.batchSize,framelen=opt.length,winlen=opt.frameWin)
    end = np.zeros((4,4))
    for i in [17]:
        model = Model.inno_model1(input_channel=opt.input_channel,classes=opt.nClasses,height=224,width=40)
        checkpoint = torch.load("Checkpoint/Audio_DSCASA_" + str(i) + ".pth")
        model.load_state_dict(checkpoint)
        model.cuda()
        feature,label= test_UA_WA(train_loader, model)

        # data = np.random.randint(0, 90000, (5, 324))
        # label = np.array([1, 2, 2, 3, 2])
        # print(data.shape)
        sc = StandardScaler()
        feature = sc.fit_transform(feature)
        # print('Computing t-SNE embedding')
        tsne = TSNE(n_components=2, init='pca', random_state=0)

        result = tsne.fit_transform(feature)

        fig = plot_embedding(result, label,
                             '(e)')
        plt.savefig('DATA/tsne_audio.jpg')

        plt.show()



if __name__=="__main__":

    main()
