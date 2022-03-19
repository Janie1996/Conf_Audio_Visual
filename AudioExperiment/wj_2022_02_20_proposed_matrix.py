# -*- coding: UTF-8 -*-
"""
@file:wj_2022_02_20_proposed_matrix.py
@author: Wei Jie
@date: 2022/2/20
@description: 只进行测试，绘制  matrix
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
from AudioDataset import wj_2022_01_26_dataload as load_dataset
from AudioModels import wj_2022_01_08_DSCASA_Transformer as Model
import pickle
import numpy as np

# np.random.seed(12)
# torch.manual_seed(12)  #为CPU设置种子用于生成随机数，以使得结果是确定的   　　
# torch.cuda.manual_seed(12) #为当前GPU设置随机种子；  　　
# cudnn.deterministic = True

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
import numpy as np

import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(cm, savename, classes,title='Confusion Matrix'):
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(savename, format='png')
    plt.show()


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

            all += label.size(0)

        trace_y = np.concatenate(trace_y)
        trace_pre_y = np.concatenate(trace_pre_y)


        weighted_accuracy = accuracy_score(trace_y,trace_pre_y)
        unweighted_accuracy = balanced_accuracy_score(trace_y,trace_pre_y)
        cm= confusion_matrix(trace_y,trace_pre_y)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    print('Test *Prec@Audio {:.3f};  {:.3f};  {:.3f}  '.format(right/all,right1/all,right2/all))

    return max(right/all,right1/all,right2/all),unweighted_accuracy,cm_normalized




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
    for i in [0,4,11,12,17]:
        model = Model.inno_model1(input_channel=opt.input_channel,classes=opt.nClasses,height=224,width=40)
        checkpoint = torch.load("Checkpoint/Audio_DSCASA_" + str(i) + ".pth")
        model.load_state_dict(checkpoint)
        model.cuda()
        acc, acc1,cm = test_UA_WA(test_loader, model)
        end+=cm
    classes = ['ang', 'hap', 'neu', 'sad']
    trueNu =5*np.ones((4,4))
    # end=end*20
    plot_confusion_matrix(end/trueNu*100, 'fusion.png', classes)
    np.save('DATA/cm_6990.npy', end)


if __name__=="__main__":
    x1 = np.load('DATA/cm_6990.npy')
    x2 = np.load('DATA/cm_0.npy')
    x3 = np.load('DATA/cm_4259.npy')
    x4 = np.load('DATA/cm_1156.npy')
    x5 = np.load('DATA/cm_6364.npy')
    end = 25*np.ones((4,4))
    pre = (x1+x2+x3+x4+x5)/end
    classes = ['ang', 'hap', 'neu', 'sad']
    plot_confusion_matrix(pre*100,'audio.png',classes)
    main()