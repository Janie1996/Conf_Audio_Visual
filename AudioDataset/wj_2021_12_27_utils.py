# -*- coding: UTF-8 -*-
"""
@file:wj_2021_12_27_utils.py
@author: Wei Jie
@date: 2021/12/27
@description: 划分训练集、测试集函数，提供了各种不同的划分方式
"""
import sys
import os

sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])

import random
import numpy as np

"""
5531 utterances 
randomly chose 80% for training and 20% for testing

70.07%   WA
70.67%   UA

"""

"""
5531 utterances.    

10-fold
cross-validation with 8, 1, 1 in train, dev, test set respectively

speech only：
WA 66.6% 
UA  68.4%

Speech + text
WA  80.3%
UA   81.4%

"""


"""
merged the ‘happy’ and ‘excited’ as ‘happy’ ；combined ‘sad’ and ‘frustrated’ as ‘sad’

randomly split 80% of the samples into a training set and 20% into a testing set （5-fold cross-validation） 77.93%   WA   77.03%   UA

"""
def randomSplit(length,testNum):

    # x= np.random.randint(10000)
    # print(x)
    random.seed(1156)
    # length=100
    # testNum=20
    test_list = random.sample(range(0, length), int(testNum))
    print(test_list)
    #print(sum(test_list))
    train_list = list(set(np.arange(length)) - set(test_list))
    # np.random.shuffle(train_list)

    return train_list,test_list #,x

"""
5531 utterances

Speaker-independent experiments
first 4 sessions are used for training. The last session is used for testing.

61.32%   WA
60.43%   UA
61.65%   F1-score

5-fold leave-one-session-out (LOSO) cross validation

UA   68.3%
WA  66.9%

"""


"""
5-fold cross-validation

three sessions for training the model, one session for development, and the remaining session for testing

weighted f1-score 63.78

"""

def leaveOneSession(name_emotionLabel,session):

    test_list = []
    names = list(name_emotionLabel.keys())
    for i in range(len(names)):
        if session in names[i]:
            test_list.append(i)

    train_list = list(set(np.arange(len(names))) - set(test_list))
    np.random.shuffle(train_list)
    return train_list,test_list


"""
10-fold leave-one-speaker-out     improvised  70.5%  WA    72.5%  UA

5531
10 fold leave-one-speaker out cross-validation ; 8 speakers as the training set，one speaker for validation and one for testing

WA  58.62%
UA   59.91%

"""

def leaveOneSpeaker(name_emotionLabel,session,man):


    test_list = []
    names = list(name_emotionLabel.keys())
    for i in range(len(names)):
        if session in names[i] and man in names[i]:
            test_list.append(i)

    train_list = list(set(np.arange(len(names))) - set(test_list))
    np.random.shuffle(train_list)
    return train_list, test_list


if __name__ == '__main__':
    randomSplit(5531,6)

    x=np.zeros([3,2])
    x[2][1]=3
    x1=np.ones(3)
    x1[2]=10
    state = np.random.get_state()
    np.random.shuffle(x)
    print(x)

    np.random.set_state(state)
    np.random.shuffle(x1)
    print(x1)

    np.random.shuffle(x)
    print(x)
    randomSplit(5,2)