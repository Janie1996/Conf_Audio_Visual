# -*- coding: UTF-8 -*-
"""
@file:wj_2022_01_07_LSTM.py
@author: Wei Jie
@date: 2022/1/7
@description:
"""
import sys
import os

sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])

import torch.nn as nn
import numpy as np


class EmotionRecog(nn.Module):
    def __init__(self,classes,featureDim):
        super(EmotionRecog, self).__init__()
        self.classes=classes
        self.featureDim = featureDim

        self.temp_lstm1 = nn.LSTM(input_size=self.featureDim, hidden_size=128, num_layers=1, batch_first=True,
                                  bidirectional=False)
        self.temp_lstm2 = nn.LSTM(input_size=128, hidden_size=self.classes, num_layers=1, batch_first=True, bidirectional=False)

    def forward(self,input):


        output,_=self.temp_lstm1(input)
        output,_=self.temp_lstm2(output)


        return output[:, -1, :]



if __name__ == "__main__":


    import torch
    a=torch.ones([2,5,2048]).cuda()
    b=torch.ones([2,5,512]).cuda()

    input = np.concatenate((a, b), axis=2)
    input = torch.from_numpy(input)

    model = EmotionRecog(4,2048+512)
    output = model(input)

    print()