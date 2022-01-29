# -*- coding: UTF-8 -*-
"""
@file:wj_2022_01_12_VGGFACE.py
@author: Wei Jie
@date: 2022/1/12
@description:
"""
import sys
import os

sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])


import torch
import torch.nn as nn
import torch.legacy.nn as lnn
from functools import reduce
from torch.autograd import Variable
from torchvision import datasets,transforms
import torch.optim as optim

from preProcess.readVideo import MyData

import scipy.io as io
import numpy as np

class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class LambdaMap(LambdaBase):
    def forward(self, input):
        return list(map(self.lambda_func,self.forward_prepare(input)))

class LambdaReduce(LambdaBase):
    def forward(self, input):
        return reduce(self.lambda_func,self.forward_prepare(input))

VGG_FACE=nn.Sequential( # Sequential,
    nn.Conv2d(3,64,(3, 3),(1, 1),(1, 1)),
    nn.ReLU(),
    nn.Conv2d(64,64,(3, 3),(1, 1),(1, 1)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
    nn.Conv2d(64,128,(3, 3),(1, 1),(1, 1)),
    nn.ReLU(),
    nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
    nn.Conv2d(128,256,(3, 3),(1, 1),(1, 1)),
    nn.ReLU(),
    nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1)),
    nn.ReLU(),
    nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
    nn.Conv2d(256,512,(3, 3),(1, 1),(1, 1)),
    nn.ReLU(),
    nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
    nn.ReLU(),
    nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
    nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
    nn.ReLU(),
    nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
    nn.ReLU(),
    nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
    Lambda(lambda x: x.view(x.size(0),-1)), # View,
    nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(25088,4096)), # Linear,
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(4096,4096)), # Linear,
    # nn.ReLU(),
    # nn.Dropout(0.5),
    # nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(4096,2622)), # Linear,
    # nn.Linear(2622,7),
    # nn.Softmax(),
    )



#Pretrained

model_dict = VGG_FACE.state_dict()
vggface_state_dict = torch.load("VGG_FACE.pth")
#pretrained_dict = {k: v for k, v in vggface_state_dict.items() if k in model_dict and not k.startswith("38")}
pretrained_dict = {k: v for k, v in vggface_state_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
VGG_FACE.load_state_dict(model_dict)




data_dir = "E:\\ESVG\\emotional database\\phase_1\\face_videos"  #"E:\\2019-07-dong\\10frame" #'E:\\videoEmotionReco\\10frame-new\\1\\train'

train=MyData(data_dir,10,224,224,50,30)
train_loaders = torch.utils.data.DataLoader(train, batch_size=1,shuffle=False)


VGG_FACE.cuda()
VGG_FACE.train(False)


feature=[]  #videoNum,length,featureNum
for j, data in enumerate(train_loaders, 0):
    inputs, labels = data
    inputs = inputs.cuda()


    x=inputs
    x = x.permute(1, 0, 2, 3,4)  # length,batch,channel,height,width
    newx=None
    for i in range(x.shape[0]):
        y=VGG_FACE(x[i])
        y=y.unsqueeze(0)
        if(i==0):
            newx=y
        else:
            newx=torch.cat((newx,y),0)   #length,batch,feature
    out=newx.permute(1,0,2)    # batch,length,feature,

    out=out.cpu().detach().numpy()
    if(j==0):
        feature=out
    else:
        feature=np.vstack((feature,out))
    #print('t') #resnet101_ImageNet_train_data
io.savemat('D:\\VGG_FACE_train_data4',{'test_data':feature})


train=MyData(data_dir,10,224,224,50,40)
train_loaders = torch.utils.data.DataLoader(train, batch_size=1,shuffle=False)


VGG_FACE.cuda()
VGG_FACE.train(False)


feature=[]  #videoNum,length,featureNum
for j, data in enumerate(train_loaders, 0):
    inputs, labels = data
    inputs = inputs.cuda()


    x=inputs
    x = x.permute(1, 0, 2, 3,4)  # length,batch,channel,height,width
    newx=None
    for i in range(x.shape[0]):
        y=VGG_FACE(x[i])
        y=y.unsqueeze(0)
        if(i==0):
            newx=y
        else:
            newx=torch.cat((newx,y),0)   #length,batch,feature
    out=newx.permute(1,0,2)    # batch,length,feature,

    out=out.cpu().detach().numpy()
    if(j==0):
        feature=out
    else:
        feature=np.vstack((feature,out))
    #print('t') #resnet101_ImageNet_train_data
io.savemat('D:\\VGG_FACE_train_data5',{'test_data':feature})


if __name__ == "__main__":
    print()