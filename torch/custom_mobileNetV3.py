import torch
import csv
import torch.nn as nn
from torch.nn import init
import torchvision 
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from torchvision.transforms import functional
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import cv2
from PIL import Image
from torch import tensor
import warnings
warnings.filterwarnings(action='ignore') # warning 무시
import utils
# class별로 출력
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("MobileNetv3_custom_logs")
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import glob
import math
# from torchmetrics.classification import MulticlassPrecision, MulticlassRecall
import torcheval

transform = transforms.Compose([          
    transforms.ToTensor()])

sum = 0
cnt = 0



#ccsv = sorted(glob.glob('C:/Users/QR22002/Desktop/chaeyun/dataset/Holder_name/**/*.csv', recursive=True)) #116528
#png_file_path = sorted(glob.glob('C:/Users/QR22002/Desktop/chaeyun/dataset/Holder_name/**/*.png', recursive=True)) #116526
# ppng = sorted(glob.glob('C:/Users/QR22002/Desktop/chaeyun/dataset/Holder_name/**/*.png', recursive=True)) #116526

#Test dataset 저장
png_file_path = sorted(glob.glob('../Dataset/custom_dataset/**/*.png', recursive=True)) #116528

# for i in range(len(ccsv)):
#     tmp_csv_name = ccsv[i].split('.')
#     ccsv[i] = tmp_csv_name[0]
# for i in range(len(ppng)):
#     tmp_png_name = ppng[i].split('.')
#     ppng[i] = tmp_png_name[0]

# set_csv = set(ccsv)
# set_png = set(ppng)
# gone_file = list(set_csv - set_png)
# print(gone_file)
# ==== PNG 없는 파일 ====#
#['C:/Users/QR22002/Desktop/chaeyun/dataset/Holder_name\\hold_name_0430\\Holder_name_alpabet\\0293_9445411545060702\\00174_16_9445411545060702_0820_PARKHAECHUL_cardBox', 
# 'C:/Users/QR22002/Desktop/chaeyun/dataset/Holder_name\\hold_name_0510\\InsufficientAlphabet_Set\\TextFrame', 
# 'C:/Users/QR22002/Desktop/chaeyun/dataset/Holder_name\\hold_name_0409\\hold_name_new\\0346\\001087_100000lux_cardBox', 
# 'C:/Users/QR22002/Desktop/chaeyun/dataset/Holder_name\\hold_name_0430\\Holder_name_alpabet\\0293_9445411545060702\\00176_16_9445411545060702_0820_PARKHAECHUL_cardBox']

#========MobileNet V3========#

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
num_classes = 30

class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV3_Large(nn.Module):
    def __init__(self, num_classes=30):
        super(MobileNetV3_Large, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1),
            Block(3, 16, 64, 24, nn.ReLU(inplace=True), None, 1),
            # Block(3, 24, 72, 24, nn.ReLU(inplace=True), None, 1),
            Block(3, 24, 72, 40, nn.ReLU(inplace=True), SeModule(40), 1),
            # Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
            Block(3, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
            Block(3, 40, 240, 80, hswish(), None, 2),
            # Block(3, 80, 200, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            # Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 480, 112, hswish(), SeModule(112), 1),
            # Block(3, 112, 672, 112, hswish(), SeModule(112), 1),
            Block(5, 112, 672, 160, hswish(), SeModule(160), 1),
            Block(3, 160, 672, 160, hswish(), SeModule(160), 2),
            Block(3, 160, 960, 160, hswish(), SeModule(160), 1),
        )

        
        self.conv2 = nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.hs2 = hswish()
        self.linear3 = nn.Linear(960, 1280)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = hswish()
        self.linear4 = nn.Linear(1280, num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck(out)
        out = self.hs2(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 5)
        out = out.view(out.size(0), -1)
        out = self.hs3(self.bn3(self.linear3(out)))
        out = self.linear4(out)
        return out


class MobileNetV3_Small(nn.Module):
    def __init__(self, num_classes=30):
        super(MobileNetV3_Small, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), SeModule(16), 2),
            Block(3, 16, 72, 24, nn.ReLU(inplace=True), None, 1),
            Block(3, 24, 88, 24, nn.ReLU(inplace=True), None, 1),
            Block(5, 24, 96, 40, hswish(), SeModule(40), 2),
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            Block(5, 40, 120, 48, hswish(), SeModule(48), 1),
            Block(5, 48, 144, 48, hswish(), SeModule(48), 1),
            Block(5, 48, 288, 96, hswish(), SeModule(96), 2),
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
        )

        #마지막 conv layer 6
        # self.bneck = nn.Sequential(
        #     Block(3, 16, 16, 16, nn.ReLU(inplace=True), SeModule(16), 2),
        #     Block(3, 16, 72, 24, nn.ReLU(inplace=True), None, 1),
        #     Block(3, 24, 88, 24, nn.ReLU(inplace=True), None, 1),
        #     Block(5, 24, 96, 40, hswish(), SeModule(40), 1),
        #     Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
        #     Block(5, 40, 240, 40, hswish(), SeModule(40), 2),
        #     Block(5, 40, 120, 48, hswish(), SeModule(48), 1),
        #     Block(5, 48, 144, 48, hswish(), SeModule(48), 1),
        #     Block(5, 48, 288, 96, hswish(), SeModule(96), 1),
        #     Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
        #     Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
        # )

        self.conv2 = nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(576)
        self.hs2 = hswish()
        self.linear3 = nn.Linear(576, 1280)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = hswish()
        self.linear4 = nn.Linear(1280, num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck(out)
        out = self.hs2(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 3)
        out = out.view(out.size(0), -1)
        out = self.hs3(self.bn3(self.linear3(out)))
        out = self.linear4(out)
        return out




from torchinfo import summary
# 모델 확인 
from torchvision import models
model = MobileNetV3_Small()
summary(model, (2,3,48,48), device="cpu")
