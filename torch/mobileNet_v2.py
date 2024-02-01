import torch
import torch.nn as nn
import torchvision 
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from torchvision.transforms import functional
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import cv2
from PIL import Image

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import glob
    
transform = transforms.Compose([          
    transforms.ToTensor()])
path = 'C:\\Users\\QR22002\\Desktop\\chaeyun\\dataset\\Holder_name'
sum = 0
cnt = 0

csv_file_path = glob.glob('C:/Users/QR22002/Desktop/chaeyun/dataset/Holder_name/**/*.csv', recursive=True) #116528
png_file_path = glob.glob('C:/Users/QR22002/Desktop/chaeyun/dataset/Holder_name/**/*.png', recursive=True) #116526
csv_file_path.sort()
png_file_path.sort()

length = len(png_file_path)
for idx in range(length):
    tmp_csv_name = csv_file_path[idx].split('.')    #split => return list
    tmp_png_name = png_file_path[idx].split('.')
    if(tmp_csv_name[0] == tmp_png_name[0]):
        
        data = pd.read_csv(csv_file_path[idx], index_col=False, header=None, sep=';') #header=None(주의!!)
        image = cv2.imread(png_file_path[idx])
        image = transform(image)
        #print(data[1][0])   #열-행
        len = data.shape
        for i in range(len[0]):
            cnt+=1
            tmpLabel = data[0][i]
            if tmpLabel.isdigit():
                if 1 <= tmpLabel and tmpLabel <= 26:
                    label = chr(tmpLabel+64)
            elif tmpLabel.isalpha():
                if 'A' <= tmpLabel and tmpLabel <= 'Z':
                    label = tmpLabel
            else:
                label = "\' \'"
            
            pos = []
            for j in range(1, len[1]):
                pos.append(data[j][i])
            # 이미지 편집
            cropped_img = torchvision.transforms.functional.crop(image, pos[1]-5, pos[0]-5, pos[3]+10, pos[2]+15)   # 자르기
            resize = torchvision.transforms.Resize((220, 220)) # 확대
            enlarged_img = resize(cropped_img)
            padded_img = torchvision.transforms.functional.pad(enlarged_img, (2,2,2,2), fill=0)  # padding
            transform_to_img = transforms.ToPILImage()
            img = transform_to_img(padded_img)
            img.save(f'C:/Users/QR22002/Desktop/chaeyun/dataset/train/{label}/' + f'{os.path.basename(tmp_csv_name[0])}' + f'{idx}_' + f'{cnt}.png')
        cnt = 0
        
#========MobileNet V2========#
device = (
    "cuda"
    if torch.cuda.is_available()
    else "gpu"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class MobileNet_v2(nn.Module):
    def __init__(self):
        super(MobileNet_v2, self).__init__()
        
        # Layer1 
        self.layer1 = torch.nn.Sequential(
            nn.Conv2d(kernel_size=3, in_channels=3, out_channels=32, stride=2, padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(0.2))
        
        # Layer2 (bottleneck)
        self.layer2 = torch.nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=32, out_channels=16, padding='same', stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(),
            nn.Dropout2d(0.4))
        
        # Layer3-(1) (bottleneck)
        self.layer3_1 = torch.nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=16, out_channels=24*6, padding='same', stride=2),
            nn.ReLU6(),
            nn.Conv2d(kernel_size=3, in_channels=24*6, out_channels=24*6, padding='same', stride=2, groups=3),
            nn.ReLU6(),
            nn.Linear(kernel_size=1, in_channels=24*6, out_channels=24, padding='same', stride=2),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.4))
        
        # Layer3-(2) (bottleneck)
        self.layer3_2 = torch.nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=24, out_channels=24*6, padding='same', stride=2),
            nn.ReLU6(),
            nn.Conv2d(kernel_size=3, in_channels=24*6, out_channels=24*6, padding='same', stride=2, groups=3),
            nn.ReLU6(),
            nn.Linear(kernel_size=1, in_channels=24*6, out_channels=24, padding='same', stride=2),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.4))
        
        # Layer4-(1)
        self.layer4_1 = torch.nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=24, out_channels=32*6, padding='same', stride=2),
            nn.ReLU6(),
            nn.Conv2d(kernel_size=3, in_channels=32*6, out_channels=32*6, padding='same', stride=2, groups=3),
            nn.ReLU6(),
            nn.Linear(kernel_size=1, in_channels=32*6, out_channels=32, padding='same', stride=2),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.4))
        
        # Layer4-(2)
        self.layer4_2 = torch.nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=32, out_channels=32*6, padding='same', stride=2),
            nn.ReLU6(),
            nn.Conv2d(kernel_size=3, in_channels=32*6, out_channels=32*6, padding='same', stride=2, groups=3),
            nn.ReLU6(),
            nn.Linear(kernel_size=1, in_channels=32*6, out_channels=32, padding='same', stride=2),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.4))
        
        # Layer4-(3)
        self.layer4_3 = torch.nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=32, out_channels=32*6, padding='same', stride=2),
            nn.ReLU6(),
            nn.Conv2d(kernel_size=3, in_channels=32*6, out_channels=32*6, padding='same', stride=2, groups=3),
            nn.ReLU6(),
            nn.Linear(kernel_size=1, in_channels=32*6, out_channels=32, padding='same', stride=2),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.4))
        
        # Layer5-(1)
        self.layer5_1 = torch.nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=32, out_channels=64*6, padding='same', stride=2),
            nn.ReLU6(),
            nn.Conv2d(kernel_size=3, in_channels=64*6, out_channels=64*6, padding='same', stride=2, groups=3),
            nn.ReLU6(),
            nn.Linear(kernel_size=1, in_channels=64*6, out_channels=64, padding='same', stride=2),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.4))
        
        # Layer5-(2)
        self.layer5_2 = torch.nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=64, out_channels=64*6, padding='same', stride=2),
            nn.ReLU6(),
            nn.Conv2d(kernel_size=3, in_channels=64*6, out_channels=64*6, padding='same', stride=2, groups=3),
            nn.ReLU6(),
            nn.Linear(kernel_size=1, in_channels=64*6, out_channels=64, padding='same', stride=2),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.4))
        
        # Layer5-(3)
        self.layer5_3 = torch.nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=64, out_channels=64*6, padding='same', stride=2),
            nn.ReLU6(),
            nn.Conv2d(kernel_size=3, in_channels=64*6, out_channels=64*6, padding='same', stride=2, groups=3),
            nn.ReLU6(),
            nn.Linear(kernel_size=1, in_channels=64*6, out_channels=64, padding='same', stride=2),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.4))
        
        # Layer5-(4)
        self.layer5_4 = torch.nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=64, out_channels=64*6, padding='same', stride=2),
            nn.ReLU6(),
            nn.Conv2d(kernel_size=3, in_channels=64*6, out_channels=64*6, padding='same', stride=2, groups=3),
            nn.ReLU6(),
            nn.Linear(kernel_size=1, in_channels=64*6, out_channels=64, padding='same', stride=2),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.4))
    
        # Layer6-(1)
        self.layer6_1 = torch.nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=64, out_channels=96*6, padding='same', stride=1),
            nn.ReLU6(),
            nn.Conv2d(kernel_size=3, in_channels=96*6, out_channels=96*6, padding='same', stride=1, groups=3),
            nn.ReLU6(),
            nn.Linear(kernel_size=1, in_channels=96*6, out_channels=96, padding='same', stride=1),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.4))
        
        # Layer6-(2)
        self.layer6_2 = torch.nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=96, out_channels=96*6, padding='same', stride=1),
            nn.ReLU6(),
            nn.Conv2d(kernel_size=3, in_channels=96*6, out_channels=96*6, padding='same', stride=1, groups=3),
            nn.ReLU6(),
            nn.Linear(kernel_size=1, in_channels=96*6, out_channels=96, padding='same', stride=1),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.4))
        
        # Layer6-(3)
        self.layer6_2 = torch.nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=96, out_channels=96*6, padding='same', stride=1),
            nn.ReLU6(),
            nn.Conv2d(kernel_size=3, in_channels=96*6, out_channels=96*6, padding='same', stride=1, groups=3),
            nn.ReLU6(),
            nn.Linear(kernel_size=1, in_channels=96*6, out_channels=96, padding='same', stride=1),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.4))
        
        # Layer7-(1)
        self.layer7_1 = torch.nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=96, out_channels=160*6, padding='same', stride=2),
            nn.ReLU6(),
            nn.Conv2d(kernel_size=3, in_channels=160*6, out_channels=160*6, padding='same', stride=2, groups=3),
            nn.ReLU6(),
            nn.Linear(kernel_size=1, in_channels=160*6, out_channels=96, padding='same', stride=2),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.4))
        
        # Layer7-(2)
        self.layer7_2 = torch.nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=96, out_channels=160*6, padding='same', stride=2),
            nn.ReLU6(),
            nn.Conv2d(kernel_size=3, in_channels=160*6, out_channels=160*6, padding='same', stride=2, groups=3),
            nn.ReLU6(),
            nn.Linear(kernel_size=1, in_channels=160*6, out_channels=160, padding='same', stride=2),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.4))
        
        # Layer7-(3)
        self.layer7_3 = torch.nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=160, out_channels=160*6, padding='same', stride=2),
            nn.ReLU6(),
            nn.Conv2d(kernel_size=3, in_channels=160*6, out_channels=160*6, padding='same', stride=2, groups=3),
            nn.ReLU6(),
            nn.Linear(kernel_size=1, in_channels=160*6, out_channels=160, padding='same', stride=2),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.4))
        
        # Layer8
        self.layer8 = torch.nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=160, out_channels=320*6, padding='same', stride=1),
            nn.ReLU6(),
            nn.Conv2d(kernel_size=3, in_channels=320*6, out_channels=320*6, padding='same', stride=2, groups=3),
            nn.ReLU6(),
            nn.Linear(kernel_size=1, in_channels=320*6, out_channels=320, padding='same', stride=2),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.4))
        
        # Layer9
        self.layer9 = torch.nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=320, out_channels=1280, padding='same', stride=1),
            nn.ReLU6(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.4))
        
        # Layer10
        self.layer10 = torch.nn.Sequential(
            nn.AvgPool2d(kernel_size=7))
        
        # Layer11
        self.fc_layer = torch.nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=1280),
            nn.Linear(1280, 27))
        
    # Training 진행 순서
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3_1(out)
        out = self.layer3_2(out)
        out = self.layer4_1(out)
        out = self.layer4_2(out)
        out = self.layer4_3(out)
        out = self.layer5_1(out)
        out = self.layer5_2(out)
        out = self.layer5_3(out)
        out = self.layer5_4(out)
        out = self.layer6_1(out)
        out = self.layer6_2(out)
        out = self.layer7_1(out)
        out = self.layer7_2(out)
        out = self.layer7_3(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = nn.Flatten()(out)
        out = self.fc_layer(out)
        return out


#Model 불러오기
model = MobileNet_v2().to(device)

#Tensorflow에서 model.compile() 부분에 해당
loss_fn = torch.nn.CrossEntropyLoss().to(device)   #Softmax 함수 & Negative Log Liklihood(NLL)까지 포함되어 있음
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.00004)

