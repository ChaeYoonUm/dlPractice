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
    transforms.ToTensor()                
                                ])

path = 'C:\\Users\\QR22002\\Desktop\\chaeyun\\dataset\\Holder_name'
path0 = 'C:\\Users\\QR22002\\Desktop\\chaeyun\\dataset\\Holder_name\\ETC_holder_name\\holder_name_old'
path1 = 'C:\\Users\\QR22002\\Desktop\\chaeyun\\dataset\\Holder_name\\ETC_holder_name\\QA-Blank'
# for path_idx in range(len(path_list)):
#     path = path_list[path_idx]

# for(root, directories, files) in os.walk(path):
#     print(directories) 

# root: 디렉토리
sum = 0
cnt = 0
for(root, directories, files) in os.walk(path0):
    if len(files) == 0:
        continue
    for file in files:
        cnt += 1
        fileCSV = r'.csv'
        filePNG = r'.png'
        csv_list = [file for file in files if file.endswith(fileCSV)]
        png_list = [file for file in files if file.endswith(filePNG)]  
        
        for idx in range(len(csv_list)):
            tmpCVSPath = os.path.join(root, csv_list[idx])
            tmpPNGPath = os.path.join(root, png_list[idx])
            data = pd.read_csv(tmpCVSPath, index_col=False, header=None) #header=None(주의!!)
            image = cv2.imread(tmpPNGPath)
            image = transform(image)
            #print(data[1][0])   #열-행
            len = data.shape
            for i in range(len[0]):
                tmpLabel = data[0][i]
                if 1 <= tmpLabel and tmpLabel <= 26:
                    label = chr(tmpLabel+64)
                elif 'A' <= tmpLabel and tmpLabel <= 'Z':
                    label = tmpLabel
                else:
                    label = ' '
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
                print(tmpCVSPath)
                name = csv_list[idx].split('.')
                #img.save(f'C:/Users/QR22002/Desktop/chaeyun/dataset/train/{label}/' + f'{name}' + f'{cnt}.png')
            cnt=0
    #sum += (len(csv_list) + len(png_list)) 
