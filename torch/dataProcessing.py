import torch
import csv
import torch.nn as nn
import torchvision 
import random
import shutil
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from torchvision.transforms import functional
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import cv2
from PIL import Image
from torch import tensor
import numpy as np
import os
import pandas as pd
import glob

ccsv = sorted(glob.glob('C:/Users/QR22002/Desktop/chaeyun/dataset/Holder_name/**/*.csv', recursive=True)) #116528
png_file_path = sorted(glob.glob('C:/Users/QR22002/Desktop/chaeyun/dataset/Holder_name/**/*.png', recursive=True)) #116526
ppng = sorted(glob.glob('C:/Users/QR22002/Desktop/chaeyun/dataset/Holder_name/**/*.png', recursive=True)) #116526

#Test dataset 저장
png_file_path = sorted(glob.glob('../Dataset/OCR_HolderName/**/*.png', recursive=True)) #116528

for i in range(len(ccsv)):
    tmp_csv_name = ccsv[i].split('.')
    ccsv[i] = tmp_csv_name[0]
for i in range(len(ppng)):
    tmp_png_name = ppng[i].split('.')
    ppng[i] = tmp_png_name[0]

set_csv = set(ccsv)
set_png = set(ppng)
gone_file = list(set_csv - set_png)
print(gone_file)

# ==== PNG 없는 파일 ====#
#['C:/Users/QR22002/Desktop/chaeyun/dataset/Holder_name\\hold_name_0430\\Holder_name_alpabet\\0293_9445411545060702\\00174_16_9445411545060702_0820_PARKHAECHUL_cardBox', 
# 'C:/Users/QR22002/Desktop/chaeyun/dataset/Holder_name\\hold_name_0510\\InsufficientAlphabet_Set\\TextFrame', 
# 'C:/Users/QR22002/Desktop/chaeyun/dataset/Holder_name\\hold_name_0409\\hold_name_new\\0346\\001087_100000lux_cardBox', 
# 'C:/Users/QR22002/Desktop/chaeyun/dataset/Holder_name\\hold_name_0430\\Holder_name_alpabet\\0293_9445411545060702\\00176_16_9445411545060702_0820_PARKHAECHUL_cardBox']

#hold_name = sorted(glob.glob('C:/Users/QR22002/Desktop/chaeyun/dataset/Holder_name/hold_name_0409/hold_name_new/0008/*.png', recursive=True))

#Train dataset 저장
#66 -> 0004, 0007, 0012, 0015 ... etc 
# train_png_file_path = sorted(glob.glob('C:/Users/QR22002/Desktop/chaeyun/dataset/Holder_name/**/*.png', recursive=True)) #116526

# test_png_file_path = sorted(glob.glob('C:/Users/QR22002/Desktop/chaeyun/dataset/hold_name_test/**/*.png', recursive=True)) #116528

# origin_train_path = sorted(glob.glob('C:/Users/QR22002/Desktop/chaeyun/dataset/dataset/train/**/*.png', recursive=True))

# origin_test_path = sorted(glob.glob('C:/Users/QR22002/Desktop/chaeyun/dataset/dataset/test/**/*.png', recursive=True))

# origin_val_path = sorted(glob.glob('C:/Users/QR22002/Desktop/chaeyun/dataset/dataset/validation/**/*.png', recursive=True))

# custom_val_path = sorted(glob.glob('C:/Users/QR22002/Desktop/chaeyun/custom_dataset/validation/**/*.png', recursive=True))

transform = transforms.Compose([          
    transforms.ToTensor()])
"""        
# Train data 추출
""" 
def train_data_processing (train_png_file_path, origin_train_path, origin_val_path) :
    train_csv_file_path = []
    for i in range(len(train_png_file_path)):
        tmp_csv_name = os.path.splitext(train_png_file_path[i])
        train_csv_file_path.append(tmp_csv_name[0]+'.csv')

    print(f'csv file: {len(train_csv_file_path)}')
    print(f'png file: {len(train_png_file_path)}')

    for i in range(len(origin_train_path)):
        origin_train_path[i] = os.path.basename(origin_train_path[i])
    print(f'origin train data: {len(origin_train_path)}')

    for i in range(len(origin_val_path)):
        origin_val_path[i] = os.path.basename(origin_val_path[i])
    print(f'origin train data: {len(origin_val_path)}')

    length = len(train_png_file_path)
    for idx in range(length):
        tmp_csv_name = os.path.splitext(train_csv_file_path[idx])   #split => return list
        tmp_png_name = os.path.splitext(train_png_file_path[idx])
        if(tmp_csv_name[0] == tmp_png_name[0]):
            data = pd.read_csv(train_csv_file_path[idx], index_col=False, header=None, sep=';') #header=None(주의!!)
            image = cv2.imread(train_png_file_path[idx])
            image = transform(image)
            #print(data[1][0])   #열-행
            data_len = data.shape
            etc = data[5][0]
            
            for i in range(data_len[0]):
                cnt+=1
                tmpLabel = data[0][i]
                #print(tmpLabel)
                if type(tmpLabel) is str:
                    if etc == -1:
                        label = 100 #blank
                    else:
                        label = ord(tmpLabel)
                elif type(tmpLabel) is int:
                    if etc is int:
                        label = tmpLabel
                    elif type(etc) is float:
                        if 1 <= tmpLabel and tmpLabel <= 26:
                            label = tmpLabel+64
                    elif etc == ' ':
                        if 1 <= tmpLabel and tmpLabel <= 26:
                            label = tmpLabel+64
                elif type(tmpLabel) is float:
                    label = 100
                
                pos = []
                for j in range(1, data_len[1]):
                    pos.append(data[j][i])
                
                # 이미지 편집
                crop_pixel = random.randint(1,2)
                cropped_img = torchvision.transforms.functional.crop(image, pos[1]-crop_pixel, pos[0]-crop_pixel, pos[3]+crop_pixel, pos[2]+crop_pixel)   # 자르기
                resize = torchvision.transforms.Resize((46, 46)) # 확대
                enlarged_img = resize(cropped_img)
                padded_img = torchvision.transforms.functional.pad(enlarged_img, (1,1,1,1), fill=0)  # padding
                transform_to_img = transforms.ToPILImage()
                img = transform_to_img(padded_img)
                
                #tmpPath = f'C:/Users/QR22002/Desktop/chaeyun/dataset/train/{label}/'
                
                tmpPath = f'C:/Users/QR22002/Desktop/chaeyun/custom_dataset/train/{label}/'
                tmpValPath = f'C:/Users/QR22002/Desktop/chaeyun/custom_dataset/validation/{label}/'
                
                if not os.path.exists(tmpPath):
                    os.makedirs(tmpPath)
                if not os.path.exists(tmpValPath):
                    os.makedirs(tmpValPath)
                
                comparePath = f'{os.path.basename(tmp_csv_name[0])}' + '_' + f'{cnt}.png'
            
                if comparePath in origin_train_path:
                    print(f'same: {comparePath}')
                    img.save(tmpPath + comparePath)
                elif comparePath in origin_val_path:
                    print(f'not same: {comparePath}')
                    img.save(tmpValPath + comparePath)
                #img.save(f'C:/Users/QR22002/Desktop/chaeyun/dataset/test/{label}/' + f'{os.path.basename(tmp_csv_name[0])}' + '_' + f'{cnt}.png')
            cnt = 0
        print(f"done: index: {idx} {os.path.basename(tmp_csv_name[0])}")
    
def test_data_processing(test_png_file_path):
    # Test data 추출
    print(f'{len(test_png_file_path)}')
    test_csv_file_path = []
    for i in range(len(test_png_file_path)):
        tmp_csv_name = os.path.splitext(test_png_file_path[i])
        test_csv_file_path.append(tmp_csv_name[0]+'.csv')

    print(f'csv file: {len(test_csv_file_path)}')
    print(f'png file: {len(test_png_file_path)}')

    length = len(test_png_file_path)
    for idx in range(length):
        tmp_csv_name = os.path.splitext(test_csv_file_path[idx])   #split => return list
        tmp_png_name = os.path.splitext(test_png_file_path[idx])
        if(tmp_csv_name[0] == tmp_png_name[0]):
            data = pd.read_csv(test_csv_file_path[idx], index_col=False, header=None) #header=None(주의!!)
            image = cv2.imread(test_png_file_path[idx])
            image = transform(image)
            #print(data[1][0])   #열-행
            len = data.shape
            
            if len[1] != 5: # 열 != 5 => skip
                continue
            
            for i in range(len[0]):
                cnt+=1
                tmpLabel = data[0][i]
                
                label = int(tmpLabel)+64
                
                pos = []
                flag = False
                for j in range(1, len[1]):
                    if pd.isna(data[j][i]):
                        flag = True
                        continue
                    else:
                        pos.append(int(data[j][i]))
                        
                # 이미지 편집
                if flag == True: continue
                crop_pixel = random.randint(1,2)
                cropped_img = torchvision.transforms.functional.crop(image, pos[1]-crop_pixel, pos[0]-crop_pixel, pos[3]+crop_pixel, pos[2]+crop_pixel)   # 자르기
                resize = torchvision.transforms.Resize((46, 46)) # 확대
                enlarged_img = resize(cropped_img)
                padded_img = torchvision.transforms.functional.pad(enlarged_img, (1,1,1,1), fill=0)  # padding
                transform_to_img = transforms.ToPILImage()
                img = transform_to_img(padded_img)
                
                #tmpPath = f'C:/Users/QR22002/Desktop/chaeyun/dataset/train/{label}/'
                tmpPath = f'C:/Users/QR22002/Desktop/chaeyun/custom_dataset/test/{label}/'
                if not os.path.exists(tmpPath):
                    os.makedirs(tmpPath)
                #img.save(f'C:/Users/QR22002/Desktop/chaeyun/dataset/train/{label}/' + f'{os.path.basename(tmp_csv_name[0])}' + '_' + f'{cnt}.png')
                img.save(tmpPath + f'{os.path.basename(tmp_csv_name[0])}' + '_' + f'{cnt}.png')
            cnt = 0
        print(f"done: index: {idx} {os.path.basename(tmp_csv_name[0])}")
