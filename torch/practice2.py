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
import warnings
warnings.filterwarnings(action='ignore') # warning 무시

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

csv_file_path = sorted(glob.glob('C:/Users/QR22002/Desktop/chaeyun/dataset/Holder_name/**/*.csv', recursive=True)) #116528
png_file_path = sorted(glob.glob('C:/Users/QR22002/Desktop/chaeyun/dataset/Holder_name/**/*.png', recursive=True)) #116526

print(f'csv file: {len(csv_file_path)}')
print(f'png file: {len(png_file_path)}')

csv_file_path = set(csv_file_path)
png_file_path = set(png_file_path)

length = len(png_file_path)
print(f'length: {length}')
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
            img.save(f'C:/Users/QR22002/Desktop/chaeyun/dataset/train/{label}/' + f'{os.path.basename(tmp_csv_name[0])}' + '_' + f'{label}_{cnt}.png')
        cnt = 0
    print(f"done: {os.path.basename(tmp_csv_name[0])}")
  
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
            nn.Conv2d(kernel_size=3, in_channels=3, out_channels=32, padding='same', stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(0.2))
        
        # Layer2 (bottleneck)
        self.layer2 = torch.nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=32, out_channels=16,  padding='same', stride=1),
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

train_data = ImageFolder(root='C:/Users/QR22002/Desktop/chaeyun/dataset/train')
train_data_loader = DataLoader(dataset=train_data, batch_size=64)
#train_data, val_data = torch.utils.data.random_split(train_data, [50000, 10000])

test_data = ImageFolder(root='C:/Users/QR22002/Desktop/chaeyun/dataset/test')
test_data_loader = DataLoader(dataset=test_data, batch_size=64)
"""
batch_size = 64
data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)


#Model 불러오기
model = MobileNet_v2().to(device)

#Tensorflow에서 model.compile() 부분에 해당
loss_fn = torch.nn.CrossEntropyLoss().to(device)   #Softmax 함수 & Negative Log Liklihood(NLL)까지 포함되어 있음
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.00004)

total_batch = len(train_data_loader)
total_train = len(train_data)
for epoch in range(40): 
    total = 0.0
    running_accuracy = 0.0
    running_vall_loss = 0.0 
    total_loss, total_acc = 0.0, 0.0
    cnt = 0
    with tqdm(train_data_loader, unit="batch") as tepoch: #progress bar, batch=32 -> 1563번 = 1epoch
        for X, Y in tepoch:
            model.train()#train 단계임을 명시
            tepoch.set_description(f"Epoch {epoch+1}")
            
            prediction = model(X)
            loss = loss_fn(prediction, Y)   # Y: label
            
            total_loss += loss.item()
            total_acc += (prediction.argmax(1)==Y).sum().item()
            
            #Backpropagation
            optimizer.zero_grad()
            loss.backward()
            
            #parameter update
            optimizer.step()
            
            #progress bar에 loss 정보 추가
            tepoch.set_postfix(loss=loss.item())
        

        # tensorboard --logdir=runs --port=8000
        writer.add_scalar('Loss/Train', total_loss/total_batch, epoch) # -> tot_loss/1563 (for문이 1563번 도니까)
        writer.add_scalar('Accuracy/Train', total_acc/total_train*100, epoch) # -> tot_acc/50000 (total_acc는 input 전체 값 더해지니까 /50000)

        # validation check
        with torch.no_grad(): 
            model.eval() # eval() -> update (X)
            for data in val_data_loader: 
                inputs, labels = data
                predicted_outputs = model(inputs) 
                val_loss = loss_fn(predicted_outputs, labels) 


size = len(test_data_loader.dataset) 
num_batches = len(test_data_loader)
#모델 평가 모드 - model.eval() => dropout, normalization 제외 
model.eval()
total_test_loss = 0.0
total_test_accuracy = 0.0
with torch.no_grad():
    for data, target in test_data_loader:
        pred = model(data)
        test_loss = loss_fn(pred, target).item()
        
        total_test_loss +=  test_loss
        total_test_accuracy += (pred.argmax(1)==target).type(torch.float).sum().item()  #dim=1: 행에서 가장 큰 값의 idx return 

total_test_loss /= size
total_test_accuracy /= size
print(f"=====Test Error===== \nAccuracy: {(100*total_test_accuracy):>0.1f}%, Avg loss: {total_test_loss:>8f} \n")


writer.close()"""