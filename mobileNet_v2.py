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
from torch import tensor
import warnings
warnings.filterwarnings(action='ignore') # warning 무시

# class별로 출력
#from sklearn.metrics import classification_report

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("MobileNetv2_logs")
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import glob
import math
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall

transform = transforms.Compose([          
    transforms.ToTensor()])

sum = 0
cnt = 0



#ccsv = sorted(glob.glob('C:/Users/QR22002/Desktop/chaeyun/dataset/Holder_name/**/*.csv', recursive=True)) #116528
#png_file_path = sorted(glob.glob('C:/Users/QR22002/Desktop/chaeyun/dataset/Holder_name/**/*.png', recursive=True)) #116526
# ppng = sorted(glob.glob('C:/Users/QR22002/Desktop/chaeyun/dataset/Holder_name/**/*.png', recursive=True)) #116526

#Test dataset 저장
png_file_path = sorted(glob.glob('C:/Users/QR22002/Desktop/chaeyun/dataset/hold_name_test/**/*.png', recursive=True)) #116528

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

"""        
# Train data 추출
""" 
# csv_file_path = []
# for i in range(len(png_file_path)):
#     tmp_csv_name = os.path.splitext(png_file_path[i])
#     csv_file_path.append(tmp_csv_name[0]+'.csv')

# print(f'csv file: {len(csv_file_path)}')
# print(f'png file: {len(png_file_path)}')

# length = len(png_file_path)
# for idx in range(length):
#     tmp_csv_name = os.path.splitext(csv_file_path[idx])   #split => return list
#     tmp_png_name = os.path.splitext(png_file_path[idx])
#     if(tmp_csv_name[0] == tmp_png_name[0]):
#         data = pd.read_csv(csv_file_path[idx], index_col=False, header=None, sep=';') #header=None(주의!!)
#         image = cv2.imread(png_file_path[idx])
#         image = transform(image)
#         #print(data[1][0])   #열-행
#         len = data.shape
#         etc = data[5][0]
        
#         for i in range(len[0]):
#             cnt+=1
#             tmpLabel = data[0][i]
#             #print(tmpLabel)
#             if type(tmpLabel) is str:
#                 if etc == -1:
#                     label = 100 #blank
#                 else:
#                     label = ord(tmpLabel)
#             elif type(tmpLabel) is int:
#                 if etc is int:
#                     label = tmpLabel
#                 elif type(etc) is float:
#                      if 1 <= tmpLabel and tmpLabel <= 26:
#                         label = tmpLabel+64
#                 elif etc == ' ':
#                      if 1 <= tmpLabel and tmpLabel <= 26:
#                         label = tmpLabel+64
#             elif type(tmpLabel) is float:
#                 label = 100
            
#             pos = []
#             for j in range(1, len[1]):
#                 pos.append(data[j][i])
#             # 이미지 편집
#             cropped_img = torchvision.transforms.functional.crop(image, pos[1]-5, pos[0]-5, pos[3]+10, pos[2]+15)   # 자르기
#             resize = torchvision.transforms.Resize((220, 220)) # 확대
#             enlarged_img = resize(cropped_img)
#             padded_img = torchvision.transforms.functional.pad(enlarged_img, (2,2,2,2), fill=0)  # padding
#             transform_to_img = transforms.ToPILImage()
#             img = transform_to_img(padded_img)
            
#             #tmpPath = f'C:/Users/QR22002/Desktop/chaeyun/dataset/train/{label}/'
#             tmpPath = f'C:/Users/QR22002/Desktop/chaeyun/dataset/test/{label}/'
#             if not os.path.exists(tmpPath):
#                 os.makedirs(tmpPath)
#             #img.save(f'C:/Users/QR22002/Desktop/chaeyun/dataset/train/{label}/' + f'{os.path.basename(tmp_csv_name[0])}' + '_' + f'{cnt}.png')
#             img.save(f'C:/Users/QR22002/Desktop/chaeyun/dataset/test/{label}/' + f'{os.path.basename(tmp_csv_name[0])}' + '_' + f'{cnt}.png')
#         cnt = 0
#     print(f"done: index: {idx} {os.path.basename(tmp_csv_name[0])}")

"""
# Test data 추출

print(f'{len(png_file_path)}')
csv_file_path = []
for i in range(len(png_file_path)):
    tmp_csv_name = os.path.splitext(png_file_path[i])
    csv_file_path.append(tmp_csv_name[0]+'.csv')

print(f'csv file: {len(csv_file_path)}')
print(f'png file: {len(png_file_path)}')

length = len(png_file_path)
for idx in range(7225, length):
    tmp_csv_name = os.path.splitext(csv_file_path[idx])   #split => return list
    tmp_png_name = os.path.splitext(png_file_path[idx])
    if(tmp_csv_name[0] == tmp_png_name[0]):
        data = pd.read_csv(csv_file_path[idx], index_col=False, header=None) #header=None(주의!!)
        image = cv2.imread(png_file_path[idx])
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
            cropped_img = torchvision.transforms.functional.crop(image, pos[1]-5, pos[0]-5, pos[3]+10, pos[2]+15)   # 자르기
            resize = torchvision.transforms.Resize((220, 220)) # 확대
            enlarged_img = resize(cropped_img)
            padded_img = torchvision.transforms.functional.pad(enlarged_img, (2,2,2,2), fill=0)  # padding
            transform_to_img = transforms.ToPILImage()
            img = transform_to_img(padded_img)
            
            #tmpPath = f'C:/Users/QR22002/Desktop/chaeyun/dataset/train/{label}/'
            tmpPath = f'C:/Users/QR22002/Desktop/chaeyun/dataset/test/{label}/'
            if not os.path.exists(tmpPath):
                os.makedirs(tmpPath)
            #img.save(f'C:/Users/QR22002/Desktop/chaeyun/dataset/train/{label}/' + f'{os.path.basename(tmp_csv_name[0])}' + '_' + f'{cnt}.png')
            img.save(f'C:/Users/QR22002/Desktop/chaeyun/dataset/test/{label}/' + f'{os.path.basename(tmp_csv_name[0])}' + '_' + f'{cnt}.png')
        cnt = 0
    print(f"done: index: {idx} {os.path.basename(tmp_csv_name[0])}")
""" 

#========MobileNet V2========#

device = (
    "cuda"
    if torch.cuda.is_available()
    else "gpu"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

num_classes = 30

def conv3x3(in_channels, out_channels, stride):
    nn.Sequential(
            nn.Conv2d(kernel_size=3, in_channels=in_channels, out_channels=out_channels , padding = 1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6())

def conv1x1(in_channels, out_channels):
    nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=in_channels, out_channels=out_channels ,stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU6())

class bottleNeckResidualBlock (nn.Module):
    # initialize
    def __init__(self, in_channels, out_channels, stride, t): # t = expansion factor
        super().__init__()
        assert stride in [1,2] #예외처리 (stride는 1 또는 2)
        
        widen = round(in_channels*t)
        self.canSkipConnection = stride == 1 and in_channels == out_channels

        # widen 안되는 경우
        if t == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=widen, out_channels=widen, kernel_size=3, stride=stride, padding=1, groups=widen),
                nn.BatchNorm2d(widen), #input size
                nn.ReLU6(),
                nn.Conv2d(in_channels=widen, out_channels=out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels)
            )
        
        # widen 되는 경우
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=widen, kernel_size=1, stride=1),
                nn.BatchNorm2d(widen), #input size
                nn.ReLU6(),
                nn.Conv2d(in_channels=widen, out_channels=widen, kernel_size=3, stride=stride, padding=1, groups=widen),
                nn.BatchNorm2d(widen),
                nn.ReLU6(),
                nn.Conv2d(in_channels=widen, out_channels=out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels)
                # nn.Dropout2d(0.3)
            )
        
    # skip connection 할지 말지
    def forward(self, x):
        if self.canSkipConnection:
            return self.conv(x) + x
        else:
            return self.conv(x)
              
class MobileNet_v2(nn.Module):
    def __init__(self, classes=num_classes):
        super().__init__()
        self.param = [ 
            # 논문에 나온 parameter 순서: 
            # [t, out_channels, repeat, stride]
            

	        [1, 16, 1 ,1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]
        ]

        resBlock = bottleNeckResidualBlock
        self.mobileNetV2_layers = [conv3x3(3, 32, 2)]	
        in_channel = 32
        for t, c, n, s in self.param:
            for i in range(n):
                self.mobileNetV2_layers.append(resBlock(in_channel, c, s, t))
                in_channel = c
        # self.mobileNetV2_layers = nn.Sequential(*mobileNetV2_layers) 
        self.layers = nn.Sequential(*self.mobileNetV2_layers)
        out_channels = 1280
        self.lastconv = conv1x1(in_channel, out_channels)
        self.avgPool = nn.AdaptiveAvgPool2d((1,1))
        self.fc_layer = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features=out_channels, out_features=classes))
    
    def forward(self, x):
        out = self.layers(x)  # error 발생 부분
        out = self.lastconv(out)
        out = self.avgPool(out)
        out = self.fc_layer(out)
        return out



#data augmentation
aug = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=0, shear=0.5),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor()
])

# def imgAugmentation(img):
#     image = transform(img)
#     cropped_img = torchvision.transforms.functional.crop(image, 2, 2, 220, 170)
#     resize = torchvision.transforms.Resize((216, 216))
#     enlarged_img = resize(cropped_img)
#     padding = torchvision.transforms.functional.pad(enlarged_img, (4,4,4,4), fill=0)
#     augment = aug(padding)
#     toImg = transforms.ToPILImage()
#     outImg = toImg(augment)
#     return outImg

  

batch_size = 128

train_data = ImageFolder(root='C:/Users/QR22002/Desktop/chaeyun/dataset/dataset/train',
                         transform=aug)
train_data_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

val_data = ImageFolder(root='C:/Users/QR22002/Desktop/chaeyun/dataset/dataset/validation',
                       transform=aug)
val_data_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

#Model 불러오기
model = MobileNet_v2().to(device)

#Tensorflow에서 model.compile() 부분에 해당
loss_fn = torch.nn.CrossEntropyLoss().to(device)   #Softmax 함수 & Negative Log Liklihood(NLL)까지 포함되어 있음
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.00004).to(device) 

#Accuracy - Precision & Recall
precision = MulticlassPrecision(num_classes=num_classes).to(device)
recall = MulticlassRecall(num_classes=num_classes).to(device)

total_batch = len(train_data_loader)
total_train = len(train_data)
epoch_size = 100

print(f'len: train data: {len(train_data_loader)}') #7976
for epoch in range(epoch_size): 
    total = 0.0
    running_accuracy = 0.0
    running_vall_loss = 0.0 
    total_loss, total_acc = 0.0, 0.0
    cnt = 0
    with tqdm(train_data_loader, unit="batch") as tepoch: 
        for it, (X, Y) in enumerate(tepoch):
            model.train()	#train 단계임을 명시
            tepoch.set_description(f"Epoch {epoch+1}")
            
            prediction = model(X)
            loss = loss_fn(prediction, Y)   # Y: label
            
            total_loss += loss.item()
            
            acc_precision = precision(prediction, Y)
            print(f'Precision on batch {tepoch}: {acc_precision}')
            
            acc_recall = recall(prediction, Y)
            print(f'Recall on batch {tepoch}: {acc_recall}')
            
            #Backpropagation
            optimizer.zero_grad()
            loss.backward()
            
            #parameter update
            optimizer.step()
            
            #progress bar에 loss 정보 추가
            tepoch.set_postfix(loss=loss.item())
        

        acc_precision = precision.compute()
        acc_recall = recall.compute()
        # tensorboard --logdir=runs --port=8000
        writer.add_scalar('Loss/Train', total_loss/total_batch, epoch) 
        writer.add_scalar('Precision/Train', acc_precision, epoch)
        writer.add_scalar('Recall/Train', acc_recall, epoch)

        precision.reset()
        recall.reset()

        val_acc_precision = MulticlassPrecision(num_classes=num_classes).to(device)
        val_acc_recall = MulticlassRecall(num_classes=num_classes).to(device)
        # validation check
        with torch.no_grad(): 
            model.eval() # eval() -> update (X)
            for it, (inputs, labels) in enumerate(val_data_loader): 
                predicted_outputs = model(inputs) 
                val_loss = loss_fn(predicted_outputs, labels) 
                
                # The label with the highest value will be our prediction
                _, predicted = torch.max(predicted_outputs, dim=1)
                running_vall_loss += val_loss.item()
                total += labels.size(0)
                running_accuracy += (predicted == labels).sum().item()
                val_precision = val_acc_precision(predicted_outputs, labels)
                vall_recall = val_acc_recall(predicted_outputs, labels)
                
        val_loss_value = running_vall_loss/len(val_data_loader)
        accuracy = (running_accuracy / total)
        val_precision = val_acc_precision.compute()
        vall_recall = val_acc_recall.compute()
        writer.add_scalar('Loss/Validation', val_loss_value, epoch)
        writer.add_scalar('Accuracy/Validation', accuracy, epoch)
        writer.add_scalar('Precision/Validation', val_precision, epoch)
        writer.add_scalar('Recall/Validation', vall_recall, epoch)
        val_acc_precision.reset()
        val_acc_recall.reset()

# 모델 저장
torch.save(model.state_dict(), 'model_state_dict.pt')
model = MobileNet_v2()
model.load_state_dict('model_state_dict.pt')

class_list = ['100', '44',	'45',	'46',   '65',	
              '66',	'67',	'68',	'69',	'70',	'71',	'72',	'73',	'74',	
              '75',	'76',	'77',	'78',	'79',	'80',	'81',	
              '82',	'83',	'84',	'85',	'86',	'87',	'88',	'89',	'90']

for i in class_list:
    path = f'C:/Users/QR22002/Desktop/chaeyun/dataset/dataset/test/'+class_list
    test_data = ImageFolder(root=path)
    test_data_loader = DataLoader(dataset=test_data)
    
    size = len(test_data_loader.dataset) 
    num_batches = len(test_data_loader)
    #모델 평가 모드 - model.eval() => dropout, normalization 제외 
    model.eval()
    total_test_loss = 0.0
    total_test_accuracy = 0.0
    test_acc_precision = MulticlassPrecision(num_classes=num_classes).to(device)
    test_acc_recall = MulticlassRecall(num_classes=num_classes).to(device)


    with torch.no_grad():
        for it, (data, target) in enumerate(test_data_loader):
            pred = model(data)
            test_loss = loss_fn(pred, target).item()
            
            test_precision = test_acc_precision(pred, target)
            test_recall = test_acc_recall(pred, target)
            total_test_loss +=  test_loss
            total_test_accuracy += (pred.argmax(1)==target).type(torch.float).sum().item()  #dim=1: 행에서 가장 큰 값의 idx return 

    total_test_loss /= size
    total_test_accuracy /= size
    total_acc_precision = test_precision.compute()
    total_acc_recall = test_recall.compute()
    print(f"=====Test Error===== \nAccuracy: {(total_test_accuracy):>8f}, Avg loss: {total_test_loss:>8f} \n")
    print(f"=====Test Error===== \nAccuracy: {(total_acc_precision):>8f}, Avg loss: {total_test_loss:>8f} \n")
    print(f"=====Test Error===== \nAccuracy: {(total_acc_recall):>8f}, Avg loss: {total_test_loss:>8f} \n")

    test_precision.reset()
    test_recall.reset()

writer.close()
