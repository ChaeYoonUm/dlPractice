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


#ccsv = sorted(glob.glob('C:/Users/QR22002/Desktop/chaeyun/dataset/Holder_name/**/*.csv', recursive=True)) #116528
png_file_path = sorted(glob.glob('C:/Users/QR22002/Desktop/chaeyun/dataset/Holder_name/**/*.png', recursive=True)) #116526
# ppng = sorted(glob.glob('C:/Users/QR22002/Desktop/chaeyun/dataset/Holder_name/**/*.png', recursive=True)) #116526

#Test dataset 저장
#ccsv = sorted(glob.glob('C:/Users/QR22002/Desktop/chaeyun/dataset/Holder_name_test/**/*.csv', recursive=True)) #116528

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
        

csv_file_path = []
for i in range(len(png_file_path)):
    tmp_csv_name = os.path.splitext(png_file_path[i])
    csv_file_path.append(tmp_csv_name[0]+'.csv')

print(f'csv file: {len(csv_file_path)}')
print(f'png file: {len(png_file_path)}')

length = len(png_file_path)
for idx in range(length):
    tmp_csv_name = os.path.splitext(csv_file_path[idx])   #split => return list
    tmp_png_name = os.path.splitext(png_file_path[idx])
    if(tmp_csv_name[0] == tmp_png_name[0]):
        data = pd.read_csv(csv_file_path[idx], index_col=False, header=None, sep=';') #header=None(주의!!)
        image = cv2.imread(png_file_path[idx])
        image = transform(image)
        #print(data[1][0])   #열-행
        len = data.shape
        etc = data[5][0]
        
        for i in range(len[0]):
            cnt+=1
            tmpLabel = data[0][i]
            if tmpLabel.isdigit() and etc.isdigit():
                label = tmpLabel
            elif tmpLabel.isdigit() and etc == '':
                if 1 <= tmpLabel and tmpLabel <= 26:
                    label = tmpLabel+64
            elif tmpLabel.isalpha():
                if 'A' <= tmpLabel and tmpLabel <= 'Z':
                    label = ord(tmpLabel)
            elif tmpLabel == '.':
                label = ord('.')
            elif tmpLabel == ',':
                label = ord(',')
            elif tmpLabel == ' ':
                label = 100
            
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
            
            tmpPath = f'C:/Users/QR22002/Desktop/chaeyun/dataset/train/{label}/'
            if not os.path.exists(tmpPath):
                os.makedirs(tmpPath)
            img.save(f'C:/Users/QR22002/Desktop/chaeyun/dataset/train/{label}/' + f'{os.path.basename(tmp_csv_name[0])}' + '_' + f'{cnt}.png')
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


def conv3x3(in_channels, out_channels, stride):
    nn.Sequential(
            nn.Conv2d(kernel_size=3, in_channels=in_channels, out_channels=out_channels , padding = 1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(),
            nn.Dropout2d(0.2))

def conv1x1(in_channels, out_channels):
    nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=in_channels, out_channels=out_channels ,stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU6(),
            nn.Dropout2d(0.2))

class bottleNeckResidualBlock (nn.Module):
    def __init__(self, in_channels, out_channels, stride, t): # t = expansion factor
        super(bottleNeckResidualBlock, self).__init__()
        stride = [1,2]
        
        widen = round(in_channels*t)
        canSkipConnection = stride == 1 and in_channels == out_channels

        # widen 안되는 경우
        if t == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=widen, out_channels=widen, kernel_size=3, stride=1, dilation=1),
                nn.BatchNorm2d(widen),
                nn.ReLU6(),
                nn.Conv2d(in_channels=widen, out_channels=out_channels, kernel_size=1, stride=1, ),
                nn.BatchNorm2d(widen),
                nn.Dropout2d(0.3)
            )
        
        # widen 되는 경우
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=widen, kernel_size=1, stride=1),
                nn.BatchNorm2d(widen),
                nn.ReLU6(),
                nn.Conv2d(in_channels=widen, out_channels=widen, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(widen),
                nn.ReLU6(),
                nn.Conv2d(in_channels=widen, out_channels=out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(widen),
                nn.Dropout2d(0.3)
            )
        
    # skip connection 할지 말지
    def forward(self, x):
        if self.identity:
            return self.conv(x) + x
        else:
            return self.conv()
              
class MobileNet_v2(nn.Module):
    def __init__(self, classes=39):
        super(MobileNet_v2, self).__init__()
        param = [ 
            # 논문에 나온 parameter 순서: 
            # [t, out_channels, repeat, stride]
            [1, 16, 1 ,2],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]
        ]

        resBlock = bottleNeckResidualBlock
        mobileNetV2_layers = [conv3x3(3, 32, 2)]
        in_channel = 32
        for t, c, n, s in param:
            for i in range(n):
                mobileNetV2_layers.append(resBlock(in_channel, c, s, t))
                in_channel = c
        self.wholeLayers = nn.Sequential(mobileNetV2_layers)      
        out_channels = 1280
        self.conv = conv1x1(in_channel, out_channels)
        self.avgPool = nn.AdaptiveAvgPool2d((1,1))
        self.fc_layer = nn.Linear(out_channels, classes)
    
    def forward(self, x):
        x = self.wholeLayers(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = self.fc_layer(x)
        return x

train_data = ImageFolder(root='C:/Users/QR22002/Desktop/chaeyun/dataset/train')
train_data_loader = DataLoader(dataset=train_data, batch_size=64)
#train_data, val_data = torch.utils.data.random_split(train_data, [50000, 10000])

test_data = ImageFolder(root='C:/Users/QR22002/Desktop/chaeyun/dataset/test')
test_data_loader = DataLoader(dataset=test_data, batch_size=64)

batch_size = 128
data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
#val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)


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
    running_val_loss = 0.0 
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
        writer.add_scalar('Loss/Train', total_loss/total_batch, epoch) 
        writer.add_scalar('Accuracy/Train', total_acc/total_train*100, epoch)

        # validation check
        # with torch.no_grad(): 
        #     model.eval() # eval() -> update (X)
        #     for data in val_data_loader: 
        #         inputs, labels = data
        #         predicted_outputs = model(inputs) 
        #         val_loss = loss_fn(predicted_outputs, labels) 

# 모델 저장
torch.save(model.state_dict(), 'model_state_dict.pt')
model = MobileNet_v2()
model.load_state_dict('model_state_dict.pt')

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


writer.close()
