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
import warnings
warnings.filterwarnings(action='ignore') # warning 무시
import utils
# class별로 출력
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("MobileNetv2_logs_custom")
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

#Train dataset 저장
train_png_file_path = sorted(glob.glob('C:/Users/QR22002/Desktop/chaeyun/dataset/Holder_name/**/*.png', recursive=True)) #116526

#Test dataset 저장
test_png_file_path = sorted(glob.glob('C:/Users/QR22002/Desktop/chaeyun/dataset/hold_name_test/**/*.png', recursive=True)) #116528

origin_train_path = sorted(glob.glob('C:/Users/QR22002/Desktop/chaeyun/dataset/dataset/train/**/*.png', recursive=True))

origin_test_path = sorted(glob.glob('C:/Users/QR22002/Desktop/chaeyun/dataset/dataset/test/**/*.png', recursive=True))

origin_val_path = sorted(glob.glob('C:/Users/QR22002/Desktop/chaeyun/dataset/dataset/validation/**/*.png', recursive=True))

custom_val_path = sorted(glob.glob('C:/Users/QR22002/Desktop/chaeyun/custom_dataset/validation/**/*.png', recursive=True))

     
# ==== PNG 없는 파일 ====#
#['C:/Users/QR22002/Desktop/chaeyun/dataset/Holder_name\\hold_name_0430\\Holder_name_alpabet\\0293_9445411545060702\\00174_16_9445411545060702_0820_PARKHAECHUL_cardBox', 
# 'C:/Users/QR22002/Desktop/chaeyun/dataset/Holder_name\\hold_name_0510\\InsufficientAlphabet_Set\\TextFrame', 
# 'C:/Users/QR22002/Desktop/chaeyun/dataset/Holder_name\\hold_name_0409\\hold_name_new\\0346\\001087_100000lux_cardBox', 
# 'C:/Users/QR22002/Desktop/chaeyun/dataset/Holder_name\\hold_name_0430\\Holder_name_alpabet\\0293_9445411545060702\\00176_16_9445411545060702_0820_PARKHAECHUL_cardBox']

#hold_name = sorted(glob.glob('C:/Users/QR22002/Desktop/chaeyun/dataset/Holder_name/hold_name_0409/hold_name_new/0008/*.png', recursive=True))

"""        
# Train data 추출
""" 
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
exit()

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





#========MobileNet V2========#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

num_classes = 30

class bottleNeckResidualBlock (nn.Module):
    # initialize
    def __init__(self, in_channels, out_channels, t, stride=1): # t = expansion factor
        super().__init__()
        
        #assert stride in [1,2]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        
        expand = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * t, 1, bias = False),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace = True),
            
        )
        depthwise = nn.Sequential(
            nn.Conv2d(in_channels * t, in_channels * t, 3, stride = stride, padding = 1, groups = in_channels * t, bias = False),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace = True),
        )
        pointwise = nn.Sequential(
            nn.Conv2d(in_channels * t, out_channels, 1, bias = False),
            nn.BatchNorm2d(out_channels),
        )
        
        residual_list = []
        if t > 1:
            residual_list += [expand]
        residual_list += [depthwise, pointwise]
        self.residual = nn.Sequential(*residual_list)
    
    def forward(self, x):
        if self.stride == 1 and self.in_channels == self.out_channels:
            out = self.residual(x) + x
        else:
            out = self.residual(x)
    
        return out
              
class MobileNet_v2(nn.Module):
    def __init__(self, n_classes = num_classes):
        super().__init__()

        self.first_conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace = True),
        )

        self.bottlenecks = nn.Sequential(
            self.make_stage(16, 8, t = 1, n = 1),
            self.make_stage(8, 12, t = 6, n = 2, stride = 1),
            self.make_stage(12, 16, t = 6, n = 3, stride = 2),
            self.make_stage(16, 32, t = 6, n = 4, stride = 1),
            self.make_stage(32, 68, t = 6, n = 3),
            self.make_stage(68, 80, t = 6, n = 3, stride = 2),
            self.make_stage(80, 160, t = 6, n = 1)
        )

        self.last_conv = nn.Sequential(
            nn.Conv2d(160, 320, 1, bias = False),
            nn.BatchNorm2d(320),
            nn.ReLU6(inplace = True),
            nn.Dropout(0.2)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
        	nn.Dropout(0.2),
            nn.Linear(320, n_classes)
        )
    
    def forward(self, x):
        x = self.first_conv(x)
        x = self.bottlenecks(x)
        x = self.last_conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1) 
        x = self.fc(x)
        return x
    
    def make_stage(self, in_channels, out_channels, t, n, stride = 1):
        layers = [bottleNeckResidualBlock(in_channels, out_channels, t, stride)]
        in_channels = out_channels
        for _ in range(n):
            layers.append(bottleNeckResidualBlock(in_channels, out_channels, t))
        
        return nn.Sequential(*layers)

# # 모델 확인 
# from torchinfo import summary
# mobilenet_v2 = models.mobilenet_v2()
# summary(mobilenet_v2, (2,3,224,224), device="cpu")


#data augmentation
aug = transforms.Compose([
    # transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAutocontrast(),
    transforms.RandomAffine(degrees=0, shear=0.5),
    transforms.RandomHorizontalFlip(0.5),
    # transforms.Grayscale(),
    transforms.ToTensor()
])

toTensor = transforms.Compose([
    transforms.ToTensor()
])


def evaluate_for_video(dataloader, net, idx_to_class):
    target_all = [[] for _ in range(num_classes)]
    pred_all = [[] for _ in range(num_classes)]
 
    target_all2 = []
    pred_all2 = []
    
    # device = torch.device('cuda')
    # net = MobileNet_v2()
    # path = PATH
    # net.load_state_dict(torch.load(path))
    # net.to(device)
 
    with torch.no_grad():
        best_model.eval()
        for it, (inputs, targets) in enumerate(tqdm(dataloader)):
            inputs = inputs.to(device)
            targets = targets.to(device)
    
            # Forward pass
            outputs = net(inputs)
            # print(outputs)
            
            _, pred = outputs.topk(1, dim=1, largest=True, sorted=True)
            preds = pred.t().view(-1)
    
            for ii in range(len(outputs)):
                # label = targets[ii]
                # target_all[label].append(targets[ii].cpu().item())
                # pred_all[label].append(preds[ii].cpu().item())
    
                pred_all2.append(preds[ii].cpu().item())
                target_all2.append(targets[ii].cpu().item())
           
    for i in range(num_classes):
        class_name = idx_to_class[i]
        print(f"idx-class_name {i} {class_name}")
    #     utils.get_confustion_matrix_score(class_name, target_all[i],  pred_all[i])
       
    utils.get_classification_report(target_all2, pred_all2)
    return
 
 
def test(model, test_dir='Dataset/OCR_HolderName/test', datalist=None):
    #testdir = os.path.join(args.data, 'test')
    testdir = test_dir
    test_dataset = datasets.ImageFolder(
            testdir,
            transforms.Compose([
                transforms.ToTensor()
        ]))
    test_loader = torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size = batch_size,
                    shuffle = False,    
                    num_workers = 4,
                    pin_memory = True)
    class_to_idx = test_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    evaluate_for_video(test_loader, model, idx_to_class)
    return
 
 
def get_classification_report(target_all, pred_all):
    print("====================== classification_report ==============================")
    print(classification_report(target_all, pred_all))
    print("===================================================================")
    data = []
    for i in range(len(target_all)):
        row = []
        row.append(target_all[i])
        row.append(pred_all[i])
        data.append(row)
 
    csv_file_path = 'result.csv'
    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        for row in data:
            writer.writerow(row)   
    return
 
def get_confustion_matrix_score(class_name, pred_all, target_all):
 
    # print(pred_all)
    # print(target_all)
    # try :
    #     tn, fp, fn, tp = confusion_matrix(target_all, pred_all, 30)
    # except :
    #     ret_conf = confusion_matrix(target_all, pred_all, 30)
    #     print(f"Exception error :{ret_conf}")
    #     tn = None
    #     fp = None
    #     fn = None
    #     tp = None
   
    print(f" ================== {class_name} ==================")
    print(classification_report(target_all, pred_all))
    precision = precision_score(target_all, pred_all, average='None')
    recall = recall_score(target_all, pred_all, average='None')
    # precision, recall = (tp / (tp+fp), tp / (tp+fn))
    # print(f"class_name : {class_name} precision:{precision} \t recall:{recall} \t tn:{tn} \t fp:{fp} \t fn:{fn} \t tp:{tp}")
    print(f"class_name : {class_name} precision:{precision} \t recall:{recall}")
    return # precision, recall, tn, fp, fn, tp

batch_size = 128

train_data = ImageFolder(root='Dataset/OCR_HolderName/train', transform=aug)
train_data_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)


val_data = ImageFolder(root='Dataset/OCR_HolderName/validation', transform=toTensor)
val_data_loader = DataLoader(dataset=val_data, batch_size=batch_size)

#Model 불러오기
model = MobileNet_v2().to(device)

#Tensorflow에서 model.compile() 부분에 해당
loss_fn = torch.nn.CrossEntropyLoss().to(device)   #Softmax 함수 & Negative Log Liklihood(NLL)까지 포함되어 있음
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3, momentum=0.9, eps=0.002, weight_decay=0.00004 ) #learning rate = 0.001 Good ,weight_decay=0.00004

total_batch = len(train_data_loader)
total_train = len(train_data)
epoch_size = 100

print(f'len: train data: {len(train_data_loader)}')
print(f'len: validation data: {len(val_data_loader)}')

# torch.save(model.state_dict(), '../checkpoint/model_state_dict.pt')
# torch.save(model.state_dict(), '../checkpoint/model_state_dict.pt')
best_model = MobileNet_v2().to(device)
PATH = 'mobileNetv2/checkpoint/mobileNetv2/model_state_dict_48.tar'
checkpoint = torch.load(PATH)
best_model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
# best_model.eval()
test(best_model)
exit()

for epoch in range(epoch_size): 
    total = 0.0
    running_accuracy = 0.0
    running_vall_loss = 0.0 
    total_loss, total_acc = 0.0, 0.0
    cnt = 0
    with tqdm(train_data_loader, unit="batch") as tepoch: #progress bar, batch=128 -> 7976번 = 1epoch
        for it, (X, Y) in enumerate(tepoch):
            X, Y = X.cuda(),Y.cuda()

            model.train()#train 단계임을 명시
            tepoch.set_description(f"Epoch {epoch+1}")
            
            prediction = model(X)
            loss = loss_fn(prediction, Y)   # Y: label
            
            total_loss += loss.item()
            total_acc += (prediction.argmax(1)==Y).sum().item() #argmax(1) -> [[], []] 각각의 배열 set 마다의 최댓값 index 반환 
            # argmax(0) -> 열 마다 최댓값
            # argmax(1) -> 행 마다 최댓값
            
            #Backpropagation
            optimizer.zero_grad()
            loss.backward()
            
            #parameter update
            optimizer.step()
            
            #progress bar에 loss 정보 추가
            tepoch.set_postfix(loss=loss.item())
        
        # tensorboard --logdir=runs --port=8000
        # tensorboard --logdir=MobileNetv2_logs --port=8000 --host 192.168.0.109
        writer.add_scalar('Loss/Train', total_loss/total_batch, epoch)  #batch당 loss
        writer.add_scalar('Accuracy/Train', total_acc/total_train*100, epoch)

        # validation check
        print("Validation")
        with torch.no_grad(): 
            model.eval() # eval() -> update (X)
            with tqdm(val_data_loader, unit="batch") as valEpoch:
                for val_it, (inputs, labels) in enumerate(valEpoch): 
                    inputs, labels = inputs.cuda(), labels.cuda()
                    valEpoch.set_description(f"Validation Progress")
                    
                    predicted_outputs = model(inputs) 
                    val_loss = loss_fn(predicted_outputs, labels) 
                    
                    # The label with the highest value will be our prediction
                    _, predicted = torch.max(predicted_outputs, dim=1)
                    running_vall_loss += val_loss.item()
                    total += labels.size(0) # 0: 행 개수, 1: 열 개수 
                    running_accuracy += (predicted == labels).sum().item()
                    valEpoch.set_postfix(loss=val_loss.item())
                
        val_loss_value = running_vall_loss/len(val_data_loader)
        accuracy = (100 * running_accuracy / total) # divided by the total num of predictions done
        writer.add_scalar('Loss/Validation', val_loss_value, epoch)
        writer.add_scalar('Accuracy/Validation', accuracy, epoch)
    #torch.save(model, '../checkpoint/mobileNetv2/model_state_dict_%d.pt'%(epoch+1))
    torch.save({
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'model_state_dict': model.state_dict(), 
        'loss': total_loss}, 'checkpoint/mobileNetv2/model_state_dict_%d.tar'%(epoch+1))

# 모델 저장
#torch.save(model.state_dict(), 'checkpoint/model_state_dict.pt')
# torch.save(model.state_dict(), 'model_state_dict.pth')
# torch.save(model, 'model_state_dict.pth')

# model = MobileNet_v2()
# model.load_state_dict('model_state_dict_48.pth')
# model.eval()
# model = torch.load('model.pth')

# 저장된 모델로 test 돌리기
# model = torch.load('checkpoint/model_state_dict_.pt')
# model.eval()
# checkpoint = torch.load('checkpoint/model_state_dict_.pt')
# model = MobileNet_v2()
# model.load_state_dict(checkpoint['model'])

writer.close()