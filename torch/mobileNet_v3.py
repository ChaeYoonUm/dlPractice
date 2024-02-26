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
writer = SummaryWriter("MobileNetv3_logs")
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
png_file_path = sorted(glob.glob('../Dataset/OCR_HolderName/**/*.png', recursive=True)) #116528

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
            Block(3, 16, 64, 24, nn.ReLU(inplace=True), None, 2),
            Block(3, 24, 72, 24, nn.ReLU(inplace=True), None, 1),
            Block(5, 24, 72, 40, nn.ReLU(inplace=True), SeModule(40), 2),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
            Block(3, 40, 240, 80, hswish(), None, 2),
            Block(3, 80, 200, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 480, 112, hswish(), SeModule(112), 1),
            Block(3, 112, 672, 112, hswish(), SeModule(112), 1),
            Block(5, 112, 672, 160, hswish(), SeModule(160), 1),
            Block(5, 160, 672, 160, hswish(), SeModule(160), 2),
            Block(5, 160, 960, 160, hswish(), SeModule(160), 1),
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
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.hs3(self.bn3(self.linear3(out)))
        out = self.linear4(out)
        return out



from torchinfo import summary
# 모델 확인 
from torchvision import models
model = MobileNetV3_Large()
summary(model, (2,3,224,224), device="cpu")
exit()

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


# precision_score = MulticlassPrecision(num_classes=num_classes).to(device)
# recall_score = MulticlassRecall(num_classes=num_classes).to(device)
# f1_score = torcheval.metrics.functional.multiclass_f1_score(num_classes=num_classes)
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
       
    get_classification_report(target_all2, pred_all2)
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
model = MobileNetV3_Large().to(device)

# torch.tensor(..., device="cuda") 
# torch.tensor(...).cuda() 
# torch.tensor(...).to("cuda")

#Tensorflow에서 model.compile() 부분에 해당
loss_fn = torch.nn.CrossEntropyLoss().to(device)   #Softmax 함수 & Negative Log Liklihood(NLL)까지 포함되어 있음
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3, momentum=0.9, eps=0.002, weight_decay=0.00004 ) #learning rate = 0.001 Good ,weight_decay=0.00004

#Accuracy - Precision & Recall
# precision = MulticlassPrecision(num_classes=num_classes).to(device)
# recall = MulticlassRecall(num_classes=num_classes).to(device)

total_batch = len(train_data_loader)
total_train = len(train_data)
epoch_size = 100

print(f'len: train data: {len(train_data_loader)}')
print(f'len: validation data: {len(val_data_loader)}')

best_model = MobileNetV3_Large().to(device)
PATH = 'mobileNetv3/checkpoint/model_state_dict_20.tar'
# best_model.load_state_dict(torch.load())
checkpoint = torch.load(PATH)
best_model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
# best_model.eval()
test(best_model)
exit()

"""
# torch.save(model.state_dict(), './checkpoint/model_state_dict.pt')
# torch.save(model.state_dict(), '../checkpoint/model_state_dict.pt')
best_model = mobilenet_v3_large().to(device)
PATH = 'mobileNetv2/checkpoint/mobileNetv2/model_state_dict_48.tar'
# best_model.load_state_dict(torch.load())
checkpoint = torch.load(PATH)
best_model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
# best_model.eval()
test(best_model)"""

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
            #acc_precision = precision(prediction, Y)
            #print(f'Precision on batch {tepoch}: {acc_precision}')
            
            #acc_recall = recall(prediction, Y)
            #print(f'Recall on batch {tepoch}: {acc_recall}')
            
            #Backpropagation
            optimizer.zero_grad()
            loss.backward()
            
            #parameter update
            optimizer.step()
            
            #progress bar에 loss 정보 추가
            tepoch.set_postfix(loss=loss.item())
        

        #acc_precision = precision.compute()
        #acc_recall = recall.compute()
        # tensorboard --logdir=runs --port=8000
        # tensorboard --logdir=MobileNetv2_logs --port=8000 --host 192.168.0.109
        writer.add_scalar('Loss/Train', total_loss/total_batch, epoch)  #batch당 loss
        writer.add_scalar('Accuracy/Train', total_acc/total_train*100, epoch) 
        #writer.add_scalar('Precision/Train', acc_precision, epoch)
        #writer.add_scalar('Recall/Train', acc_recall, epoch)

        #precision.reset()
        #recall.reset()

        #val_acc_precision = MulticlassPrecision(num_classes=num_classes).to(device)
        #val_acc_recall = MulticlassRecall(num_classes=num_classes).to(device)
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
                    #val_precision = val_acc_precision(predicted_outputs, labels)
                    #vall_recall = val_acc_recall(predicted_outputs, labels)
                
        val_loss_value = running_vall_loss/len(val_data_loader)
        accuracy = (100 * running_accuracy / total) # divided by the total num of predictions done
        #val_precision = val_acc_precision.compute()
        #vall_recall = val_acc_recall.compute()
        writer.add_scalar('Loss/Validation', val_loss_value, epoch)
        writer.add_scalar('Accuracy/Validation', accuracy, epoch)
        #writer.add_scalar('Precision/Validation', val_precision, epoch)
        #writer.add_scalar('Recall/Validation', vall_recall, epoch)
        #val_acc_precision.reset()
        #val_acc_recall.reset()
    #torch.save(model, 'checkpoint/mobileNetv2/model_state_dict_%d.pt'%(epoch+1))
    torch.save({
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'model_state_dict': model.state_dict(), 
        'loss': total_loss}, 'mobileNetv3/checkpoint/model_state_dict_%d.tar'%(epoch+1))

# 모델 저장
#torch.save(model.state_dict(), 'checkpoint/model_state_dict.pt')
# torch.save(model.state_dict(), 'model_state_dict.pth')
# torch.save(model, 'model_state_dict.pth')

# model = MobileNet_v2()
#model.load_state_dict('model_state_dict_48.pth')
# model.eval()
# model = torch.load('model.pth')

# 저장된 모델로 test 돌리기
# model = torch.load('checkpoint/model_state_dict_.pt')
# model.eval()
# checkpoint = torch.load('checkpoint/model_state_dict_.pt')
# model = MobileNet_v2()
# model.load_state_dict(checkpoint['model'])


"""
class_list = ['100', '44',	'45',	'46',   '65',	
              '66',	'67',	'68',	'69',	'70',	'71',	'72',	'73',	'74',	
              '75',	'76',	'77',	'78',	'79',	'80',	'81',	
              '82',	'83',	'84',	'85',	'86',	'87',	'88',	'89',	'90']

for i in class_list:
    path = f'../Dataset/OCR_HolderName/test/'+class_list
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
            data, target = data.cuda(), target.cuda()
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
"""



writer.close()
