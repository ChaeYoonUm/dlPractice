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
import torch.onnx
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

#========MobileNet V3========#

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
num_classes = 30

from functools import partial
import torch

from torch import nn, Tensor

from torchvision.models.mobilenetv2 import _make_divisible
from torchvision.ops.misc import ConvNormActivation
from typing import Any, Callable, Dict, List, Optional, Sequence
from torch.hub import load_state_dict_from_url
from torch.utils.model_zoo import load_url as load_state_dict_from_url

model_urls = {
    "mobilenet_v3_large": "https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth",
    "mobilenet_v3_small": "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth",
}
class SqueezeExcitation(nn.Module):
    def __init__(self, input_channels: int, squeeze_factor: int = 4):
        super().__init__()
        squeeze_channels = _make_divisible(input_channels // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)

    def _scale(self, input: Tensor, inplace: bool) -> Tensor:
        scale = F.adaptive_avg_pool2d(input, 1)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        return F.hardsigmoid(scale, inplace=inplace)

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input, True)
        return scale * input

class InvertedResidualConfig:

    def __init__(self, input_channels: int, kernel: int, expanded_channels: int, out_channels: int, use_se: bool,
                 activation: str, stride: int, dilation: int, width_mult: float):
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride
        self.dilation = dilation

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return _make_divisible(channels * width_mult, 8)

class InvertedResidual(nn.Module):

    def __init__(self, cnf: InvertedResidualConfig, norm_layer: Callable[..., nn.Module],
                 se_layer: Callable[..., nn.Module] = SqueezeExcitation):
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError('illegal stride value')

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        # expand
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(ConvNormActivation(cnf.input_channels, cnf.expanded_channels, kernel_size=1,
                                           norm_layer=norm_layer, activation_layer=activation_layer))

        # depthwise
        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(ConvNormActivation(cnf.expanded_channels, cnf.expanded_channels, kernel_size=cnf.kernel,
                                       stride=stride, dilation=cnf.dilation, groups=cnf.expanded_channels,
                                       norm_layer=norm_layer, activation_layer=activation_layer))
        if cnf.use_se:
            layers.append(se_layer(cnf.expanded_channels))

        # project
        layers.append(ConvNormActivation(cnf.expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer,
                                       activation_layer=nn.Identity))

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result += input
        return result

class MobileNetV3(nn.Module):

    def __init__(
            self,
            inverted_residual_setting: List[InvertedResidualConfig],
            last_channel: int,
            num_classes: int = 1000,
            block: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        """
        MobileNet V3 main class

        Args:
            inverted_residual_setting (List[InvertedResidualConfig]): Network structure
            last_channel (int): The number of channels on the penultimate layer
            num_classes (int): Number of classes
            block (Optional[Callable[..., nn.Module]]): Module specifying inverted residual building block for mobilenet
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
        """
        super().__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (isinstance(inverted_residual_setting, Sequence) and
                  all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(ConvNormActivation(3, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer,
                                       activation_layer=nn.Hardswish))

        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(ConvNormActivation(lastconv_input_channels, lastconv_output_channels, kernel_size=1,
                                       norm_layer=norm_layer, activation_layer=nn.Hardswish))

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(lastconv_output_channels, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(last_channel, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _mobilenet_v3_conf(arch: str, params: Dict[str, Any]):
    # non-public config parameters
    reduce_divider = 2 if params.pop('_reduced_tail', False) else 1
    dilation = 2 if params.pop('_dilated', False) else 1
    width_mult = params.pop('_width_mult', 1.0)

    bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_mult=width_mult)

    if arch == "mobilenet_v3_large":
        # def __init__(self, input_channels: int, kernel: int, expanded_channels: int, out_channels: int, use_se: bool,
        #          activation: str, stride: int, dilation: int, width_mult: float):
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, False, "RE", 1, 1),
            bneck_conf(16, 3, 64, 24, False, "RE", 2, 1),  # C1
            bneck_conf(24, 3, 72, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 72, 40, True, "RE", 2, 1),  # C2
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 3, 240, 80, False, "HS", 2, 1),  # C3
            bneck_conf(80, 3, 200, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 480, 112, True, "HS", 1, 1),
            bneck_conf(112, 3, 672, 112, True, "HS", 1, 1),
            bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2, dilation),  # C4
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
        ]
        last_channel = adjust_channels(1280 // reduce_divider)  # C5
    elif arch == "mobilenet_v3_small":
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, True, "RE", 2, 1),  # C1
            bneck_conf(16, 3, 72, 24, False, "RE", 2, 1),  # C2
            bneck_conf(24, 3, 88, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 96, 40, True, "HS", 2, 1),  # C3
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 120, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 144, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2, dilation),  # C4
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
        ]
        last_channel = adjust_channels(1024 // reduce_divider)  # C5
    else:
        raise ValueError("Unsupported model type {}".format(arch))

    return inverted_residual_setting, last_channel

def _mobilenet_v3_model(
    arch: str,
    inverted_residual_setting: List[InvertedResidualConfig],
    last_channel: int,
    pretrained: bool,
    progress: bool,
    **kwargs: Any
):
    model = MobileNetV3(inverted_residual_setting, last_channel, **kwargs)
    if pretrained:
        if model_urls.get(arch, None) is None:
            raise ValueError("No checkpoint is available for model type {}".format(arch))
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model

def mobilenet_v3_large(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MobileNetV3:
    """
    Constructs a large MobileNetV3 architecture from
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    arch = "mobilenet_v3_large"
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(arch, kwargs)
    return _mobilenet_v3_model(arch, inverted_residual_setting, last_channel, pretrained, progress, **kwargs)


def mobilenet_v3_small(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MobileNetV3:
    """
    Constructs a small MobileNetV3 architecture from
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    arch = "mobilenet_v3_small"
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(arch, kwargs)
    return _mobilenet_v3_model(arch, inverted_residual_setting, last_channel, pretrained, progress, **kwargs)
from torchinfo import summary
# 모델 확인 
from torchvision import models
model = mobilenet_v3_small()
summary(model, (1,3,224,224), device="cpu")
dummy_data = torch.empty(1, 3, 224, 224, dtype = torch.float32)
# mobilenet = torchvision.models.mobilenet_v3_small()
# torch.onnx.export(mobilenet, dummy_data, "torch_mobileNetV3.onnx", verbose=True)
# torch.onnx.export(model, dummy_data, "custom_mobilenetV3.onnx", verbose=True)
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
