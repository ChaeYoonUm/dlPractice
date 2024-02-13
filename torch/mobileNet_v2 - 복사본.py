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

png_file_path = sorted(glob.glob('C:/Users/QR22002/Desktop/chaeyun/dataset/train/72/**/*.png', recursive=True)) #116528

transform = transforms.Compose([          
    transforms.ToTensor()])

#data augmentation
aug = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.
    transforms.RandomAffine(degrees=0, shear=0.5),
    transforms.RandomHorizontalFlip(0.5)
])

def imgAugmentation(img):
    image = transform(img)
    cropped_img = torchvision.transforms.functional.crop(image, 2, 2, 220, 170)
    resize = torchvision.transforms.Resize((216, 216))
    enlarged_img = resize(cropped_img)
    padding = torchvision.transforms.functional.pad(enlarged_img, (4,4,4,4), fill=0)
    augment = aug(padding)
    toImg = transforms.ToPILImage()
    outImg = toImg(augment)
    return outImg


print(f'png file: {len(png_file_path)}')

length = len(png_file_path)
cnt = 0
for idx in range(length):
    tmp_png_name = os.path.splitext(png_file_path[idx])
    image = cv2.imread(png_file_path[idx])
    image = imgAugmentation(image)
        
    tmpPath = 'C:/Users/QR22002/Desktop/chaeyun/dataset/augment/'
    image.save(tmpPath + f'{os.path.basename(tmp_png_name[0])}.png')
    cnt += 1
    print(f'done: {cnt} {os.path.basename(tmp_png_name[0])}')

    

