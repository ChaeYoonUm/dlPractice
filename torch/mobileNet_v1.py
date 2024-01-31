import torch
import torch.nn as nn
import torchvision 
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        
        self.input = df.iloc() 
        self.out = df.iloc()
    def __len__(self): 
        return len(self.input)
    
    

        

