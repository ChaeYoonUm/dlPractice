import torch
import torch.nn as nn
import torchvision 
from torchvision import transforms, datasets
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import random

device = (
    "cuda"
    if torch.cuda.is_available()
    else "gpu"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# MNIST 데이터 다운
# train = True: training data 
# train = False: Test data
train_data = torchvision.datasets.MNIST('./data/MNIST', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])) # transforms.ToTensor() -> 0~1 값
test_data = torchvision.datasets.MNIST('./data/MNIST', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()])) # transforms.ToTensor() -> 0~1 값

# Data loader: 데이터 불러오기
# 학습에 사용될 전체 데이터(train_data) 가지고 있다가 iteration 개념으로 데이터 넘겨줌
data_loader = torch.utils.data.DataLoader(train_data, batch_size=100)
test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=100)

# data/label 나누기
x_train, y_train = train_data.data, train_data.targets
x_test, y_test = test_data.data, test_data.targets

print(len(test_data))

# CNN
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        # Layer1
        self.layer1 = torch.nn.Sequential(
            nn.Conv2d(kernel_size=3, in_channels=1, out_channels=16, padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2, stride=2))
        
        # Layer2
        self.layer2 = torch.nn.Sequential(
            nn.Conv2d(kernel_size=3, in_channels=16, out_channels=32, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.4),
            nn.MaxPool2d(2, stride=2))
        
        # Fully Connected Layer
        self.fc_layer = torch.nn.Sequential(
            nn.Linear(7*7*32, 128),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.Linear(128, 10))
    
    # Training 진행 순서
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = nn.Flatten()(out)
        out = self.fc_layer(out)
        return out

#Model 불러오기
model = ConvNet().to(device)

#Tensorflow에서 model.compile() 부분에 해당
loss_fn = torch.nn.CrossEntropyLoss().to(device)   #Softmax 함수 포함되어 있음
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#Tensorflow에서 model.fit() 부분에 해당
#Training
total_batch = len(data_loader)
print('총 배치의 수 : {}'.format(total_batch))
for epoch in range(40):
    with tqdm(data_loader, unit="batch") as tepoch:
        for X, Y in tepoch:
            tepoch.set_description(f"Epoch {epoch+1}")
            
            prediction = model(X)
            loss = loss_fn(prediction, Y)
            accuracy = 1 - loss
            
            #Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # tensorboard --logdir=runs --port=8000
            writer.add_scalar('Loss/Train', loss, epoch)
            writer.add_scalar('Accuracy/Train', accuracy, epoch)
            tepoch.set_postfix(loss=loss.item())


#Tensorflow에서 model.evaluate()에 해당
# Test
size = len(test_data_loader.dataset)
num_batches = len(test_data_loader)
#모델 평가 모드 - dropout, normalization 제외시키는 역할 - model.eval()
model.eval()
test_loss, correct = 0, 0
with torch.no_grad():
    for X, y in test_data_loader:
        X, y = X.to(device), y.to(device)
        pred = model(X) #prediction
        test_loss += loss_fn(pred, y).item() 
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
test_loss /= len(test_data_loader.dataset)
correct /= size
print(f"Test Error: \nAccuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
writer.close()


# inference
test_batch_size=1000
columns = 6
rows = 6
fig = plt.figure(figsize=(10,10))
label_tags = {
    0: '0', 
    1: '1', 
    2: '2', 
    3: '3', 
    4: '4', 
    5: '5', 
    6: '6',
    7: '7', 
    8: '8', 
    9: '9' }
for i in range(1, columns*rows+1):
    data_idx = np.random.randint(len(test_data))
    #squeeze: 1인 값 지워줌 #[3, 1, 20, 128] -> [3, 20, 128]
    #unsqueeze: 1 추가(때문에 어디에 넣을지 지정해야 함) 
    #unsqueeze(1) # [3, 100, 100] -> [3, 1, 100, 100]
    input_img = test_data[data_idx][0].unsqueeze(dim=0).to(device)
    output = model(input_img)
    
    # _,: ignores the unneeded Tensor above.
    _, argmax = torch.max(output, 1)
    pred = label_tags[argmax.item()]
    label = label_tags[test_data[data_idx][1]]
    
    fig.add_subplot(rows, columns, i)
    if pred == label:
        plt.title(pred + ', right !!')
        cmap = 'Blues'
    else:
        plt.title('Not ' + pred + ' but ' +  label)
        cmap = 'Reds'
    plot_img = test_data[data_idx][0][0,:,:]
    plt.imshow(plot_img, cmap=cmap)
    plt.axis('off')
    
plt.show() 

"""
#MNIST에서 무작위 하나로 테스트 해보기
r = random.randint(0, len(test_data)-1)
X_single_data = test_data.test_data[r:r+1].view(-1, 28*28).float().to(device)
Y_single_data = test_data.test_labels[r:r+1].to(device)

print('Label: ', Y_single_data.item())
single_prediction = model(X_single_data)
print('Prediction: ', torch.argmax(single_prediction, 1).item())

plt.imshow(test_data.test_data[r:r+1].view(28,28), cmap='Greys', interpolation='nearest')
plt.show()
"""
