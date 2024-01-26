import torch
import torch.nn as nn
import torchvision 
from torchvision import transforms, datasets
import torch.nn.functional as F
from torch.utils.data import random_split

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import random

""".item"""
# 만약 tensor에 하나의 값만 존재한다면, 
# .item() 을 사용하여 python scalar 값 뽑아옴. 
# tensor에 하나의 값이 아니라 여러개가 존재한다면 사용 불가능

"""torch.argmax(input, dimension)"""
# dim X: tensor(max 값 index) 출력
# dim = 0: 열을 기준으로 max 값이 있는 index를 출력
# dim = 1: 행을 기준으로 max 값이 있는 index를 출력

"""torch.max(input, dimension)"""
# dim X: 최댓값과 인덱스 모두 출력
# dim = 0: 열을 기준(각 열마다)으로 최댓값과 인덱스를 출력
# dim = 1: 행을 기준(각 행마다)으로 최댓값과 인덱스를 동시에 출력


"""tensor.size()관련 메모"""
#b= torch.tensor([[1,2,3], [2,3,4]])
#b.size(dim=0) -> 2
#b.size(dim=1) -> 3

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

# train set -> train(50000) + val(10000)으로 쪼개기
train_data, val_data = torch.utils.data.random_split(train_data, [50000, 10000])
# print(len(train_data))
# print(len(val_data))


# Data loader: 데이터 불러오기
# 학습에 사용될 전체 데이터(train_data) 가지고 있다가 iteration 개념으로 데이터 넘겨줌
batch_size = 32
data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

# data/label 나누기
#x_train, y_train = train_data.data, train_data.targets
#x_test, y_test = test_data.data, test_data.targets

# CNN
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        # Conv1d: 자연어 처리 - 1차원 convolution으로 진행
        # Conv2d: 이미지 처리
        # Conv3d: CT/비디오 처리
        # Layer1
        # padding='same': input/output 맞춰줌
        # padding='valid': padding=0과 같음
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
loss_fn = torch.nn.CrossEntropyLoss().to(device)   #Softmax 함수 & negative log liklihood까지 포함되어 있음
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#Tensorflow에서 model.fit() 부분에 해당
#Training 
total_batch = len(data_loader)
total_train = len(train_data)
print('총 배치의 수 : {}'.format(total_batch)) # 50000 / 32 = 1563
for epoch in range(10): #30에폭-test정확도: 99%, 40에폭-test정확도: 99%(validation best), 50 & 70에폭-worst 
    total = 0.0
    running_accuracy = 0.0
    running_vall_loss = 0.0 
    total_loss, total_acc = 0.0, 0.0
    cnt = 0
    with tqdm(data_loader, unit="batch") as tepoch: #progress bar, batch=32 -> 1563번 = 1epoch
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
        writer.add_scalar('Loss/Train', total_loss/total_batch, epoch) # -> tot_loss/1563 (왜냐면 for문이 1563번 도니까)
        writer.add_scalar('Accuracy/Train', total_acc/total_train*100, epoch) # -> tot_acc/50000 (total_acc는 input 전체 값 더해지니까 /50000)
        
        # validation check
        with torch.no_grad():
            model.eval()
            for data in val_data_loader: 
                inputs, labels = data
                predicted_outputs = model(inputs) 
                val_loss = loss_fn(predicted_outputs, labels) 
             
               # The label with the highest value will be our prediction
               # _: 필요없는 값 저장 용도
                _, predicted = torch.max(predicted_outputs, dim=1)
                running_vall_loss += val_loss.item()
                total += labels.size(0)
                running_accuracy += (predicted == labels).sum().item()
        val_loss_value = running_vall_loss/len(val_data_loader)
        accuracy = (100 * running_accuracy / total)
        writer.add_scalar('Loss/Validation', val_loss_value, epoch)
        writer.add_scalar('Accuracy/Validation', accuracy, epoch)

#Tensorflow에서 model.evaluate()에 해당
# Test
size = len(test_data_loader.dataset) #size = 10000
num_batches = len(test_data_loader)
#모델 평가 모드 - dropout, normalization 제외시키는 역할 - model.eval()
model.eval()
total_test_loss = 0.0
loop = 0
total_test_accuracy = 0.0
with torch.no_grad():
    for data, target in test_data_loader:
        loop += 1
        #data, target = data.to(device), target.to(device)
        pred = model(data)
        test_loss = loss_fn(pred, target).item()
        
        total_test_loss +=  test_loss
        total_test_accuracy += (pred.argmax(1)==target).type(torch.float).sum().item()

total_test_loss /= size
total_test_accuracy /= size
print(f"Test Error: \nAccuracy: {(100*total_test_accuracy):>0.1f}%, Avg loss: {total_test_loss:>8f} \n")

#tensorboard write 중지
writer.close()

"""
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
    #test set에서 인덱스 하나 랜덤으로 뽑기
    data_idx = np.random.randint(len(test_data))
    #If you have a single sample, just use input.unsqueeze(0) to add a fake batch dimension.
    
    #test_data에 data_idx의 0번 = X값 (input), 1번 = y값 (label)
    input_img = test_data[data_idx][0].unsqueeze(dim=0).to(device)
    output = model(input_img)
    
    argmax = torch.argmax(output, dim=1) #torch.max는 max값, index return, dim=1: 행으로 (행=열 방향) 
    pred = label_tags[argmax.item()]    #예측 라벨
    label = label_tags[test_data[data_idx][1]] #정답 라벨
    
    fig.add_subplot(rows, columns, i)
    if pred == label:
        plt.title(pred + ', right')
        cmap = 'Blues'
    else:
        plt.title('Not ' + pred + ' but ' +  label)
        cmap = 'Reds'
    plot_img = test_data[data_idx][0][0,:,:]
    plt.imshow(plot_img, cmap=cmap)
    plt.axis('off')
    
plt.show() 
"""

