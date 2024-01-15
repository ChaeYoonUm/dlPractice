import torch 
import torchvision # 파이토치 이미지 관련 라이브러리
import torchvision.transforms as tr # 이미지 전처리기능 라이브러리
from torch.utils.data import DataLoader, Dataset # 데이터를 모델에 사용할 수 있도록 정리해 주는 라이브러리
import numpy as np # 넘파이
import matplotlib.pyplot as plt #맷플롯립


transform = tr.Compose(
    [tr.ToTensor(),
     tr.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



#========================================================================#
# import numpy as np
# import matplotlib as plt

# # feature 값 생성
# np.random.seed(2)
# x1 = np.random.rand(100)
# x2 = np.random.rand(100)
# x3 = np.random.rand(100)

# #  다항식 정의(도출하고자 하는 값)
# y = 0.3*x1 + 0.5*x2 + 0.6*x3 + 0.8

# #임의의 weight 값 생성
# w1 = np.random.uniform(low=-1.0, high=1.0)
# w2 = np.random.uniform(low=-1.0, high=1.0)
# w3 = np.random.uniform(low=-1.0, high=1.0)

# # bias값 생성
# bias = np.random.uniform(low=-1.0, high=1.0)
# print("구하고자하는 다항식: Y=0.3X1+0.5X2+0.6X3+0.8")
# print(f"SGD 시작 다항식: Y={w1}X1{w2}X2{w3}X3{bias}")

# num_epoch=5000
# learning_rate=0.5


# for epoch in range(num_epoch):
    
#     #매 epoch마다 하나의 데이터를 골라옴
#     x1_sgd = np.random.choice(x1)
#     x2_sgd = np.random.choice(x2)
#     x3_sgd = np.random.choice(x3)
#     y_sgd = 0.3*x1_sgd + 0.5*x2_sgd + 0.6*x3_sgd + 0.8
    
#     # 데이타 한건에 대한 예측값
#     predict_sgd = w1*x1_sgd + w2*x2_sgd + w3*x3_sgd + bias
    
#     # 가중치 업데이트
#     w1 = w1 - 2*learning_rate*((predict_sgd - y_sgd)*x1_sgd)
#     w2 = w2 - 2*learning_rate*((predict_sgd - y_sgd)*x2_sgd)
#     w3 = w3 - 2*learning_rate*((predict_sgd - y_sgd)*x3_sgd)
#     bias = bias - 2*learning_rate * (predict_sgd - y_sgd)
    
#     #error값은 전체 데이터셋의 오류값을 계산해야한다.
#     predict = w1*x1 + w2*x2 + w3 + bias
#     error = ((y - predict)**2).mean()
    
#     if epoch%1000 == 0:
#         print("epoch ", epoch,"w1= ", w1, "w2= ", w2, "w3= ", w3,"bias= ", bias, "error= ", error)
        
#     if error < 0.000001:
#         break
# print("최종: ","w1= ", w1 , "w2= ", w2, "w3= ", w3,"bias= ", bias, "error= ", error)

#===================================================================================================#

