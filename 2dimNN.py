import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

#dataset 생성
N = 100     # 점 개수/클래스
D = 2       #dimension
K = 3       #class 개수
X = np.zeros((N*K, D)) #X.shape: (300, 2)
y = np.zeros(N*K, dtype='uint8')    #라벨_전체 point에 대한 라벨                                                                                                                                                                             
for j in range(K):
    # (0~100)
    # (100~200)
    # (200~300)
    idx = range(N*j, N*(j+1)) 
    
    #linespace(start, end, #data)
    r = np.linspace(0.0, 1, N)                                  #반지름
    t = np.linspace(j*4, (j+1)*4, N)+np.random.randn(N)*0.2     #theta
    
    #좌표 생성
    X[idx] = np.c_[r*np.sin(t), r*np.cos(t)]                     #np.c_[] => (1) (2) => [[1],[2]]로 바꿔줌
    y[idx] = j

# #parameter 초기화 -> 랜덤으로
# W = 0.01 * np.random.randn(D, K) # (dim, 클래스 개수) = (2,3)
# b = np.zeros((1,K))              # (1, 클래스 개수) = (1,3)
# num_examples = X.shape[0]  

# #hyperparameter
# step_size = 1e-0    #learning rate
# reg = 1e-3  #람다 값

# #gradient descent
# for i in range(200):
    
#     #Class Score 계산
#     scores = np.dot(X, W) + b
    
#     #Class 확률 계산 (Softmax Classifier)
#     exp_scores = np.exp(scores)      # exp로 보내기
#     probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  #가로방향으로 sum 진행
    
#     #loss 계산: average cross-entropy loss(Softmax) & regularization
#     #full loss = -log(probability)
#     to_logprobs = -np.log(probs[range(num_examples), y])
#     data_loss = np.sum(to_logprobs)/num_examples
#     reg_loss = 0.5 * reg * np.sum(W*W)  #backprop에서 reg gradient 편하게 구하기 위해 0.5곱함
#     loss = data_loss + reg_loss
    
#     if i % 10 == 0:
#         print("iteration %d: loss %f" %(i, loss))
    
#     #gradient 계산
#     #dscores = output
#     dscores = probs
#     dscores[range(num_examples), y] -= 1
#     dscores /= num_examples
    
#     #backprop
#     dW = np.dot(X.T, dscores)  # local gradient*upstream gradient(dscores)
#     db = np.sum(dscores, axis=0, keepdims=True) #세로방향 sum -> 같은 class끼리 sum
#     dW += reg*W    #regularization gradient 
#     #(위에서 loss = data_loss + reg_loss 구했으니까 reg_loss에 대한 gradient도 더해줘야 함)

#     #parameter 업데이트
#     W += -step_size * dW    #(gradient의 반대방향으로 향하게_loss 줄이기 위해)
#     b += -step_size * db

# #training set 정확도
# scores = np.dot(X, W) + b
# predicted_class = np.argmax(scores, axis=1) #가로방향, index 반환 => predicted_class.shape = 300
# print("training accuracy: %.2f" %(np.mean(predicted_class==y)))

# # meshgrid 범위 나누기
# meshgrid_range = .02
# x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1 #x축 최대+1, 최소+1 값 
# y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1 #y축 최대+1, 최소+1 값
# xx, yy = np.meshgrid(np.arange(x_min, x_max, meshgrid_range), np.arange(y_min, y_max, meshgrid_range)) #배경_새로 잡은 좌표값 (0.02간격으로) 나누기
# # print(xx.shape)
# # print(yy.shape)

# # meshgrid data에 대해 학습
# Z = np.dot(np.c_[xx.ravel(), yy.ravel()], W) + b  #새로 잡은 좌표값 예측1
# #ravel(): 다차원 -> 1차원, 복사x 따라서 원본도 바뀜
# #faltten(): 다차원 -> 1차원, 복사o 따라서 원본 그대로
# Z = np.argmax(Z, axis=1)    #새로 잡은 좌표값 예측2 (최대 값의 index값 Z에 저장)
# print(Z.shape)
# Z = Z.reshape(xx.shape)     #meshgrid 범위로 맞춰줘야 함 (e.g., Z.shape: 187x194) 

# #draw graph
# fig = plt.figure()
# plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.6)    #배경 칠하기
# plt.scatter(X[:, 0], X[:, 1], c=y, s=40,  edgecolor='k', cmap=plt.cm.Spectral) #s: 마커크기
# plt.show()


#=====================================Training 2 Layer Neural Network=========================================================#
# N = 100     #점 개수/클래스
# D = 2       #dimension
# K = 3       #class 개수
# 총 점 개수: N x K = 300개
h = 100 # hidden layer
W = 0.01 * np.random.randn(D, h) #randn(2, 100) -> (2,100)mat 정규분포
b = np.zeros((1,h))              #(1,100)mat
W2 = 0.01*np.random.randn(h, K)  #(100,3)mat
b2 = np.zeros((1,K))             #(1,3)mat

# hyper parameter
step_size = 1e-0
reg = 1e-3

num_examples = X.shape[0]

for i in range(10000):
    #forward pass
    hidden_layer = np.maximum(0, np.dot(X, W)+b)    #activation function: ReLu
    scores = np.dot(hidden_layer, W2) + b2          #class score
    
    #class 확률 
    exp_scores = np.exp(scores)
    probs = exp_scores/np.sum(exp_scores, axis=1, keepdims=True)
    
    #loss 계산
    to_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(to_logprobs)/num_examples
    reg_loss = 0.5 * reg * np.sum(W*W) + 0.5*reg*np.sum(W2*W2)
    loss = data_loss + reg_loss
    if i % 1000 == 0:
        print("iteration %d: loss %f" %(i, loss))
    
    #gradient 계산
    dscores = probs
    dscores[range(num_examples), y] -= 1
    dscores /= num_examples
    
    #backpropagation 계산
    #첫번째 backpropagation
    dW2 = np.dot(hidden_layer.T, dscores)
    db2 = np.sum(dscores, axis=0, keepdims=True) #axis=0: 세로방향
    #두번째 backpropagation 
    dhidden = np.dot(dscores, W2.T)
    #ReLu backpropagation
    dhidden[hidden_layer <= 0] = 0
    dW = np.dot(X.T, dhidden)
    db = np.sum(dhidden, axis=0, keepdims=True)
    
    #regularization
    dW2 += reg*W2
    dW += reg*W
    
    #parameter update
    W += -step_size*dW
    b += -step_size*db
    W2 += -step_size*dW2
    b2 += -step_size*db2

#Training 정확도
hidden_layer = np.maximum(0, np.dot(X, W) + b)
scores = np.dot(hidden_layer, W2) + b2
predicted_class = np.argmax(scores, axis=1)
print ('training accuracy: %.2f' % (np.mean(predicted_class == y)))

# meshgrid 범위 나누기
h = .02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# print(xx.shape)
# print(yy.shape)

# meshgrid data에 대해 학습
Z = np.dot(np.maximum(0, np.dot(np.c_[xx.ravel(), yy.ravel()], W) + b), W2) + b2
#ravel(): 다차원 -> 1차원, 복사x 따라서 원본도 바뀜
#faltten(): 다차원 -> 1차원, 복사o 따라서 원본 그대로
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)     #meshgrid 범위로 맞춰줘야 함

#draw graph
fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40,  edgecolor='k', cmap=plt.cm.Spectral) #s: 마커크기
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
#plt.show()


