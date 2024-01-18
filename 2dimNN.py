import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

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
    ix = range(N*j, N*(j+1)) 
    
    #linespace(start, end, #data)
    r = np.linspace(0.0, 1, N)                                  #반지름
    t = np.linspace(j*4, (j+1)*4, N)+np.random.randn(N)*0.2     #theta
    
    X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]                     #np.c_ (열 추가)
    y[ix] = j


#parameter 초기화 -> 랜덤으로
W = 0.01 * np.random.randn(D, K) # (dim, 클래스 개수) = (2,3)
b = np.zeros((1,K))              # (1, 클래스 개수) = (1,3)
num_examples = X.shape[0]  

#hyperparameter
step_size = 1e-0
reg = 1e-3

#gradient descent
for i in range(200):
    
    #Class Score 계산
    scores = np.dot(X, W) + b
    
    #Class 확률 계산 (Softmax Classifier)
    exp_scores = np.exp(scores)      # exp로 보내기
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  #가로방향으로 sum 진행
  
    #loss 계산: average cross-entropy loss(Softmax) & regularization
    #loss = -log(probability)
    correct_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(correct_logprobs)/num_examples
    reg_loss = 0.5 * reg * np.sum(W*W)
    loss = data_loss + reg_loss
    
    if i % 10 == 0:
        print("iteration %d: loss %f" %(i, loss))
    
    #gradient 계산
    dscores = probs
    dscores[range(num_examples), y] -= 1
    dscores /= num_examples
    print(dscores)
    
    #backprop
    dW = np.dot(X.T, dscores)  # local gradient*upstream gradient(dscores)
    db = np.sum(dscores, axis=0, keepdims=True)
    dW += reg*W     #regularization gradient

    #parameter 업데이트
    W += -step_size * dW
    b += -step_size * db

#training set 정확도
scores = np.dot(X, W) + b
predicted_class = np.argmax(scores, axis=1) #가로방향, index 반환
print("training accuracy: %.2f" %(np.mean(predicted_class==y)))


# meshgrid 범위 나누기
h = .02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
print(xx.shape)
print(yy.shape)

# meshgrid data에 대해 학습
Z = np.dot(np.c_[xx.ravel(), yy.ravel()], W) + b  
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









