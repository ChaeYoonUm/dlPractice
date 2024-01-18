import numpy as np
import matplotlib.pyplot as plt

N = 100     # 점 개수/클래스
D = 2       #dimension
K = 3       #class 개수
X = np.zeros((N*K, D)) #X.shape: (300, 2)
y = np.zeros(N*K, dtype='uint8')    #라벨
for j in range(K):
    ix = range(N*j, N*(j+1))
    r = np.linspace(0.0, 1, N) #반지름
    t = np.linspace(j*4, (j+1)*4, N)+np.random.randn(N)*0.2 #theta
    X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j
#draw graph
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.show()




