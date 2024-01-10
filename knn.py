import numpy as np
import matplotlib.pyplot as plt
import math
from collections import Counter
import matplotlib.colors as mcolors

# create dataset randomly
tmp1Datalist = np.random.randint(1, 21, (15, 2))
tmp2Datalist = np.random.randint(0, 3, (15, 1))
datalist = np.concatenate([tmp1Datalist, tmp2Datalist], 1)
    
# datalist = [[10, 20, 0],
#            [9, 6, 0],
#            [1, 5, 1],
#            [3, 4, 0],
#            [2, 16, 1],
#            [7, 10, 0],
#            [9, 2, 1],
#            [16, 15, 0],
#            [5, 7, 1],
#            [11, 17, 1],
#            [6, 15, 0],
#            [2, 6, 1],
#            [3, 5, 1],
#            [14, 20, 2],
#            [2, 7, 2],
#            [4, 9, 2],
#            [3, 1, 2]]
print(datalist)
print("Hi")

testData = [[10, 3, -1]]
_testData = np.array(testData)
print(_testData)
allDataset = np.append(datalist, _testData, axis=0)

# x, y 좌표와 라벨 분리
x = [item[0] for item in datalist]
y = [item[1] for item in datalist]
labels = [item[2] for item in datalist]

#Test Data 추가된 리스트
_x =  [item[0] for item in allDataset]
_y = [item[1] for item in allDataset]
_labels = [item[2] for item in allDataset]

coord = []
for i in range(datalist.shape[0]):
    coord.append([datalist[0], datalist[1]])

# 라벨에 따른 색상 지정
colors = ['red' if label == 0 else 'limegreen' if label == 1 else 'blue' if label == 2 else 'orange' for label in _labels]

# 산점도 그리기
plt.scatter(_x, _y, color=colors)

# 추가 설정
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('KNN')



# KNN algorithm 
class myKNN:
    def __init__(self, k):
        self.k = k
    
    def train(self, X, y):
        self.Xtrain = X
        self.ytrain = y
    
    def predict(self, X, test): # k neighbors
        distances = [math.sqrt(math.pow(xVal-test[0][0],2) + math.pow(yVal-test[0][1],2)) for xVal, yVal in zip(x, y)]
        npArr = np.array(distances)
        print(npArr)
        sortedNpArrIndex = np.argsort(npArr)
        print(sortedNpArrIndex)
        
        for i in range(self.k):
            plt.text(x[sortedNpArrIndex[i]], y[sortedNpArrIndex[i]], str(i+1)+ " " + "NN")
        
        sortedLabels = [labels[idx] for idx in sortedNpArrIndex]
        print("===sorted Label===")
        print(sortedLabels)
        slicedLabels = sortedLabels[:self.k]
        print(slicedLabels)
        
        # 최빈값
        count_items = Counter(slicedLabels)
        print(count_items)
        max_item = count_items.most_common(n=1)
        # print(max_item[0][0])
        print(max_item)
        return max_item[0][0]

#Real Test

k = myKNN(3)
k.train(coord, y)
myNeighbors = k.predict(coord, testData)
print('----------Predicted Label of Test Data----------')
print(myNeighbors)
_myNeighbors = myNeighbors.tolist()

plt.text(testData[0][0], testData[0][1], 'Test Data - Predicted Label: ' + str(_myNeighbors))
plt.show()



