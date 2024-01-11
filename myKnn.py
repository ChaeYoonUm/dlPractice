import numpy as np
import matplotlib.pyplot as plt
import math
from collections import Counter
import matplotlib.colors as mcolors
import cv2

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
print("=====Test Data=====")

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

# training data의 (x,y) 좌표 - coord에 저장
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
        sortedNpArrIndex = np.argsort(npArr)
        
        for i in range(self.k):
            plt.text(x[sortedNpArrIndex[i]], y[sortedNpArrIndex[i]], str(i+1)+ " " + "NN")
        
        sortedLabels = [labels[idx] for idx in sortedNpArrIndex]
        slicedLabels = sortedLabels[:self.k]
        
        # 최빈값
        count_items = Counter(slicedLabels)
        max_item = count_items.most_common(n=1)
        # print(max_item[0][0])
        return max_item[0][0]

#Real Test

knn = myKNN(3)
knn.train(coord, labels)
myNeighbors = knn.predict(coord, testData)
print('=====Predicted Label of Test Data=====')
print(myNeighbors)

# _myNeighbors = myNeighbors.tolist()
# plt.text(testData[0][0]-0.5, testData[0][1]-0.5, 'Test Data - Predicted Label: ' + str(_myNeighbors))
# plt.show()


# KNN 전체적인 boudary 그리기 
res_label = np.zeros((21,21,1), np.uint8) # predict한 라벨 값 저장
res = np.zeros((20,20,3), np.uint8)  # (높이)x(행)x(열)
print("=====res=====")
print(res)
for i in range(20) :
    for j in range(20):
        
        testData = [[i,j,-1]]
        label = knn.predict(coord, testData)
        res_label[j,i,0] = label
        print(f"myNeighbors {label}")
        if label == 0:
            res[j,i,0] = 255
        elif label == 1:
            res[j,i,1] = 255
        elif label == 2:
            res[j,i,2] = 255
        else:
            res[j,i,0] = 255
            res[j,i,1] = 255
            res[j,i,2] = 255


# 빨간색 label 맞혔는지 check
Total_num = len(datalist)
num_equal = 0
for idx, data in enumerate(datalist):   
    # enumerate => return tuple
    # idx와 element 각각 따로 뽑고 싶으면 
    # idx, data in enumerate(~)
    
    # datalist => (x,y,lable)
    if res_label[data[1],data[0],0] == data[2]:
        print(f"(x,y)=({data[0]},{data[1]}) label : {data[2]} Equal")
        num_equal = num_equal + 1
    else:
        print(f"(x,y)=({data[0]},{data[1]}) label : {data[2]} Not Equal")

print(f"equal : {num_equal}, notEqual : {Total_num - num_equal}")


# KNN 결과 이미지 저장 및 출력
res = cv2.resize(res, (200, 200), interpolation=cv2.INTER_NEAREST)
cv2.imwrite("knn_result.jpg", res)


# 정확도 확인
"""
res_label = np.zeros((20,20,1), np.uint8)
res = np.zeros((20,20,3), np.uint8)
for i in range(20) :
    for j in range(20):
       
        testData = [[i + 1,j + 1,-1]]
        label = k.predict(coord, testData)
        res_label[j,i,0] = label
        print(f"myNeighbors {myNeighbors}")
        if label == 0:
            res[j,i,0] = 255
        elif label == 1:
            res[j,i,1] = 255
        elif label == 2:
            res[j,i,2] = 255
        else:
            res[j,i,0] = 255
            res[j,i,1] = 255
            res[j,i,2] = 255

Total_num = len(datalist)
num_equal = 0
for idx, data in enumerate(datalist):
    if res_label[data[1],data[0],0] == data[2]:
        print(f"(x,y)=({data[0]},{data[1]}) label : {data[2]} Equal")
        num_equal = num_equal + 1
    else:
        print(f"(x,y)=({data[0]},{data[1]}) label : {data[2]} Not Equal")

print(f"equal : {num_equal}, notEqual : {Total_num - num_equal}")
cv2.imwrite("knn_result.jpg", res)
"""