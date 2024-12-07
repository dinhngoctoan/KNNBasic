import numpy as np
import math
import pandas as pd

def calculate_distance(x1,x2):
    distance = 0.0
    for i in range(len(x1)):
        distance += (x1[i]-x2[i])**2
    return math.sqrt(distance)
class KNNModel:
    #Initialize model
    def __init__(self,k):
        self.X_train = None
        self.y_train = None
        self.k = k
    #Save training data

    def fit(self,X_train,y_train):
        self.X_train = X_train
        self.y_train = y_train
    def getLabel(self,k,data):
        label = []
        for i in range(len(self.X_train)):
            label.append({
                "label":self.y_train[i],
                "distance":calculate_distance(data,self.X_train[i])
            })
        label.sort(key = lambda x:x["distance"])
        return label[:k]
    #Use major vote to decide labels
    def major_vote(self,data):
        numB = 0
        numM = 0
        label = self.getLabel(self.k,data)
        #Đếm số lượng mỗi nhãn trong k điểm gần nhất
        for element in label:
            if element['label'] == 'B':
                numB = numB + 1
            else:
                numM = numM + 1
#so sánh số lượng mỗi nhãn để chọn ra nhãn gán cho điểm
        if numB > numM :
            return 'B'
        else:
            return 'M'
    #Predict
    def predict(self,X_test):
        if len(X_test.shape) == 1:
            return self.major_vote(X_test)
        else:
            y_predict = []
            for data in X_test:
                y_predict.append(self.major_vote(data))
            return y_predict
