import numpy as np
import pandas as pd

import math

datasets = np.array([[1, 'S', -1], [1, 'M', -1], [1, 'M', 1], [1, 'S', 1],
                     [1, 'S', -1], [2, 'S', -1], [2, 'M', -1], [2, 'M', 1],
                     [2, 'L', 1], [2, 'L', 1], [3, 'L', 1], [3, 'M', 1],
                     [3, 'M', 1], [3, 'L', 1], [3, 'L', -1]])

X, y = datasets[:, :-1], datasets[:, -1]


class Bayes:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def predict(self, input_data):
        label = list(set(self.y))
        probabilities = {}

        y_count = {}
        xy_count = {}
        py = {}
        pxy = {}
        feature = {}

        for k in range(len(label)):
            y_count[k] = 0
            py[k] = 1
            probabilities[k] = 1

        for k in range(len(self.X[0])):
            xy_count[k] = 0
            pxy[k] = 1
            feature[k] = 0  # 每个特征的个数

        for k in range(len(self.X[0])):
            feature[k] = len(list(set(self.X.T[k])))

        for i in range(len(label)):
            for k in range(len(self.X[0])):
                xy_count[k] = 0
            for k, v in zip(self.X, self.y):
                if v == '-1':
                    v = 0
                else:
                    v = 1

                if v == i:
                    y_count[i] += 1

                for sj in range(len(self.X[0])):
                    if i == v and k[sj] == input_data[sj]:
                        xy_count[sj] += 1

            # 计算P(Y)
            py[i] = (y_count[i] + 1) / (len(self.X) + len(label))
            probabilities[i] = py[i]

            # 计算P(X|Y),P(X,Y)
            for k in range(len(self.X[0])):
                pxy[k] = (xy_count[k] + 1) / (y_count[i] + feature[k])
                probabilities[i] *= pxy[k]

        label = sorted(probabilities.items(), key=lambda x: x[1])[-1][0]
        return label

    def score(self, X_test, y_test):
        right = 0
        for X, y in zip(X_test, y_test):
            label = self.predict(X)
            if y == label:
                right += 1
        return right / len(X_test)


model = Bayes(X, y)
test = model.predict(np.array([2, 'S']))

if test == 1:
    print('positive!')
else:
    print('negative!')
