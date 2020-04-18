import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

from sklearn.datasets import load_iris

import time

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']

data = np.array(df.iloc[:100, [0, 1, -1]])
X, y = data[:, :-1], data[:, -1]
# X, y = data[30:70, :-1], data[30:70, -1]
y = np.array([1 if i == 1 else -1 for i in y])


class Model:
    def __init__(self):
        self.a = []  # 每个x的权值
        self.b = 0
        self.l_rate = 1

    def sign(self, mat, i, y_train, a, b):
        fx = np.dot(a * y_train, mat[i, :]) + b
        return fx

    def fit(self, X_train, y_train):
        # 初始化矩阵
        mat = np.dot(X_train, X_train.T)
        self.a = np.ones(len(X_train), dtype=np.float32)

        is_wrong = False
        while not is_wrong:
            wrong_count = 0
            wrong_point = []

            for i in range(len(X_train)):
                if y_train[i] * self.sign(mat, i, y_train, self.a, self.b) <= 0:
                    wrong_point.append(i)
                    wrong_count += 1

            if wrong_count == 0:
                break

            point = random.choice(wrong_point)
            self.a[point] = self.a[point] + self.l_rate
            self.b = self.b + self.l_rate * y_train[point]

        return 'Perceptron Model!'


start = time.time()
perceptron = Model()
perceptron.fit(X, y)
end = time.time()
print('time:%d' % (end - start))

# x_points,y_都是数组
x_points = np.linspace(4, 7, 10)

w = np.zeros(len(data[0]) - 1, dtype=np.float32)
for i in range(len(X)):
    w += perceptron.a[i] * y[i] * X[i]

y_ = -(w[0] * x_points + perceptron.b) / w[1]

plt.plot(x_points, y_)

plt.plot(data[:50, 0], data[:50, 1], 'rx', label='0')
plt.plot(data[50:, 0], data[50:, 1], 'bo', label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()
