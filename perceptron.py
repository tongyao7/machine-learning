import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']

data = np.array(df.iloc[:100, [0, 1, -1]])
print(data)
X, y = data[:, :-1], data[:, -1]
y = np.array([1 if i == 1 else -1 for i in y])


class Model:
    def __init__(self):
        self.w = np.ones(len(data[0]) - 1, dtype=np.float32)
        self.b = 0
        self.l_rate = 0.1

    def sign(self, x, w, b):
        y = np.dot(w, x) + b
        return y

    def fit(self, X_train, y_train):
        is_wrong = False
        while not is_wrong:
            wrong_count = 0
            wrong_point = []

            for i in range(len(X_train)):
                if y_train[i] * self.sign(X_train[i], self.w, self.b) <= 0:
                    wrong_point.append(i)
                    wrong_count += 1

            if wrong_count == 0:
                break

            point = random.choice(wrong_point)  # 列表中选择任意下标i
            self.w = self.w + self.l_rate * np.dot(y_train[point], X_train[point])
            self.b = self.b + self.l_rate * y_train[point]

        return 'Perceptron Model!'


perceptron = Model()
perceptron.fit(X, y)

# x_points,y_都是数组
x_points = np.linspace(4, 7, 10)
y_ = -(perceptron.w[0] * x_points + perceptron.b) / perceptron.w[1]
print(perceptron.w[0], perceptron.w[1], perceptron.b)

plt.plot(x_points, y_)

plt.plot(data[:50, 0], data[:50, 1], 'rx', label='0')
plt.plot(data[50:100, 0], data[50:100, 1], 'bo', label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()
