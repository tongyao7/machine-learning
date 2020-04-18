from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def create_data():
    iris = load_iris()
    print(iris)
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width',
                  'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0, 1, -1]])
    return data[:, :2], data[:, -1]


X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


class LogisticRegressionClassifier:
    def __init__(self, max_iter=200, learning_rate=0.01):
        self.max_iter = max_iter
        self.learning_rate = learning_rate

    def data_matrix(self, X):  # 扩充权值向量
        data_mat = []
        for d in X:
            data_mat.append([1.0, *d])
        return data_mat

    def sigmoid(self, X):
        return 1 / (1 + math.exp(-X))

    def fit(self, X, y):
        self.w = np.zeros((len(X[0]) + 1, 1), dtype=np.float32)
        mat = self.data_matrix(X)

        for iter_ in range(self.max_iter):  # 梯度下降
            for i in range(len(X)):
                result = self.sigmoid(np.dot(mat[i], self.w))
                error = y[i] - result
                self.w += self.learning_rate * error * np.transpose([mat[i]])
        print('Model: learning_rate={}, max_iter={}'.format(
            self.learning_rate, self.max_iter))

    def score(self, X, y):
        right_count = 0
        X = self.data_matrix(X)
        for x, y in zip(X, y):
            result = np.dot(x, self.w)
            if (result > 0 and y == 1) or (result < 0 and y == 0):
                right_count += 1
        return right_count / len(X)


clf = LogisticRegressionClassifier()
clf.fit(X_train, y_train)
print(clf.w)

x_point = np.arange(4, 8)
y_ = -(clf.w[1] * x_point + clf.w[0]) / clf.w[2]
plt.plot(x_point, y_)

plt.plot(X[:50, 0], X[:50, 1], 'rx', label='0')
plt.plot(X[50:, 0], X[50:, 1], 'bo', label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()

plt.show()

score = clf.score(X_test, y_test)
print('score=', score)

# --------------------------------------------------------


clf = LogisticRegression(max_iter=200)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
print(clf.coef_, clf.intercept_)

x_points = np.arange(4, 8)
y_ = -(clf.coef_[0][0] * x_points + clf.intercept_) / clf.coef_[0][1]
plt.plot(x_points, y_)  # 画出分类直线

plt.plot(X[:50, 0], X[:50, 1], 'rx', label='0')
plt.plot(X[50:, 0], X[50:, 1], 'bo', label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()
