import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']

data = np.array(df.iloc[:100, [0, 1, -1]])
X, y = data[:, :-1], data[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


class KNN:
    def __init__(self, X_train, y_train, k=3, p=2):
        '''k为临近点个数，p为距离度量'''
        self.X_train = X_train
        self.y_train = y_train
        self.k = k
        self.p = p

    def predict(self, X):
        '''根据模型，返回输出的类。找出k个最近的点'''
        knn_list = []
        for i in range(self.k):
            dist = np.linalg.norm(X - self.X_train[i], ord=self.p)
            knn_list.append((dist, self.y_train[i]))

        for i in range(self.k, len(self.X_train)):
            max_index = knn_list.index(max(knn_list, key=lambda x: x[0]))
            dist = np.linalg.norm(X - self.X_train[i], ord=self.p)
            if knn_list[max_index][0] > dist:
                knn_list[max_index] = (dist, self.y_train[i])

        knn = [k[-1] for k in knn_list]
        count = Counter(knn)
        return count.most_common(1)[0][0]

    def score(self, X_test, y_test):
        right_count = 0
        for X, y in zip(X_test, y_test):
            if self.predict(X) == y:
                right_count += 1
        return right_count / len(X_test)


clf = KNN(X_train, y_train)

test_point = [5.5, 3.5]
print('Test Point:{}, predict:{}'.format(test_point, clf.predict(test_point)))

score = clf.score(X_test, y_test)
print('score=', score)

plt.plot(data[:50, 0], data[:50, 1], 'bo', label='0')
plt.plot(data[50:100, 0], data[50:100, 1], 'rx', label='1')
plt.plot(test_point[0], test_point[1], 'go', label='test point')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()

# -------------------------------------------------------
from sklearn.neighbors import KNeighborsClassifier

clf_sk = KNeighborsClassifier()
clf_sk.fit(X_train, y_train)

y_predict = clf_sk.predict(X_test)
score = clf_sk.score(X_test, y_test)
print(score)

print('acc:{}'.format(sum(y_predict == y_test) / len(X_test)))
