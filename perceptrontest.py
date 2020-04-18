import random
import numpy as np
import matplotlib.pyplot as plt


def sign(v):
    if v >= 0:
        return 1
    else:
        return -1


def train(train_num, train_datas, lr):
    w = [0, 0]
    b = 0
    for i in range(train_num):
        x = random.choice(train_datas)
        x1, x2, y = x
        if y * sign(w[0] * x1 + w[1] * x2 + b) <= 0:
            w[0] += lr * y * x1
            w[1] += lr * y * x2
            b += lr * y
    return w, b


def plot_point(train_datas, w, b):
    plt.figure()
    x1 = np.linspace(0, 8, 100)
    x2 = -(w[0] * x1 + b) / w[1]
    plt.plot(x1, x2, 'r', label='y1 data')
    for i in range(len(train_datas)):
        if train_datas[i][-1] == 1:
            plt.scatter(train_datas[i][0], train_datas[i][1], c='b', s=50)
        else:
            plt.scatter(train_datas[i][0], train_datas[i][1], c='r', marker='x', s=50)
    plt.show()


def train2(train_num, train_datas, lr):
    w = 0.0
    b = 0
    alpha = [0 for i in range(len(train_datas))]
    train_array = np.array(train_datas)
    gram = np.matual(train_array[:, 0:-1], train_array[:, 0:-1].T)
    for idx in range(train_num):
        tmp = 0
        i = random.randint(0, len(train_datas) - 1)
        yi = train_array[i, -1]
        for j in range(len(train_datas)):
            tmp += alpha[j] * train_array[j, -1] * gram[i, j]
        tmp += b
        if yi * tmp <= 0:
            alpha[i] = alpha[i] + lr
            b = b + lr * yi
    for i in range(len(train_datas)):
        w += alpha[i] * train_array[i, -1] * train_array[i, 0:-1]
    return w, b, alpha, gram


train_data1 = [[1, 3, 1], [2, 2, 1], [3, 8, 1], [2, 6, 1]]
train_data2 = [[2, 1, -1], [4, 1, -1], [6, 2, -1], [7, 3, -1]]
train_datas = train_data1 + train_data2

# train_num指训练次数
# w,b=train(train_num=50,train_datas=train_datas,lr=0.01)
w, b, alpha, gram = train(train_num=500, train_datas=train_datas, lr=0.01)
plot_point(train_datas, w, b)
