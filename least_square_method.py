import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import leastsq

# (x,y)为训练数据，x_points为测试数据
x = np.linspace(0, 1, 10)
x_points = np.linspace(0, 1, 1000)
y_ = np.sin(2 * np.pi * x)
y = [np.random.normal(0, 0.1) + y1 for y1 in y_]


# 需要拟合的函数
def func(p, x):
    f = np.poly1d(p)
    return f(x)


# 误差函数
def error(p, x, y):
    return func(p, x) - y


# 初始值
def fitting(M=0):
    '''M为多项式的次数'''
    # 随机化多项式参数
    p_init = np.random.rand(M + 1)
    # 最小二乘法
    paras = leastsq(error, p_init, args=(x, y))
    print('Fitting parameters:', paras[0])

    # 可视化
    plt.plot(x_points, np.sin(2 * np.pi * x_points), label='real')
    plt.plot(x_points, func(paras[0], x_points), label='fitted curve')
    plt.plot(x, y, 'bo', label='noise')
    plt.legend(loc='upper right')
    return paras


# M0=fitting(M=0)
# M1=fitting(M=1)
# M4=fitting(M=3)
M9 = fitting(M=9)

# M=9时，过拟合
regularization = 0.0001


def residuals_func_regularization(p, x, y):
    ret = func(p, x) - y

    # L2范数作为正则化项
    ret = np.append(ret, np.sqrt(0.5 * regularization * np.square(p)))
    return ret


p_init = np.random.rand(9 + 1)
p_regularization = leastsq(residuals_func_regularization, p_init, args=(x, y))

plt.plot(x_points, np.sin(2 * np.pi * x_points), label='real')
plt.plot(x_points, func(M9[0], x_points), label=' regularized fitted curve')
plt.plot(x, y, 'bo', label='noise')
plt.legend(loc='upper right')

plt.show()
