import numpy as np
import math


class EM:
    def __init__(self, prob):
        self.pro_A, self.pro_B, self.pro_C = prob

    # 参数mu,表示硬币B的概率
    def pmf(self, i):
        pro_1 = self.pro_A * math.pow(self.pro_B, data[i]) * math.pow(1 - self.pro_B, 1 - data[i])
        pro_2 = (1 - self.pro_A) * math.pow(self.pro_C, data[i]) * math.pow(1 - self.pro_C, 1 - data[i])
        return pro_1 / (pro_1 + pro_2)

    def fit(self, data):
        length = len(data)
        print('init prob:{},{},{}'.format(self.pro_A, self.pro_B, self.pro_C))
        for d in range(length):
            _ = yield
            _pmf = [self.pmf(k) for k in range(length)]
            pro_A = 1 / length * sum(_pmf)
            pro_B = sum([_pmf[k] * data[k] for k in range(length)]) / sum([_pmf[k] for k in range(length)])
            pro_C = sum([(1 - _pmf[k]) * data[k] for k in range(length)]) / sum([(1 - _pmf[k]) for k in range(length)])
            print('{}/{} pro_a:{:.3f},pro_b:{:.3f},pro_c:{:.3f}'.format(d + 1, length, pro_A, pro_B, pro_C))
            self.pro_A = pro_A
            self.pro_B = pro_B
            self.pro_C = pro_C


# 观测数据
data = [1, 1, 0, 1, 0, 0, 1, 0, 1, 1]

em = EM(prob=[0.5, 0.5, 0.5])
f = em.fit(data)
next(f)
f.send(1)
f.send(2)

em = EM(prob=[0.4, 0.6, 0.7])
f2 = em.fit(data)
next(f2)
f2.send(1)
f2.send(2)
