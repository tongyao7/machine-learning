import numpy as np


class HiddenMarkov:
    def forward(self, Q, V, A, B, O, PI):
        N = len(Q)  # 状态序列的大小
        M = len(O)  # 观测序列的大小
        alphas = np.zeros((N, M))  # 状态转移概率矩阵
        T = M  # 观测序列的长度
        for t in range(T):  # t是遍历时刻
            indexOfO = V.index(O[t])
            for i in range(N):
                if t == 0:
                    alphas[i][t] = PI[t][i] * B[i][indexOfO]
                    print('alpha1(%d)=p%db(o1)=%f' % (i, i, alphas[i][t]))
                else:
                    alphas[i][t] = np.dot([alpha[t - 1] for alpha in alphas], [a[i] for a in A]) * B[i][indexOfO]
                    print('alpha%d(%d)=[sigma alpha%d(i)ai%d]b%d(o%d)=%f' % (t, i, t - 1, i, i, t, alphas[i][t]))
                    # print(alphas)
        P = np.sum([alpha[M - 1] for alpha in alphas])
        print(P)

    def backward(self, Q, V, A, B, O, PI):
        N = len(Q)  # 状态序列的大小
        M = len(O)  # 观测序列的大小
        betas = np.ones((N, M))
        for i in range(N):
            print('beta%d(%d)=1' % (M, i))
        for t in range(M - 2, -1, -1):
            # 从最后一个元素向前，依次找出序列对应的索引
            indexOfO = V.index(O[t + 1])
            for i in range(N):
                betas[i][t] = np.dot(np.multiply(A[i], [b[indexOfO] for b in B]), [beta[t + 1] for beta in betas])
                realT = t + 1
                realI = i + 1
                print('beta%d(%d)=[sigma a%djbj(o%d)]beta%d(j)=(' % (realT, realI, realI, realT + 1, realT + 1), end='')
                for j in range(N):
                    print('%.2f*%.2f*%.2f+' % (A[i][j], B[j][indexOfO], betas[j][t + 1]), end='')
                print('0)=%.3f' % betas[i][t])
        print(betas)
        indexOfO = V.index(O[0])
        P = np.dot(np.multiply(PI, [b[indexOfO] for b in B]), [beta[0] for beta in betas])
        print([beta[0] for beta in betas])
        print('P(O|lambda)=', end='')
        for i in range(N):
            print('%.1f*%.1f*%.5f+' % (PI[0][i], B[i][indexOfO], betas[i][0]), end='')
        print('0=%f' % P)

    def viterbi(self, Q, V, A, B, O, PI):
        N = len(Q)
        M = len(O)
        deltas = np.zeros((N, M))
        psis = np.zeros((N, M))
        I = np.zeros((1, M))
        for t in range(M):
            realT = t + 1
            indexOfO = V.index(O[t])
            for i in range(N):
                realI = i + 1
                if t == 0:
                    deltas[i][t] = PI[0][i] * B[i][indexOfO]
                    psis[i][t] = 0
                    print('delta1(%d)=pi%d*b%d(o1)=%.2f*%.2f=%.2f' % (
                        realI, realI, realI, PI[0][i], B[i][indexOfO], deltas[i][t]))
                    print('psis1(%d)=0' % (realI))
                else:
                    deltas[i][t] = np.max(np.multiply([delta[t - 1] for delta in deltas], [a[i] for a in A])) * B[i][
                        indexOfO]
                    print('delta%d(%d)=max[delta%d(j)aj%d]b%d(o%d)=%.5f' % (
                        realT, realI, realT - 1, realI, realI, realT, deltas[i][t]))
                    psis[i][t] = np.argmax(np.multiply([delta[t - 1] for delta in deltas], [a[i] for a in A]))
                    print('psis%d(%d)=argmax[delta%d(j)aj%d]=%d' % (realT, realI, realT - 1, realI, psis[i][t]))
        print(deltas)
        print(psis)
        I[0][M - 1] = np.argmax([delta[M - 1] for delta in deltas])
        print('i%d=argmax[deltaT(i)]=%d' % (M, I[0][M - 1] + 1))
        for t in range(M - 2, -1, -1):
            I[0][t] = psis[int(I[0][t + 1])][t + 1]
            print('i%d=psis%d(i%d)=%d' % (t + 1, t + 2, t + 2, I[0][t] + 1))
        print(I)


# 习题10.1
Q = [1, 2, 3]
V = ['红', '白']
A = [[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]]
B = [[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]]
O = ['红', '白', '红', '白']
PI = [[0.2, 0.4, 0.4]]

HMM = HiddenMarkov()
# HMM.forward(Q, V, A, B, O, PI)
# HMM.backward(Q, V, A, B, O, PI)
HMM.viterbi(Q, V, A, B, O, PI)
