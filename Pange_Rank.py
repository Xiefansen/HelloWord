import numpy as np
from tqdm import tqdm

def page_rank(G):
    node = list(set(n for edge in G for n in edge))
    N = len(node)
    # 初始化转移矩阵
    S = np.zeros((N, N))

    for edge in G:
        S[edge[1], edge[0]] = 1
        # S[edge[0], edge[1]] = 1

    # 更新状态转移矩阵
    for j in range(N):
        sum_of_col = sum(S[:, j])
        for i in range(N):
            S[i, j] /= sum_of_col

    # 计算分数矩阵
    d = 0.8
    A = (1-d)/N + d*S

    # 初始化节点的pangeRank分数
    P_n= np.ones(N)/N
    P_n1 = np.zeros(N)

    e = 10000
    k = 0
    while e > 0.001:
        P_n1 = np.dot(A, P_n)
        e = P_n1 - P_n

        e = max(abs(e))

        P_n = P_n1
        k += 1

    return P_n


if __name__ == '__main__':

    G = [(0, 1), (1, 3), (3, 2), (2, 1), (1, 0)]

    S = page_rank(G)
    print(S)
