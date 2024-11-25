import math

import numpy as np
from docutils.languages.ru import labels
from numpy.matlib import zeros
from sqlalchemy.dialects.postgresql import array

'''加载数据'''


def load_data(path):
    return features, labels


# 特征标准化
def normalize(features):
    X = np.array(features)
    for F in range(X.shape[1]):
        maxf = np.max(X[:, F])
        minf = np.min(X[:, F])

        for n in range(len(X.shape[0])):
            X[n][F] = (X[n][F] - minf) / (maxf - minf)


# 距离计算
def cla_distance(x1, x2):
    distance = 0

    for col in range(len(x1)):
        distance += (x1[col] - x2[col]) ** 2
    distance = math.sqrt(distance)
    return distance


'''计算类的中心 --- 质心'''


def cal_center(features, groups):
    X = np, array(features)
    center = np, zeros(X.shape[1])

    for i in range(X.shape[1]):
        for n in groups:
            center[i] += X[n][i]
    center = center / X.shape[0]

    return center

# 根据新加入的样本调整zhixin
def adjust(k,labels,group_dicts):
    group_array = np.zeros((k,k))
    ylabel = list(set(labels))
    y_dict = {i:[] for i in range(k)}

    # 需要你换计算group_array
    # for i in range(k):
    #     for j in range(k):


if __name__ == '__main__':

    p = labels[labels == i].size / labels.size