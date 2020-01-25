import numpy as np


class StandardScaler:

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def standard_fit(self, X):
        """根据训练数据集X获得训练数据集的方差和标准差"""
        assert X.ndim == 2, \
            "The dimension of X must be 2"

        self.mean_ = np.array([np.mean(X[:, i]) for i in range(len(X.shape[1]))])
        self.scale_ = np.array([np.std(X[:, i]) for i in range(len(X.shape[1]))])

        return self

    def standard_transform(self, X):
        """将X根据这个StandardScaler进行均值方差归一化处理"""

        assert X.ndim == 2, \
            "The dimension of X must be 2"
        assert self.mean_ is not None and self.scale_ is not None, \
            "must fit before transform!"
        assert X.shape[1] == len(self.mean_), \
            "the feature number of X must be equal to mean_ and std_"

        standard_resX = np.empty(shape=X.shape, dtype=float)
        for col in range(X.shape[1]):
            standard_resX[:, col] = (X[:, col] - self.mean_) / self.scale_

        return standard_resX


class MaxMinScaler:

    def __init__(self):
        self.max_ = None
        self.min_ = None

    def max_min_fit(self, X):
        """计算X的最大值和最小值"""
        self.max_ = np.array([np.max(X[:, i]) for i in range(X.shape[1])])
        self.min_ = np.array([np.min(X[:, i]) for i in range(X.shape[1])])

    def max_min_transform(self, X):
        """将X根据最值进行归一化处理"""
        assert X.ndim == 2, \
            "The dimension of X must be 2"
        assert self.max_ is not None and self.min_ is not None, \
            "must fit before transform!"
        assert X.shape[1] == len(self.max_), \
            "the feature number of X must be equal to mean_ and std_"

        max_min_resX = np.empty(shape=X.shape, dtype=float)
        for col in range(X.shape[1]):
            max_min_resX[:, col] = (X[:, col] - self.min_) / (self.max_ - self.min_)

        return max_min_resX
