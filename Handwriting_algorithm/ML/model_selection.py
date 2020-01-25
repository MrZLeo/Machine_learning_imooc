import numpy as np


def train_test_split(X, y, test_ratio=0.2, seed=None):
    """将数据 X 和 y 按照test_ratio 分割成X_train, X_test, y_train, y_test"""
    assert X.shape[0] == y.shape[0], \
        "the size of X must be equal to the size of y"
    assert 0.0 <= test_ratio <= 1.0, \
        "test_ration must be valid"

    if seed:
        np.random.seed(seed)

    # 获得一个长度为X的，0-x的随机数组
    shuffled_indexes = np.random.permutation(len(X))

    # 获得随机分配的test和train的index，使用fancy indexing的方法
    test_size = int(len(X) * test_ratio)
    test_indexes = shuffled_indexes[:test_size]
    train_indexes = shuffled_indexes[test_size:]

    # 将原数组进行分组
    X_test = X[test_indexes]
    X_train = X[train_indexes]

    y_test = y[test_indexes]
    y_train = y[train_indexes]

    return X_test, X_train, y_test, y_train
