import numpy as np
from .metrics import r2_score
from .metrics import mean_squared_error


class LinearRegression:

    def __init__(self):
        """初始化模型"""
        self.coef_ = None
        self.interception_ = None
        self._theta = None

    def fit_normal(self, x_train, y_train):
        """使用正规解训练模型"""
        assert x_train.shape[0] == y_train.shape[0], \
            "the size of x_train must be equal to the size of y_train"

        x_b = np.hstack([np.ones(len(x_train), 1), x_train])
        self._theta = np.linalg.inv(x_b.dot(x_b)).dot(x_b.T).dot(y_train)
        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def fit_gd(self, x_train, y_train, eta=0.001, n_iters=1e4, epsilon=1e-8):
        """使用梯度下降法训练模型"""
        assert x_train.shape[0] == y_train.shape[0], \
            "the size of x_train must be equal to the size of y_train"

        def J(theta, x_b, y):
            """cost function J(θ)"""
            try:
                return mean_squared_error(y, x_b.dot(theta)) / 2.
            except:
                return float('inf')

        def dJ(theta, x_b, y):
            """the derivative of J(θ)"""
            return x_b.T.dot(x_b.dot(theta) - y) / len(y)

        def gradient_descent(x_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):
            """梯度下降法"""

            theta = initial_theta
            cur_iters = 0

            while cur_iters < n_iters:
                gradient = dJ(theta, x_b, y)
                last_theta = theta
                theta -= eta * gradient
                if abs(J(theta, x_b, y) - J(theta, x_b, y)) < epsilon:
                    break

                cur_iters += 1

                return theta

        x_b = np.hstack([np.ones(len(x_train), 1), x_train])
        initial_theta = np.zeros(x_b.shape[0])
        self._theta = gradient_descent(x_b, y_train, initial_theta, eta, n_iters, epsilon)

        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def predict(self, x_test):
        """根据x_test预测y_predict"""
        assert self.interception_ is not None and self.coef_ is not None, \
            "must fit before predict!"
        assert x_test.shape[1] == len(self.coef_), \
            "the feature number of X_predict must be equal to X_train"

        x_b = np.hstack([np.ones(len(x_test), 1), x_test])
        return x_b.dot(self._theta)

    def score(self, x_test, y_test):
        """根据测试集给出模型预测准确度"""

        y_predict = self.predict(x_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "LinearRegression()"
