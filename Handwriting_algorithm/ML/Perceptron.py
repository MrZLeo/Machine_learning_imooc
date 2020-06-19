import numpy as np


class Perceptron:
    """neural network, perceptron"""

    def __init__(self, sizes):
        assert sizes > 0, "size must be bigger than 0"
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weights = [np.random.randn(n, m) for m, n in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.random.randn(n, 1) for n in sizes[1:]]

    def fit(self, X, y, eta, iterations, activation="ReLu"):
        assert self.weights is not None, "weights must be inited"

        n = 0
        while n < iterations:
            # h = self._dot(X, self.weights)
            # v = self._activation(activation, h)
            n += 1

    def _ReLU(self, h):
        if h > 0 or h == 0:
            return h
        else:
            return 0

    def _sigmoid(self, h):
        return 1.0 / (1.0 + np.exp(-h))

    def _init_weight(self, size):
        return np.ones(len(size))

    def _dot(self, X, w):
        return X.dot(w.T)

    def _activation(self, activation, h):
        if activation == "ReLu":
            return self._ReLU(h)
        elif activation == "sigmoid":
            return self._sigmoid(h)

    def dJ(self, w):
        """the derivative of E(w)"""
        # if w ==

    def J(self, theta, x_b, y):
        """cost function J(θ)"""
        try:
            return
        except:
            return float('inf')

    # def feed_forward(self, X):
