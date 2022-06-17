import numpy as np


class PoolingLayer:
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride


    def forward_propogation(self, X):
        self.input = X
        self.output = self.pool(X)
        return self.output


    def backward_propogation(self, dE_dY, learning_rate):
        return self.unpool(dE_dY)


    def pool(self, X):
        return np.max(X, axis=(2, 3))


    def unpool(self, dE_dY):
        return np.repeat(dE_dY, self.stride, axis=2)