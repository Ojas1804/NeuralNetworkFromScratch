import numpy as np


class ActivationFunctions:
    def __init__(self):
        pass

    
    def relU(self, X, derivative=False):
        if derivative:
            return 1.0 * (X > 0)
        return X * (X > 0)


    def sigmoid(self, X, derivative=False):
        if derivative:
            return X * (1 - X)
        return 1 / (1 + np.exp(-X))


    def tanh(self, X, derivative=False):
        if derivative:
            return 1 - X ** 2
        return np.tanh(X)