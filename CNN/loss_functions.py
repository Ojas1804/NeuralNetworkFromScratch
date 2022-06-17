import numpy as np


class LossFunctions:
    def __init__(self, error_function):
        self.error_function = self.L1_error
        if(error_function == "L2"):
            self.error_function = self.L2_error
        elif(error_function == "cross_entropy"):
            self.error_function = self.cross_entropy_error


    def L1_error(self, y_expected, y_predicted, derivative=False):
        if derivative:
            return 1/y_expected.shape[0]
        return 1/y_expected.shape[0] * np.sum(np.absolute(y_predicted - y_expected))


    def L2_error(self, y_expected, y_predicted, derivative=False):
        if derivative:
            return 1/y_expected.shape[0] * (y_predicted - y_expected)
        return 1/y_expected.shape[0] * np.sum((y_predicted - y_expected)**2)


    def cross_entropy_error(self, y_expected, y_predicted, derivative=False):
        from math import log
        if derivative:
            return -log(2) * y_expected/y_predicted
        return -1 * np.sum(y_expected * np.log(y_predicted) / log(2))