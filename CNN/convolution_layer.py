from scipy import signal
import numpy as np
from math import floor


class ConvolutionLayer:
    def __init__(self, input_shape, filter_depth, kernel_size, padding=0, stride=2):
        self.input_shape = input_shape
        self.input_depth = input_shape[0]  # depth = input[0], height = input[1], width = input[2]
        self.padding = padding
        self.stride = stride
        self.filter_depth = filter_depth
        self.kernel_size = kernel_size
        depth = input[0]
        height = input[1]
        width = input[2]
        self.output_shape = (depth, self.calculate_height_width(height), self.calculate_height_width(width))
        self.kernel_shape = (filter_depth, depth, kernel_size, kernel_size)
        self.biases, self.kernels = self.initialize_kernel_and_bias()
        self.input = 0
        self.output = 0


    def initialize_kernel_and_bias(self):
        biases = np.random.randn(*self.output_shape)
        kernels = np.random.randn(*self.kernels_shape)
        return biases, kernels


    def calculate_height_width(self, x):
        return floor(((x + 2 * self.padding - self.kernel_size) / self.stride) + 1)


    def forward_propogation(self, X):
        self.input = X
        self.output = np.copy(self.biases)
        for i in range(0, self.filter_depth, self.stride):
            for j in range(0, self.input_depth, self.stride):
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
        return self.output


    def backward_propogation(self, dE_dY, learning_rate):
        dE_dK = np.zeros(self.kernels_shape)     # kernel gradient
        dE_dX = np.zeros(self.input_shape)       # input gradient

        for i in range(0, self.filter_depth, self.stride):
            for j in range(0, self.input_depth, self.stride):
                dE_dK[i, j] = signal.correlate2d(self.input[j], dE_dY[i], "valid")
                dE_dX[j] += signal.convolve2d(dE_dY[i], self.kernels[i, j], "full")

        self.kernels -= learning_rate * dE_dK
        self.biases -= learning_rate * dE_dY
        return dE_dX
