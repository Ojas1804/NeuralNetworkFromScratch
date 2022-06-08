from math import floor
from scipy import signal
import numpy as np

class ConvolutionalLayer:
    def __init__(self, input_shape, kernel_size, depth=1, padding=0, stride=1):
        self.depth = depth
        self.padding = padding
        self.stride = stride
        self.input_shape = input_shape
        self.input_depth = self.input_shape[2]
        self.output_shape = (self.depth, self.calculate_output_length(input_shape[0], kernel_size, padding, stride), 
                             self.calculate_output_length(input_shape[1], kernel_size, padding, stride))
        self.kernel_shape = (depth, self.input_depth, kernel_size, kernel_size)
        self.kernels, self.bias = self.initialize_kernels_and_bias(self.kernel_shape, self.output_shape)
        self.input = np.zeros(input_shape)
        self.output = np.zeros(self.output_shape)


    def calculate_output_length(self, l, f, p, s):
        return floor((l + 2 * p - f) / s)


    def initialize_kernels_and_bias(self, kernel_shape, output_shape):
        kernels = np.random.randn(*kernel_shape)
        bias = np.random.randn(*output_shape)
        return kernels, bias


    def forward_pass(self, input):
        self.input = input
        self.output = np.zeros(self.output_shape)
        self.output += self.bias

        for i in range(self.depth):
            for j in range(self.input_shape[0]):
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
        return self.output


    def back_propagation(self, prev_gradient, alpha):
        dK = np.zeros(self.kernels_shape)  # kernel gradient
        dB = prev_gradient                 # bias gradient
        dA = np.zeros(self.input_shape)    # input gradient

        for i in range(self.depth):
            for j in range(self.input_depth):
                dK[i, j] = signal.correlate2d(self.input[j], prev_gradient[i], "valid")
                dA[j] += signal.convolve2d(prev_gradient[i], self.kernels[i, j], "full")
        
        self.update_parameters(dK, dB, alpha)
        return dA


    def update_parameters(self, dK, dB, alpha):
        self.kernels -= alpha * dK
        self.biases -= alpha * dB