


class ActivationLayer:
    def __init__(self, activation):
        self.input = None
        self.activation = activation

    def forward(self, Z):
        self.input = Z
        return self.activation(Z)

    def backward(self, dA):
        return dA * self.activation(self.input, derivative = True)