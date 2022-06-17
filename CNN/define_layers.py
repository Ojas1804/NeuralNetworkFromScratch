class Layers:
    def __init__(self, layers):
        self.layers = layers


    def forward(self, input):
        Y = input
        for layer in self.layers:
            Y = layer.forward(Y)


    def backward(self, dE_dY):
        for layer in reversed(self.layers):
            dE_dY = layer.backward(dE_dY)