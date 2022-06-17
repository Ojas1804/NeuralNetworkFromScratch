# from define_layers import Layers

class CNN:
    def __init__(self, layers, epochs=1000, learning_rate=0.01):
        self.layers = layers
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.X = 0
        self.Y = 0

    def train(self, X, Y):
        self.X = X
        self.Y = Y
        for _ in range(self.epochs):
            self.layers.forward(self.X)
            self.layers.backward(self.Y, self.learning_rate)

    def predict(self, X):
        self.layers.forward(X)
        return self.layers.output