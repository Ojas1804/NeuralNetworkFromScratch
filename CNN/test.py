from convolution_layer import ConvolutionLayer as conv
from pooling_layer import PoolingLayer as pool
from define_layers import Layers
from reshape_layer import ReshapeLayer as reshape
from ann import ANN
from activation_layer import ActivationLayer as act
from cnn import CNN

def main():
    conv_layer_1 = conv(3, 3, 1, 1, 1, 1)
    act_layer_1 = act('relU')
    pool_layer_1 = pool(2, 2)
    conv_layer_2 = conv(3, 3, 1, 1, 1, 1)
    act_layer_2 = act('relU')
    pool_layer_2 = pool(2, 2)
    conv_layer_3 = conv(3, 3, 1, 1, 1, 1)
    act_layer_3 = act('relU')
    pool_layer_3 = pool(2, 2)
    reshape_layer = reshape()
    ann_layer = ANN(1, 1, 1, 0.01)

    layers = Layers([conv_layer_1, act_layer_1, pool_layer_1, 
                    conv_layer_2, act_layer_2, pool_layer_2, 
                    conv_layer_3, act_layer_3, pool_layer_3, 
                    reshape_layer, 
                    ann_layer])

    cnn_ = CNN(layers)
    # cnn_.train(inputs, target)
    # cnn_.predict(inputs)
    


if __name__ == '__main__':
    main()