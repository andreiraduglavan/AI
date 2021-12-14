import numpy as np
from keras.datasets import mnist

def one_hot(Y):
    one_hot_Y = np.zeros(10)
    one_hot_Y[Y] = 1
    return one_hot_Y

(train_X, train_y), (test_X, test_y)=mnist.load_data()
training_data=[(x.flatten()/255, one_hot(y)) for x,y in zip(train_X, train_y)]
test_data=[(x.flatten()/255, y) for x,y in zip(test_X, test_y)]

import network
net = network.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
