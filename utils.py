from random import random,seed
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def xor_data_generator_func():
    # will throw StopIteration after these 4 yields
    # in this example, the full dataset has only 4 datapoints
    yield [0,0],0
    yield [0,1],1
    yield [1,0],1
    yield [1,1],0

def xor_endless_random_sample_generator(rnd_seed=1):
    seed(rnd_seed)
    while True:
        rnd = random()
        if rnd < 0.25:
            yield [0,0],1

        elif rnd < 0.5:
            yield [0,1],0

        elif rnd < 0.75:
            yield [1,0],0

        else:
            yield [1,1],1
        

def batch_generator(batch_size, sample_generator):
    while True:
        X = []
        Y = []
    
        for i in range(batch_size):
            x,y = next(sample_generator)
            X.append(x)
            Y.append(y)
    
        X = np.reshape(X, (batch_size,2,1))
        Y = np.reshape(Y, (batch_size,1,1))
    
        yield X,Y

# plot XOR decision boundary function
def plot_xor_decision_boundary(model):
    x = np.linspace(0, 1, 20)
    y = np.linspace(0, 1, 20)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    for i in range(X.shape[1]):
        for j in range(X.shape[0]):
            Z[j,i] = model.predict(np.reshape([X[j,i],Y[j,i]], (2,1)))
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()