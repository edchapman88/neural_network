from random import random,seed
import numpy as np

def xor_sample_generator(rnd_seed=1):
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

