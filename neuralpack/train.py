from model import SerialModel
from typing import Callable, Iterator
import numpy as np
class Trainer:
    def __init__(self, model:SerialModel, sample_generator:Callable([],Iterator)):
        self.model = model
        self.sample_generator = sample_generator
        self.sample_itr = sample_generator()

    def run(self, batch_size=16, epochs=1, learning_rate=0.0001):
        self.batch_itr = self.batch_generator(batch_size=batch_size,sample_itr=self.sample_itr)

        batch_errors = []
        for epoch in epochs:
            try:
                X,Y = next(self.batch_itr)
            except StopIteration:
                self.sample_itr = self.sample_generator()
                self.batch_itr = self.batch_generator(batch_size=batch_size,sample_itr=self.sample_itr)
                continue
            batch_error = self.model.train(batch_X=X, batch_Y=Y, learning_rate=learning_rate)
            batch_errors.append(batch_error)

        return batch_errors

    def batch_generator(batch_size, sample_itr):
        while True:
            X = []
            Y = []

            for i in range(batch_size):
                x,y = next(sample_itr)
                X.append(x)
                Y.append(y)

            X = np.reshape(X, (batch_size,2,1))
            Y = np.reshape(Y, (batch_size,1,1))

            yield X,Y