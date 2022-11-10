from neuralpack.model import DenseLayer,ReluLayer,mse,mse_prime
import numpy as np
import logging
import os
logger = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


X = np.reshape([[0,0],[0,1],[1,0],[1,1]], (4,2,1))
Y = np.reshape([ 0,    1,    1,    0], (4,1,1))

model = [
    DenseLayer(input_size=2, output_size=3),
    ReluLayer(),
    DenseLayer(input_size=3, output_size=1)
]

n_epochs = 20
error_record = []

for epoch in range(n_epochs):
    logger.info(f'epoch: {epoch}')
    sum_error_epoch = 0

    for i,sample in enumerate(X):

        output = sample
        for layer in model:
            output = layer.forward(input=output)

        error = mse(y_true=Y[i], y_pred=output)
        sum_error_epoch += error

        e_grad_out = mse_prime(y_true=Y[i], y_pred=output)

        for layer in reversed(model):
            e_grad_in = layer.backward(e_grad_out=e_grad_out, learning_rate=0.1)
            e_grad_out = e_grad_in

    error_record.append(sum_error_epoch)



import matplotlib.pyplot as plt
plt.plot(error_record)
plt.show()

