from neuralpack.model import DenseLayer,ReluLayer,mse,mse_prime,SerialModel
import numpy as np
import logging
from sample_generator import batch_generator, xor_sample_generator
import math

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)


# X = np.reshape([[0,0],[0,1],[1,0],[1,1]], (4,2,1))
# Y = np.reshape([1,0,0,1], (4,1,1))

model = SerialModel(layers=[
    DenseLayer(input_size=2, output_size=5),
    ReluLayer(),
    DenseLayer(input_size=5, output_size=5),
    ReluLayer(),
    DenseLayer(input_size=5, output_size=1)
])

n_samples = 100000
batch_size = 64

n_batches = math.ceil(n_samples/batch_size)
xor_generator = xor_sample_generator(rnd_seed=1)
xor_batch_generator = batch_generator(batch_size=64,sample_generator=xor_generator)


batch_errors = []
for b in range(n_batches):
    X,Y = next(xor_batch_generator)
    batch_error = model.train(batch_X=X, batch_Y=Y, learning_rate=0.0001)
    batch_errors.append(batch_error)


print(model.predict(np.reshape([0,0], (2,1))))
print(model.predict(np.reshape([0,1], (2,1))))
print(model.predict(np.reshape([1,0], (2,1))))
print(model.predict(np.reshape([1,1], (2,1))))

import matplotlib.pyplot as plt
plt.plot(batch_errors)
# plt.ylim(0,1)
plt.show()

