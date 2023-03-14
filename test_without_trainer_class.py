from neuralpack.model import DenseLayer,ReluLayer,mse,mse_prime,SerialModel
import numpy as np
import logging
from utils import batch_generator, xor_endless_random_sample_generator, plot_xor_decision_boundary
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d  


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

model = SerialModel(layers=[
    DenseLayer(input_size=2, output_size=5),
    ReluLayer(),
    DenseLayer(input_size=5, output_size=5),
    ReluLayer(),
    DenseLayer(input_size=5, output_size=5),
    ReluLayer(),
    DenseLayer(input_size=5, output_size=5),
    ReluLayer(),
    DenseLayer(input_size=5, output_size=1)
])

n_samples = 100000
batch_size = 1  # equivalent to 'stocastic gradient decent' when batch size = 1
# for this XOR task it is counter-productive to use mini-batch decent, because it averages the error
# across samples from all 4 classes and so the mapping between inputs and outputs is lost in the average.
n_batches = math.ceil(n_samples/batch_size)

# dataset specific sample generator
xor_generator = xor_endless_random_sample_generator(rnd_seed=1)

# generic batch generator that takes a sample generator as argument
xor_batch_generator = batch_generator(batch_size=batch_size,sample_generator=xor_generator)


batch_errors = []
for b in range(1,n_batches+1):
    X,Y = next(xor_batch_generator)
    batch_error = model.train(batch_X=X, batch_Y=Y, learning_rate=0.1)
    batch_errors.append(batch_error)

# predictions will be deterministic (no stochastic elements in the network)
logger.info(model.predict(np.reshape([0,0], (2,1))))
logger.info(model.predict(np.reshape([0,1], (2,1))))
logger.info(model.predict(np.reshape([1,0], (2,1))))
logger.info(model.predict(np.reshape([1,1], (2,1))))

# plot learning
plt.plot(batch_errors)
plt.show()

# plot decision boundary in 3d
plot_xor_decision_boundary(model)




