from neuralpack.model import DenseLayer,ReluLayer,mse,mse_prime,SerialModel
from neuralpack.train import Trainer
from utils import xor_data_generator_func, plot_xor_decision_boundary
import matplotlib.pyplot as plt

model = SerialModel(layers=[
    DenseLayer(input_size=2, output_size=5),
    ReluLayer(),
    DenseLayer(input_size=5, output_size=5),
    ReluLayer(),
    DenseLayer(input_size=5, output_size=1)
])

trainer = Trainer(model=model,sample_generator_func=xor_data_generator_func)

# in this case one epoch is only 4 samples.
batch_errors = trainer.run(batch_size=1,epochs=1,learning_rate=0.1)

# plot errors over training run
plt.plot(batch_errors)
plt.show()

plot_xor_decision_boundary(model)

# do more training
batch_errors = trainer.run(batch_size=3,epochs=1000,learning_rate=0.1)
plt.plot(batch_errors)
plt.show()
plot_xor_decision_boundary(model)