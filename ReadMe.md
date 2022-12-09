## Intro
Building a neural network package.

## Summary
A `SerialModel` class builds a model from an array of layers (much like Keras). Mini-batch and stochastic gradient descent are available. If the `batch_size` passed to the `model.train()` method has only one sample, then stocastic gradient descent is used. 

A `Trainer` class handles training of a model. A dataset is passed to the Trainer as a generator function. `batch_size` and the number of `epochs` are passed as arguments to the `Trainer.run()` method along with the `learning_rate`. `Trainer.run()` may be called again to train a model further.

## Get started
Create a virtual enviroment:
```
    python3 -m venv .venv
```

Install the project dependencies:
```
    pip install -r requirements.txt
```

Run tests with or without the `Trainer` helper class. The model is tested against the XOR function (which is not linearly seperable). 3D plots visualize the learnt decision boundary of the model within the 0 -> 1 domain of both of the two function inputs.
```
    python test_without_trainer_class.py
```

or,
```
    python test_with_trainer_class.py
```