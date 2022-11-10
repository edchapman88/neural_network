import numpy as np
from typing import Callable
import logging
logger = logging.getLogger(__name__)


class BaseLayer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self,input):
        self.input = input

    def backward(self,e_grad_out,learning_rate):
        pass


class DenseLayer(BaseLayer):
    def __init__(self,input_size,output_size):
        self.weights = np.random.randn(output_size,input_size)
        self.bias = np.random.randn(output_size,1)

    def forward(self, input):
        # Y = W.X + B
        # shapes (j,) = (j,i).(i,) + (j,)
        self.input = input
        return np.dot(self.weights,input) + self.bias

    def backward(self, e_grad_out, learning_rate):
        '''
        Matrix Algebra:

        dE/dW = dE/dY . transpose(X)

        dE/DB = dE/dY

        dE/dX = transpose(W) . dE/dY
        '''
        e_grad_weights = np.dot(e_grad_out,np.transpose(self.input))
        e_grad_bias = e_grad_out
        e_grad_in = np.dot(np.transpose(self.weights),e_grad_out)
        self.weights -= learning_rate * e_grad_weights
        self.bias -= learning_rate * e_grad_bias
        return e_grad_in

class ActivationLayer(BaseLayer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(input)

    def backward(self, e_grad_out, learning_rate):
        '''
        dE/dW = dE/dY .(elemwise) f_prime(X)
        '''
        return np.multiply(e_grad_out,self.activation_prime(self.input))

class TanhLayer(ActivationLayer):
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        tanh_prime = lambda x: 1 - np.tanh(x)**2
        super().__init__(tanh, tanh_prime)

class ReluLayer(ActivationLayer):
    def __init__(self):
        relu = lambda x: np.maximum(x,np.zeros_like(x))
        relu_prime = lambda x: np.where(x<=0,0,1)
        super().__init__(relu, relu_prime)

def mse(y_true,y_pred):
    return np.mean(np.power((y_true - y_pred),2))

def mse_prime(y_true,y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)
        

