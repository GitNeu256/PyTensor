from .module import Module
import numpy as np

class Linear(Module):
    def __init__(self, in_features, out_features):
        self.weights = np.random.randn(out_features, in_features)
        self.bias = np.zeros((out_features, 1))
        self.input = None
        self.grad_input = None
        self.grad_weights = None
        self.grad_bias = None

    def forward(self, input):
        self.input = input
        output = np.dot(self.weights, input) + self.bias
        return output

    def backward(self, grad_output):
        self.grad_input = np.dot(self.weights.T, grad_output)
        self.grad_weights = np.outer(grad_output, self.input.T)
        self.grad_bias = grad_output.sum(axis=1, keepdims=True)

        return self.grad_input

    def __call__(self, input):
        return self.forward(input)