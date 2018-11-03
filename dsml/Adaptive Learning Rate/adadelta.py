import numpy as np

class Adadelta:
    def __init__(self, eta=1.0, rho=0.95, epsilon=1e-8):
        self.eta = eta
        self.rho = rho # decay rate
        self.epsilon = epsilon
        self.G = 0
        self.D = 0
    
    def __call__(self, g):
        self.G = self.rho * self.G + (1 - self.rho) * g * g
        _g = self.eta * g * np.sqrt(self.D + self.epsilon) / np.sqrt(self.G + self.epsilon)
        self.D = self.rho * self.D + (1 - self.rho) * _g ** 2
        return _g


if __name__ == '__main__':
    from test import linear_regression
    linear_regression(Adadelta(1))
