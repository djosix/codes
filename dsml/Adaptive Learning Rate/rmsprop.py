import numpy as np

class RMSProp:
    def __init__(self, eta=0.001, alpha=0.9, epsilon=1e-8):
        self.eta = eta
        self.alpha = alpha
        self.epsilon = epsilon
        self.sigma = None

    def __call__(self, g):
        if self.sigma is None:
            self.sigma = g
        else:
            self.sigma = np.sqrt(self.alpha * self.sigma ** 2
                                 + (1 - self.alpha) * g ** 2)
        return (self.eta * g) / self.sigma

if __name__ == '__main__':
    from test import linear_regression
    linear_regression(RMSProp(eta=1))
