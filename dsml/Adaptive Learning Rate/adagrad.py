import numpy as np

class Adagrad:
    def __init__(self, eta=0.01):
        self.eta = eta
        self.g_2_sum = 0

    def __call__(self, g):
        self.g_2_sum += (g ** 2)
        sigma = np.sqrt(self.g_2_sum)
        eta = self.eta
        return (g * eta) / sigma


if __name__ == '__main__':
    from test import linear_regression
    linear_regression(Adagrad(eta=10000000))
