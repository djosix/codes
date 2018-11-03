import numpy as np

class Adam:
    def __init__(self, eta=0.001, alpha=1.0, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.eta = eta
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = 0
        self.v = 0
        self.t = 0
    
    def __call__(self, g):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        self.v = self.beta2 * self.v + (1 - self.beta2) * g * g
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        return (self.alpha * m_hat) / (np.sqrt(v_hat) + self.epsilon)


if __name__ == '__main__':
    from test import linear_regression
    linear_regression(Adam(), max_iter=100)
