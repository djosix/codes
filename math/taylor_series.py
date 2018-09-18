import numpy as np
import matplotlib.pyplot as plt
from functools import lru_cache

@lru_cache(100)
def factorial(n):
    return 1. if n == 0 else n * factorial(n-1)

def taylor_series(coefs):
    def func(x):
        x_n = np.ones(x.shape)
        for n, c in enumerate(coefs):
            if n: x_n *= x
            yield (c * x_n) / factorial(n)
    return func


xs = np.linspace(-10, 10, 100)

# sin
coefs = [0, 1, 0, -1] * 8

# cos
# coefs = [1, 0, -1, 0] * 10

# exp
# coefs = [1] * 40


ys = 0
ts = taylor_series(coefs)

plt.ion()

for n in ts(xs):
    ys += n
    plt.clf()
    
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)

    plt.axvline(x=0, c='black', linewidth=0.5)
    plt.axhline(y=0, c='black', linewidth=0.5)

    plt.plot(xs, ys)
    plt.pause(0.001)

