import numpy as np
import matplotlib.pyplot as plt
from functools import lru_cache

@lru_cache(100)
def factorial(n):
    return 1. if n == 0 else n * factorial(n-1)

def taylor_series(coefs):
    def func(x):
        x_n = 1
        for n, c in enumerate(coefs):
            if n: x_n *= x
            yield (c * x_n) / factorial(n)
    return func

def sin(x):
    if x >= -np.pi and x <= np.pi:
        ts_sin = taylor_series([0, 1, 0, -1] * 3)
        return sum(ts_sin(x))
    if x >= -2 * np.pi and x < -np.pi:
        return sin(x + 2 * np.pi)
    if x > np.pi and x <= 2 * np.pi:
        return sin(x - 2 * np.pi)
    return sin(x - (x // (2 * np.pi)) * 2 * np.pi)

xs = np.linspace(-20, 20, 200)
ys = [sin(x) for x in xs]

plt.xlim(-20, 20)
plt.ylim(-15, 15)
plt.plot(xs, ys)
plt.show()
