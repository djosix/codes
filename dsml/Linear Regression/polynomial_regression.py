import matplotlib.pyplot as plt
import numpy as np

""" NOTE
data shapes
-----------
trX:    m x 1
trY:    m x 1
X:      m x n
w:      n x 1
p:      m x 1
powers: 1 x n
"""

# dimension
n = 6
powers = np.arange(n).reshape(1, -1)

# training set
trX = np.linspace(0, 300, 10).reshape(-1, 1)
trY = 10 * np.sin(trX / 150) + np.random.normal(10, 1, trX.shape)

# normalization is very important
trX = (trX - np.mean(trX, axis=0)) / np.std(trX, axis=0)
trY = (trY - np.mean(trY, axis=0)) / np.std(trY, axis=0)

# make polynomial
X = trX ** powers

# parameters
w = np.random.normal(0, 1, [1, n])

# learning rate
a = 5e-2

_e = np.inf
while True:
    
    # compute error
    p = np.sum(w * X, axis=1).reshape(-1, 1) # m x 1
    d = p - trY         # m x 1
    e = np.mean(d ** 2) # 1

    print("Error:", e)

    # convergence test
    if abs(_e - e) < 1e-5:
        break

    # gradient
    pLpw = np.mean(d * X, axis=0, keepdims=True)

    # gradient descent
    w -= a * pLpw

    # keep current errer
    _e = e


# plot model
plt.scatter(trX, trY)
plot_x = np.linspace(np.min(trX), np.max(trX), 50).reshape(-1, 1)
plot_y = np.sum(w * (plot_x ** powers), axis=1)
plt.plot(plot_x, plot_y)
plt.show()
