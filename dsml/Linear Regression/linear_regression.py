import numpy as np
import matplotlib.pyplot as plt


""" NOTE
data shapes
-----------
trX:    m
trY:    m
w:      1
b:      1
p:      1
"""

# training set
trX = np.linspace(0, 300, 10)
trY = 10 * np.sin(trX / 150) + np.random.normal(10, 1, trX.shape)

# normalization is very important
trX = (trX - np.mean(trX)) / np.std(trX)
trY = (trY - np.mean(trY)) / np.std(trY)
print(trX.shape, trY.shape)

# parameters
w = np.random.normal(0, 1, 1)
b = np.random.normal(0, 1, 1)
print(w.shape, b.shape)

# learning rate
a = 1e-1

_e = np.inf
while True:
    
    # compute error
    p = w * trX + b     # m
    d = p - trY         # m
    e = np.mean(d ** 2) # 1

    print("Error:", e)

    # convergence test
    if abs(_e - e) < 1e-5:
        break

    # partial derivatives
    pLpw = np.mean(d * trX)
    pLpb = np.mean(d)

    # gradient descent
    w -= a * pLpw
    b -= a * pLpb

    # keep current errer
    _e = e


# plot model
lim = np.array([np.min(trX), np.max(trX)])
plt.scatter(trX, trY)
plt.plot(lim, w * lim + b)
plt.show()
