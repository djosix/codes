import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import normalize

# select 2 classes
iris = load_iris()
trX = iris.data[iris.target < 2, :]
trY = iris.target[iris.target < 2].reshape(-1, 1)

# select 2 features and normalize
trX = trX[:, [0, 1]]
trX = normalize(trX, axis=0)

""" NOTE
data shapes
-----------
trX     m x 3   with bias
trY     m x 1
W       3 x 1
p       m x 1
l       1
g       3 x 1
"""

bias = np.ones((trX.shape[0], 1))
trX = np.append(trX, bias, axis=1)

W = np.random.normal(0, 1, [3, 1])

# learning rate
a = 0.6

_l = np.inf
i = 0
while True:
    # predict
    p = 1 / (1 + np.exp(-(trX @ W)))    # m x 1

    # calculate the gradient of log likelihood
    g = np.sum((trY - p) * trX, axis=0).reshape(-1, 1)

    # gradient ascent
    W += a * g

    # calculate log likelihood
    l = np.sum(trY * np.log(p) + (1 - trY) * np.log(1 - p))

    # convergence check
    if abs(_l - l) < 1e-3:
        break

    # save current log likelihood
    _l = l

    i += 1

print("log likelihood converged at", l)
print("epoch:", i)
print("accuracy:", np.sum((p > 0.5) == trY) / trY.shape[0])

# plot predictoin
p = (p > 0.5).reshape(-1)
x = trX[:, 0:2]
y = trY.reshape(-1)
plt.scatter(*x[(y==0)*(p==0)].T, color='blue', marker='$0$')
plt.scatter(*x[(y==0)*(p==1)].T, color='blue', marker='$1$')
plt.scatter(*x[(y==1)*(p==1)].T, color='green', marker='$1$')
plt.scatter(*x[(y==1)*(p==0)].T, color='green', marker='$0$')
plt.show()
