import numpy as np
import matplotlib.pyplot as plt
import itertools
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
a = 1e-2

# threshold
t = 0.98

# save loss over epochs
l = []

for i in itertools.count():
    """ NOTE
    This is still able to be seen as maximizing
    log likelihood, because of the identical update
    rules (the gradient of log likelihood), but
    it's hard to test its convergence, so I just
    set a threshold to break the loop.
    """
    
    # predict
    p = (trX @ W > 0).astype('int64')   # m x 1

    # update
    g = np.sum((trY - p) * trX, axis=0).reshape(-1, 1)
    W += a * g

    # precision
    r = np.sum(trY == p) / trY.shape[0]

    # track loss
    l += [1 - r]

    print("epoch: %d, preciseion, %f" % (i, r))

    if t <= r:
        break

plt.plot(np.arange(len(l)), l)
plt.show()
