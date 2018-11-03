import numpy as np
import matplotlib.pyplot as plt
from itertools import count
from sklearn.datasets import load_iris
from sklearn.preprocessing import normalize

# select all classes
iris = load_iris()
c = len(set(iris.target))   # n classes
m = iris.target.shape[0]    # n examples
n = iris.data.shape[1]
trY = iris.target
trX = normalize(iris.data, axis=0)

""" NOTE
In CS229, we maximized the log likelihood of multinomial
distribution with the aspect of probability, here I minimize
the cross-entropy error instead, in fact, they are the same
thing.

variables
---------
c: number of classes
n: number of features
m: number of examples

data shapes
----------
trX     m x n
trY     m x 1
X       m x (n+1)
Y       m x c
W       (n+1) x c
"""

# append intercept term
_ = np.ones((m, 1))
X = np.append(trX, _, axis=1)

# one hot encoding
_ = np.repeat(np.arange(c).reshape(1, -1), m, axis=0)
Y = (_ == iris.target.reshape(-1, 1)).astype('int64')

# initialize weights
W = np.random.normal(size=(n+1, c))

def softmax(z):
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)

def cross_entropy_error(h, y):
    s = np.sum(h * y, axis=1)
    return -np.mean(np.log(s))

def cross_entropy_gradient(h, y, x):
    return x.T @ (h - y) / m    # (n+1) x c

# learning rate
a = 5

E, P = [], []
_e = np.inf
for i in count():
    # hypotheses over classes
    h = softmax(X @ W)  # m x c

    # compute the gradient of cross-entropy error
    g = cross_entropy_gradient(h, Y, X) # (n+1) x c

    # cross-entropy error itself
    e = cross_entropy_error(h, Y)

    # precision
    p = np.mean(np.argmax(h, axis=1) == trY)

    # gradient descent
    W -= a * g

    print("epoch: %d, error: %f, precision: %f" % (i, e, p))

    # convergence test
    if abs(_e - e) < 5e-5:
        break
    
    # save current error
    _e = e
    E += [e]
    P += [p]

# plot
plt.subplot(1, 2, 1)
plt.title('Cross-entropy error')
plt.plot(range(len(E)), E)
plt.subplot(1, 2, 2)
plt.title('Precision')
plt.plot(range(len(P)), P)
plt.show()
