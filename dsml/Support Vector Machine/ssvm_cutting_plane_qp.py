from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed as debug
import sys, os

# pylint: disable=E1101

def visualize(title, predict, X, y, steps=100, classes=[0, 1]):
    try: x_steps, y_steps = steps
    except: x_steps = y_steps = steps
    plt.clf()
    plt.title(title)
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    x_margin, y_margin = (x_max - x_min) / 10, (y_max - y_min) / 10
    x_min, x_max = x_min - x_margin, x_max + x_margin
    y_min, y_max = y_min - y_margin, y_max + y_margin
    xs = np.arange(x_min, x_max, (x_max - x_min) / x_steps)
    ys = np.arange(y_min, y_max, (y_max - y_min) / y_steps)
    xm, ym = np.meshgrid(xs, ys)
    y_ = predict(np.c_[xm.ravel(), ym.ravel()])
    y_ = y_.reshape(xm.shape)
    plt.pcolormesh(xm, ym, y_, vmin=0, vmax=9, cmap='Pastel1')
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'yellow',
              'brown', 'pink', 'gray'][:len(classes)]
    for class_, color in zip(classes, colors):
        plt.scatter(*X[y == class_].T, c=color, lw=1, edgecolors='white')


n_features = 2
n_classes = 2
n_samples = 100
classes = list(range(n_classes))

dim_w = n_classes * n_features

train_X, train_y = make_blobs(n_samples=n_samples, centers=n_classes)

w = np.zeros([dim_w])
w = np.random.random([dim_w])

_phi = lambda X, y: np.kron(X, [1 if y == i else 0 for i in classes])
_loss = lambda y, y_: float(y == y_)


phi = lambda i, j: _phi(train_X[i], train_y[j])
psi = lambda i, j: phi(i, i) - phi(i, j)
loss = lambda i, j: _loss(train_y[i], train_y[j])


inference = lambda X: np.argmax([np.dot(w, _phi(X, y)) for y in classes])
predict = lambda X: np.array([inference(Xi) for Xi in X])

#==========================================================================
# Quadratic Programming

def solve_qp(constraints, regularization=1):
    import cvxopt
    I, J = map(np.array, zip(*constraints))
    n_con = len(constraints)
    c_psi = np.array([psi(i, j) for i, j in constraints]) # [n_con, dim_w]
    P = (c_psi.reshape(n_con, 1, -1) * c_psi.reshape(1, n_con, -1)).sum(-1) ** 2
    P = cvxopt.matrix(P)
    q = np.array([loss(i, j) for i, j in constraints])
    q = cvxopt.matrix(q)
    I_uniq = np.unique(I)
    G1 = (I_uniq.reshape(-1, 1) == I.reshape(1, n_con)).astype(float)
    h1 = np.ones(I_uniq.size) # forall i: sum_j a_ij <= 1
    G2 = -np.eye(n_con)
    h2 = np.zeros(n_con) # forall i,j: -a_ij <= 0
    G = cvxopt.matrix(np.concatenate([G1, G2]))
    h = cvxopt.matrix(np.concatenate([h1, h2]))
    alpha = np.array(cvxopt.solvers.qp(P, q, G, h)['x']) # [n_con, 1]
    return (alpha * c_psi).sum(0) # w

#==========================================================================
# Cutting Plane Algorithm

max_iter = 100
tolerance = -np.inf
constraints = []

for n_iter in range(max_iter):
    
    constraint_changed = False

    # Update working sets
    for i in range(n_samples):
        violating_score = np.array([loss(i, j) - w.dot(phi(i, j))
                                    for j in range(n_samples)])
        violating_score[train_y == train_y[i]] = -np.inf
        max_score = violating_score.max()

        print(max_score)

        if max_score < tolerance:
            continue

        js = np.arange(violating_score.size)[violating_score == max_score]

        for j in np.random.permutation(js):
            if (i, j) not in constraints:
                constraints.append((i, j))
                constraint_changed = True
                break
        else:
            continue
        

    if not constraint_changed:
        print('No constraint changed')
        break
    
    # constraints = [
    #     (i, j)
    #     for i in range(n_samples)
    #     for j in range(n_samples)
    # ]

    w = solve_qp(constraints, regularization=0.3)

else:
    print('Optimization reached the max iteration')
    

#==========================================================================



visualize('test', predict, train_X, train_y, steps=100, classes=classes)
plt.show()
