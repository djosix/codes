from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt


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
n_classes = 3
classes = list(range(n_classes))

w_dim = n_classes * n_features

train_X, train_y = make_blobs(n_samples=40, centers=n_classes)

w = np.zeros([w_dim])

phi = lambda X, y: np.kron(X, [1 if y == i else 0 for i in classes])
phi_Xy = lambda: np.array([phi(X, y) for X, y in zip(train_X, train_y)])
w_phi_Xy = lambda: phi_Xy() @ w.reshape(2, 1)

inference = lambda X: np.argmax([np.dot(w, phi(X, y)) for y in classes])
predict = lambda X: np.array([inference(Xi) for Xi in X])

#==========================================================================
# Structured Perceptron

for k in range(100):
    changed = False

    for i, (X, y_hat) in enumerate(zip(train_X, train_y)):
        y_pred = np.argmax([np.dot(w, phi(X, y)) for y in classes])

        if y_pred != y_hat:
            w = w + (phi(X, y_hat) - phi(X, y_pred))
            print('w:', w)
            changed = True

    if not changed:
        break

else:
    print('Probably not separable')

#==========================================================================

visualize('test', predict, train_X, train_y, steps=100, classes=classes)
plt.show()
