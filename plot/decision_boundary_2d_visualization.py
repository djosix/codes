import matplotlib.pyplot as plt
import numpy as np

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

