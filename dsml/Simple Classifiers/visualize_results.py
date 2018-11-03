import matplotlib.pyplot as plt
import numpy as np

from k_nearest_neighbors import KNearestNeighbors
from naive_bayes import NaiveBayes
from decision_tree import DecisionTree
from data import Dataset


dataset = Dataset()

# Use same data for training and testing to observe the
# decisions made by classifiers
dataset.split([1, 0])
X_train, y_train = dataset.train

# Use only 2 features
features = [0, 1]
classes = list(range(dataset.C))
n_features = len(features)
n_classes = dataset.C

# Simple dimension reduction
X_train = X_train[:, features]

# Because we reduced the dimension of X, we have to ensure
# that there is only one y for an X, or there will be some
# unclearness
X, y = [], []
for Xi, yi in zip(X_train, y_train):
    Xi, yi = Xi.tolist(), yi.tolist()
    if Xi not in X:
        X.append(Xi)
        y.append(yi)
X, y = np.array(X), np.array(y)


print('===================================================')
print('Training:', len(X_train))
print('Features:', features)
print('Classes:', classes)
print('===================================================')


# Draw decision regions
def visualize(title, clf, X, y, steps=100):
    plt.style.use('ggplot')
    plt.clf()
    plt.set_cmap('Pastel1')
    plt.title(title)
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    x_margin, y_margin = (x_max - x_min) / 10, (y_max - y_min) / 10
    x_min, x_max = x_min - x_margin, x_max + x_margin
    y_min, y_max = y_min - y_margin, y_max + y_margin
    xs = np.arange(x_min, x_max, (x_max - x_min) / steps)
    ys = np.arange(y_min, y_max, (y_max - y_min) / steps)
    xm, ym = np.meshgrid(xs, ys)
    y_ = clf.predict(np.c_[xm.ravel(), ym.ravel()])
    y_ = y_.reshape(xm.shape)
    plt.pcolormesh(xm, ym, y_, vmin=0, vmax=9)
    for c, color in zip(classes, ['red', 'blue', 'green']):
        plt.scatter(*X[y == c].T, c=color, linewidths=1, edgecolors='white')

#======================================================================
# K-Nearest Neighbors

for k in (np.arange(15) + 1):
    knn = KNearestNeighbors(n_features, n_classes, k=k)
    knn.fit(X, y)
    y_pred = knn.predict(X)
    accuracy = np.mean(y == y_pred)
    title = 'KNN: K=%d Accuracy=%.3f' % (k, accuracy)
    print(title)
    visualize(title, knn, X, y)
    plt.savefig('figures/knn_%d.png' % k)

#======================================================================
# Naive Bayes

nb = NaiveBayes(n_features, n_classes)
nb.fit(X, y)
y_pred = nb.predict(X)
accuracy = np.mean(y == y_pred)
title = 'Naive Bayes: Accuracy=%.3f' % (accuracy)
print(title)
visualize(title, nb, X, y)
plt.savefig('figures/nb.png')

#======================================================================
# Decision Tree

for splits in (np.arange(40) + 1):
    dt = DecisionTree(n_features, n_classes, splits=splits)
    dt.fit(X, y)
    from decision_tree import print_tree
    print_tree(dt._tree)
    y_pred = dt.predict(X)
    accuracy = np.mean(y == y_pred)
    title = 'Decision Tree: splits=%d Accuracy=%.3f' \
            % (splits, accuracy)
    print(title)
    visualize(title, dt, X, y)
    plt.savefig('figures/dt_%d.png' % splits)
