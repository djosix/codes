'''
Implementation of K Nearest Neighbors.
by Yuankui Lee (104502526) [toregenerate@gmail.com] 2018.03.17
'''

import numpy as np
from base import Classifier


class KNearestNeighbors(Classifier):
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs)
        k = kwargs.get('k', 5)
        assert k > 0
        self.k = k

    def fit(self, X, y):
        X, y = self._check_X_y(X, y)
        self._X = X # [n_train, n_features]
        self._y = y # [n_train]
        self._trained = True
    
    def predict(self, X):
        assert self._trained
        X = self._check_X(X)
        n_inputs = X.shape[0]
        X = np.expand_dims(X, -1) # [n_inputs, n_features, 1]
        _X = np.expand_dims(self._X.T, 0) # [1, n_features, n_train]
        # Compute distances between inputs and training data
        distance = np.sum((X - _X) ** 2, axis=1) # [n_inputs, n_train]
        # Sort distances and get nearest k labels
        indexes = distance.argsort(1)[:, :self.k] # [n_inputs, k]
        ky = self._y[indexes] # [n_inputs, k]
        # Find most occurances
        knn = np.empty([n_inputs], dtype=int)
        for i, ys in enumerate(ky):
            unique, counts = np.unique(ys, return_counts=True)
            knn[i] = unique[counts.argmax()]
        return np.array(knn)


if __name__ == '__main__':
    from util import simple_test
    simple_test(KNearestNeighbors)
