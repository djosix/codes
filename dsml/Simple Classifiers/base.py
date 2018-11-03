import numpy as np
from abc import ABCMeta, abstractmethod

class Classifier(metaclass=ABCMeta):
    def __init__(self, n_features, n_classes):
        self.n_features = n_features
        self.n_classes = n_classes
        self._trained = False

    @abstractmethod
    def fit(self):
        return NotImplemented

    @abstractmethod
    def predict(self):
        return NotImplemented
    
    def _check_X(self, X):
        X = np.array(X)
        assert X.dtype == float
        assert len(X.shape) == 2
        assert X.shape[1] == self.n_features
        return X

    def _check_y(self, y):
        y = np.array(y)
        assert y.dtype == int
        assert len(y.shape) == 1
        assert np.all(y >= 0)
        assert np.all(y < self.n_classes)
        return y

    def _check_X_y(self, X, y):
        X = self._check_X(X)
        y = self._check_y(y)
        assert X.shape[0] == y.shape[0]
        return X, y
