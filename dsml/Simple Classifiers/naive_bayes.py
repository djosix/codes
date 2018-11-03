'''
Implementation of Gaussian Naïve Bayes classifier.
by Yuankui Lee (104502526) [toregenerate@gmail.com] 2018.03.17
'''

import numpy as np
from base import Classifier


class NaiveBayes(Classifier):
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs)
        self.epsilon = kwargs.get('epsilon', 1e-5)

    def fit(self, X, y):
        X, y = self._check_X_y(X, y)
        # Check that all classes are provided
        assert np.in1d(np.arange(self.n_classes), y).all()
        # Compute class priors
        _, y_counts = np.unique(y, return_counts=True)
        self._log_prior = y_counts / len(y)
        # Compute Gaussian parameters
        self._mean_mat = np.zeros([self.n_classes, self.n_features])
        self._var_mat = np.zeros([self.n_classes, self.n_features])
        for c in range(self.n_classes):
            Xc = X[y == c] # [n_samples_in_c, n_features]
            self._mean_mat[c, :] = Xc.mean(0) # [n_features]
            self._var_mat[c, :] = Xc.var(0)   # [n_features]
        # Increase variance to address 0 variance problem
        self._var_mat += self.epsilon
        # Set trained to ensure predicting after training
        self._trained = True

    def predict(self, X):
        assert self._trained
        X = self._check_X(X)
        jll = self._joint_log_likelihood(X) # [n_samples, n_classes]
        log_prior = np.expand_dims(self._log_prior, 0) # [1, n_classes]
        y_pred = (jll + log_prior).argmax(1) # [n_samples]
        return y_pred

    def _joint_log_likelihood(self, X):
        # X : shape [n_samples, n_features]
        X = np.expand_dims(X, -1) # [n_samples, n_features, 1]
        mean = np.expand_dims(self._mean_mat.T, 0) # [1, n_features, n_classes]
        var = np.expand_dims(self._var_mat.T, 0) # [1, n_features, n_classes]
        ll = -1/2 * (np.log(2 * np.pi * var) + (X - mean) ** 2 / var)
        jll = ll.sum(1) # [n_samples, n_classes]
        return jll


if __name__ == '__main__':
    from util import simple_test
    simple_test(NaiveBayes)
