import numpy as np

class Dataset:

    def __init__(self):
        N = 0
        features, labels = [], []
        # Parse raw data
        for line in open('iris.txt').readlines():
            items = line.split(',')
            label = items.pop(-1).strip()
            feature = list(map(float, items))
            features.append(feature)
            labels.append(label)
            N += 1
        # Determine class names
        names = np.unique(labels).tolist()
        # Define attributes
        self.features = features
        self.labels = [names.index(label) for label in labels]
        self.label_names = names
        self.N = N                          # number of samples
        self.C = len(names)                 # number of classes
        self.X = np.array(self.features)    # inputs
        self.F = self.X.shape[1]            # number of features
        self.y = np.array(self.labels)      # outputs
        # Normalization
        mean = self.X.mean(0, keepdims=True)
        std = self.X.std(0, keepdims=True)
        self.X = (self.X - mean) / std
        # Split dataset to training set and test set
        self.split()

    def split(self, ratio=[7, 3]):
        # Split and shuffle dataset
        n_train = (ratio[0] * self.N) // sum(ratio)
        indexes = np.random.permutation(self.N)
        X, y = self.X[indexes], self.y[indexes]
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]
        # Make data available
        self.train = X_train, y_train
        self.test = X_test, y_test
        self.n_train = n_train
        self.n_test = n_train - self.N

    def k_folds(self, k=5):
        X, y = self.train
        # Ensure n % 5 = 0 and random permutation
        n = self.n_train - self.n_train % k
        i = np.random.permutation(self.n_train)
        X, y = X[i][:n], y[i][:n]
        # Generate training set and validation set
        n_fold = n // k
        for j in range(0, n, n_fold):
            X_train, y_train = X.tolist(), y.tolist()
            popped = [(X_train.pop(j), y_train.pop(j))
                      for _ in range(n_fold)]
            X_valid, y_valid = list(zip(*popped))
            X_train, y_train = np.array(X_train), np.array(y_train)
            X_valid, y_valid = np.array(X_valid), np.array(y_valid)
            yield (X_train, y_train), (X_valid, y_valid)
