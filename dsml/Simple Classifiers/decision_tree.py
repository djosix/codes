'''
Implementation of CART Decision Tree algorithm (numeric inputs).
by Yuankui Lee (104502526) [toregenerate@gmail.com] 2018.03.19
'''

import numpy as np
from base import Classifier

class DecisionTree(Classifier):
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs)
        self.splits = kwargs.get('splits', 10)

    def fit(self, X, y):
        X, y = self._check_X_y(X, y)
        # Compute class probability for unclear separations
        counts = [len(y == c) for c in range(self.n_classes)]
        self._class_probs = np.array(counts) / len(y)
        # Tree: feature, threshold, left, right
        # Impure: data (X, y), impurity
        # Pure: prediction (class)
        # Start with an impure leaf as the tree root which contains
        # the whole training set
        self._tree = self._create_impure_leaf(X, y)
        self._impure_leaves = [self._tree]
        # Splitting a tree: turning an impure leaf into a subtree
        # of impure leaf or pure leaf (contains only class label)
        self._splits = 0
        while self._impure_leaves and self._splits < self.splits:
            self._split_tree()
        # Make classifier deterministic
        self._finalize_leaves()
        # Delete intermediate variables
        del self._class_probs
        del self._impure_leaves
        # Set trained
        self._trained = True

    def predict(self, X):
        assert self._trained
        X = self._check_X(X)
        self._inputs = X
        self._predictions = np.empty(X.shape[0], dtype=int)
        # Run decision tree -> self._predictions
        indexes = np.arange(len(X))
        self._find_predictions(indexes, self._tree)
        predictions = self._predictions
        # Delete intermediate variables
        del self._inputs
        del self._predictions
        return predictions

    def _gini_impurity(self, y):
        # Gini impurity = 1 - sum_c p(c)^2
        unique, counts = np.unique(y, return_counts=True)
        if len(unique) < 2:
            return 0.0
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)

    def _split_tree(self):
        if not self._impure_leaves:
            # Terminate when all leaves are pure
            return False
        # Find the most impure leaf (max impurity)
        index = np.argmax([l.impurity for l in self._impure_leaves])
        leaf = self._impure_leaves.pop(index)
        X, y = leaf.data
        if leaf.impurity == 0.0:
            # No need to split because it's all pure
            leaf.init(_Node.PURE)
            leaf.prediction = y[0]
            return
        # Find the best feature and threshold by calculating the
        # information gain
        feature, threshold = None, None
        impurity = np.inf
        for f in range(self.n_features):
            Xf = X[:, f]
            # Use average of every two unique consecutive samples
            # as the threshold
            unique = np.unique(Xf) # sorted unique feature values
            thresholds = (unique[1:] + unique[:-1]) / 2
            for t in thresholds:
                y_left, y_right = y[Xf < t], y[Xf >= t]
                i = len(y_left) * self._gini_impurity(y_left)
                i += len(y_right) * self._gini_impurity(y_right)
                i /= len(y)
                if i < impurity:
                    feature = f
                    threshold = t
                    impurity = i
        if threshold is None:
            # No threshold could be choosen
            self._finalize_leaf(leaf, y[0])
            return
        Xf = X[:, feature]
        # Turn the impure leaf into a subtree
        leaf.init(_Node.TREE)
        leaf.feature = feature
        leaf.threshold = threshold
        # Split
        left = (Xf < threshold)
        right = (Xf >= threshold)
        leaf.left = self._create_impure_leaf(X[left], y[left])
        leaf.right = self._create_impure_leaf(X[right], y[right])
        if leaf.left.impurity == 0.0:
            self._finalize_leaf(leaf.left, leaf.left.data[1][0])
        else:
            self._impure_leaves.append(leaf.left)
        if leaf.right.impurity == 0.0:
            self._finalize_leaf(leaf.right, leaf.right.data[1][0])
        else:
            self._impure_leaves.append(leaf.right)
        self._splits += 1
    
    def _create_impure_leaf(self, X, y):
        leaf = _Node(_Node.IMPURE)
        leaf.data = X, y
        leaf.impurity = self._gini_impurity(y)
        return leaf

    def _finalize_leaves(self):
        # Because impure leaves hold bunches of data, we must turn all
        # leaves into pure leaves, which make deterministic predictions.
        for leaf in self._impure_leaves:
            self._finalize_leaf(leaf)

    def _finalize_leaf(self, leaf, prediction=None):
        assert leaf.type is _Node.IMPURE
        if prediction is not None:
            leaf.init(_Node.PURE)
            leaf.prediction = prediction
            return
        (X, y) = leaf.data
        # Try to find the most classes
        classes, counts = np.unique(y, return_counts=True)
        most_mask = counts == counts.max()
        most_count = most_mask.sum()
        most_classes = classes[most_mask]
        leaf.init(_Node.PURED)
        if most_count > 1:
            # Choose a class by its class probability
            class_probs = [self._class_probs[c] for c in most_classes]
            leaf.prediction = most_classes[np.argmax(class_probs)]
        else:
            leaf.prediction = most_classes[0]

    def _find_predictions(self, i, node):
        if node.type is _Node.PURE or node.type is _Node.PURED:
            self._predictions[i] = node.prediction
        elif node.type is _Node.TREE:
            # Searching in subtrees
            left = self._inputs[i, node.feature] < node.threshold
            if left.any():
                self._find_predictions(i[left], node.left)
            right = self._inputs[i, node.feature] >= node.threshold
            if right.any():
                self._find_predictions(i[right], node.right)
        else:
            assert False


class _Node:
    TREE, IMPURE, PURE, PURED = 0, 1, 2, 3

    def __init__(self, type_, **kwargs):
        self.init(type_, **kwargs)
    
    def init(self, type_, **kwargs):
        self.type = type_
        # Tree
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        # Impure leaf
        self.data = None
        self.impurity = None
        # Pure leaf
        self.prediction = None
        # Set given attributes
        for key, value in kwargs.items():
            setattr(self, key, value)


def print_tree(node, indent=0):
    from termcolor import colored
    print('    ' * indent, end='') # indent
    if node.type is _Node.IMPURE:
        attr = colored('[Impure]', 'red', attrs=['bold'])
        print('{} impurity={}, datasize={}'.format(attr,
                                                   node.impurity,
                                                   len(node.data[0])))
    elif node.type is _Node.PURED:
        attr = colored('[Pured]', 'red', attrs=['bold'])
        print('{} prediction={}'.format(attr, node.prediction))
    elif node.type is _Node.PURE:
        attr = colored('[Pure]', 'yellow', attrs=['bold'])
        print('{} prediction={}'.format(attr, node.prediction))
    elif node.type is _Node.TREE:
        attr = colored('[Tree]', 'green', attrs=['bold'])
        print('{} feature={} threshold={}'.format(attr,
                                                  node.feature,
                                                  node.threshold))
        print_tree(node.left, indent + 1)
        print_tree(node.right, indent + 1)


if __name__ == '__main__':
    from util import simple_test
    simple_test(DecisionTree, 10)
