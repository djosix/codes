import numpy as np
import matplotlib.pyplot as plt

from data import Dataset
from base import Classifier


def simple_test(classifier, n=100):
    assert issubclass(classifier, Classifier)

    data = Dataset()
    clf = classifier(data.F, data.C)
    average_accuracy = 0

    for i in range(n):
        data.split() # shuffle and split
        X_train, y_train = data.train
        X_test, y_test = data.test

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_test = np.array(y_test)
        accuracy = (y_test == y_pred).mean()

        print('Test:')
        print(y_test)
        print('Predict:')
        print(y_pred)
        print('Accuracy:', accuracy)
        print()
        average_accuracy += accuracy

    average_accuracy /= n
    print('Average:', average_accuracy)
    print('Shuffled:', 'train={}, test={}, rounds={}'.format(
        len(X_train), len(X_test), n))

    return accuracy
