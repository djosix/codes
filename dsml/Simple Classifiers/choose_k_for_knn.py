import numpy as np
import matplotlib.pyplot as plt
import tqdm

from data import Dataset
from k_nearest_neighbors import KNearestNeighbors

dataset = Dataset()
dataset.split(ratio=[8, 2])

# Max k
I = 20

# How many time for cross-validation
N = 100

# 10 folds cross-validation
J = 10

accuracy = np.zeros([I, N, J])

for i in tqdm.tqdm(range(I)):
    k = i + 1
    for n in range(N):
        for j, fold in enumerate(dataset.k_folds(J)):
            (X_train, y_train), (X_valid, y_valid) = fold
            knn = KNearestNeighbors(dataset.F, dataset.C, k=k)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_valid)
            accuracy[i, n, j] = np.mean(y_valid == y_pred)

accuracy = accuracy.mean(-1).mean(-1)

plt.title('K and accuracy')
plt.ylabel('Accuracy')
plt.xlabel('K')
plt.xticks(np.arange(I) + 1)
plt.plot(np.arange(I) + 1, accuracy)
plt.savefig('figures/choosing_k_for_knn.png')

# Test
X_train, y_train = dataset.train
X_test, y_test = dataset.test
best_k = accuracy.argmax() + 1
knn = KNearestNeighbors(dataset.F, dataset.C, k=best_k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = np.mean(y_test == y_pred)
print('Best K:', best_k)
print('Testing accuracy:', accuracy)
