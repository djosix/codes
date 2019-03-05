import numpy as np

# pylint: disable=E1101


show = False
batch_size = 1024

import matplotlib.pyplot as plt

if show:
    plt.ion()

in_circle = 0
count = 0

while True:
    vector = np.random.uniform(size=[2, batch_size])
    radius2 = np.sum(vector ** 2)

    in_circle += np.sum(radius2 < 1)
    count += batch_size

    if show:
        plt.scatter(*vector[:, radius2 < 1], color='black', marker='.')
        plt.pause(1e-8)
 
    print('estimated pi =', 4 * in_circle / count)
