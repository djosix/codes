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
    x, y = vector
    radius2 = x * x + y * y

    in_circle += (radius2 < 1).sum()
    count += batch_size

    if show:
        plt.scatter(*vector[:, radius2 < 1], color='black', marker='.')
        plt.pause(1e-8)
 
    print('esitmated pi =', 4 * in_circle / count)
