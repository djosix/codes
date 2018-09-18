import numpy as np

def almost_ft(ys):
    fs = np.empty(ys.size, dtype=complex)

    for i in range(0, ys.size):
        angles = np.arange(ys.size) * -2 * i / ys.size
        vx = np.sum(ys * np.cos(angles))
        vy = np.sum(ys * np.sin(angles))
        fs[i] = vx + vy * 1j 

    return fs


def more_almost_ft(ys):
    fs = np.empty(ys.size, dtype=complex)

    for i in range(ys.size):
        angles = np.arange(ys.size) * -2 * i / ys.size
        vectors = np.exp(angles * 1j)
        centroid = np.mean(ys * vectors)
        fs[i] = centroid

    return fs


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    xs = np.linspace(0, 50, 200)
    ys = np.sin(xs)

    plt.clf()

    plt.subplot(3, 1, 1)
    plt.plot(xs, ys)

    plt.subplot(3, 1, 2)
    plt.plot(xs, np.abs(almost_ft(ys)))

    plt.subplot(3, 1, 3)
    plt.plot(xs, np.abs(more_almost_ft(ys)))

    plt.show()
