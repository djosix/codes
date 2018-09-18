import numpy as np


def dft(ys):
    ks, ns = np.mgrid[:ys.size, :ys.size]
    x = -2 * np.pi * 1j * ks * ns / ys.size
    fs = np.sum(ys * np.exp(x), 1)
    return fs


def dift(fs):
    ks, ns = np.mgrid[:fs.size, :fs.size]
    x = 2 * np.pi * 1j * ks * ns / fs.size
    ys = np.mean(fs * np.exp(x), 1)
    return ys


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    xs = np.linspace(0, 50, 1000)
    ys = np.sin(xs)

    plt.clf()

    plt.subplot(3, 1, 1)
    plt.plot(xs, ys)

    fs = dft(ys)

    plt.subplot(3, 1, 2)
    plt.plot(xs, fs)

    ifs = dift(fs)

    plt.subplot(3, 1, 3)
    plt.plot(xs, ifs)

    plt.show()
