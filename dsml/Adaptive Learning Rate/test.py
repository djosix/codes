import numpy as np
import matplotlib.pyplot as plt
from itertools import count

def linear_regression(optimizer, max_iter=np.inf, plot=True):
    # training set
    trX = np.linspace(0, 300, 10)
    trY = 10 * np.sin(trX / 150) + np.random.normal(10, 1, trX.shape)

    # normalization is very important
    trX = (trX - np.mean(trX)) / np.std(trX)
    trY = (trY - np.mean(trY)) / np.std(trY)
    print(trX.shape, trY.shape)

    # parameters
    w = np.random.normal(0, 1, 1)
    b = np.random.normal(0, 1, 1)
    print(w.shape, b.shape)

    E = []
    for i in count():
        
        # compute error
        p = w * trX + b     # m
        d = p - trY         # m
        e = np.mean(d ** 2) # 1

        print("Epoch {}, Error: {}".format(i, e))

        # convergence test
        if i > max_iter or (E and abs(E[-1] - e) < 1e-5):
            break

        # partial derivatives
        pLpw = np.mean(d * trX)
        pLpb = np.mean(d)

        # gradient descent
        w -= optimizer(pLpw)
        b -= optimizer(pLpb)

        # keep current errer
        E += [e]

    if plot:
        # plot model
        lim = np.array([np.min(trX), np.max(trX)])
        plt.subplot(1, 2, 1)
        plt.title('Result')
        plt.scatter(trX, trY)
        plt.plot(lim, w * lim + b)
        plt.subplot(1, 2, 2)
        plt.title('Error')
        plt.plot(range(len(E)), E)
        plt.show()

    return E



if __name__ == '__main__':
    from adagrad  import Adagrad
    from adam     import Adam
    from rmsprop  import RMSProp
    from adadelta import Adadelta

    def SGD(eta=0.01):
        return lambda g: eta * g

    def TimeDecaySGD(eta=0.01):
        def func(g, t=[0], eta=eta):
            eta /= np.sqrt(t[0] + 1)
            t[0] += 1
            return eta * g
        return func

    max_iter = 500
    
    E0 = linear_regression(SGD(),          max_iter=max_iter, plot=False)
    E1 = linear_regression(Adagrad(),      max_iter=max_iter, plot=False)
    E2 = linear_regression(Adam(),         max_iter=max_iter, plot=False)
    E3 = linear_regression(RMSProp(),      max_iter=max_iter, plot=False)
    E4 = linear_regression(Adadelta(),     max_iter=max_iter, plot=False)
    E5 = linear_regression(TimeDecaySGD(), max_iter=max_iter, plot=False)

    plt.subplot(2, 3, 1)
    plt.title('SGD')
    plt.plot(range(len(E0)), E0)
    
    plt.subplot(2, 3, 2)
    plt.title('Adagrad')
    plt.plot(range(len(E1)), E1)
    
    plt.subplot(2, 3, 3)
    plt.title('Adam')
    plt.plot(range(len(E2)), E2)

    plt.subplot(2, 3, 4)
    plt.title('RMSProp')
    plt.plot(range(len(E3)), E3)
    
    plt.subplot(2, 3, 5)
    plt.title('Adadelta')
    plt.plot(range(len(E4)), E4)
    
    plt.subplot(2, 3, 6)
    plt.title('TimeDecaySGD')
    plt.plot(range(len(E5)), E5)
    
    plt.show()
