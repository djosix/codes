import numpy as np
import matplotlib.pyplot as plt


# training set
trX = np.linspace(0, 6, 10) # m
trY = np.sin(trX) + np.random.normal(0, 0.1, trX.shape)

# normalization
trX = (trX - np.mean(trX)) / np.std(trX)
trY = (trY - np.mean(trY)) / np.std(trY)

def local_weighted_regression(x, t=0.4):
    """ Locally weighted regression
    1. Non-parametric algorithm
    2. Need to use training set when predicting

    data shapes
    -----------
    l - local weights (m)
    w - parameter (1)
    b - parameter (1)
    t - bandwidth parameter (1)
    """

    w, b = np.random.normal(0, 1, 2)

    # compute local weights
    l = np.exp(-(trX - x) ** 2 / (2 * (t ** 2)))

    # learning rate
    a = 2

    # minimize loss
    _e = np.inf
    while True:
        
        # compute error
        p = w * trX + b # m
        d = p - trY     # m
        e = np.mean(l * d ** 2)

        # divergence test
        if e == np.inf:
            print('diverged')
            w, b = np.random.normal(0, 1, 2)
            a /= 1.5
            continue
        
        # convergence test
        if abs(_e - e) < 1e-6:
            print('converged at %f' % e)
            break

        w -= a * np.mean(l * (d * trX))
        b -= a * np.mean(l * d)

        # keep current errer
        _e = e
    
    return w * x + b



# plot model
plt.scatter(trX, trY)
plot_x = np.linspace(np.min(trX), np.max(trX), 30)
plot_y = []
for i, x in enumerate(plot_x):
    plot_y += [local_weighted_regression(x)]
plt.plot(plot_x, plot_y)
plt.show()
