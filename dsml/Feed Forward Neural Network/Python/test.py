from nn import *

# XOR training set
X = [[0, 0], [1, 0], [0, 1], [1, 1]]
Y = [[0], [1], [1], [0]]

while True:
    nn = Network.deep(2, [
         10, 100, 100, 100, 100, 100, 10, 1
    ], eta=0.1, decay=1)

    errors = []
    for i in range(2000):
        E = []
        for x, y in zip(X, Y):
            p = nn.forward([x])

            pEpa = (p - y)
            E.append((pEpa ** 2).mean())
            nn.backward(pEpa)
        E = np.array(E).mean()
        print("epoch: {}, error: {}".format(i, E))
        errors.append(E)
        if E < 0.005:
            break

    for x, y in zip(X, Y):
        p = nn.forward([x])
        print(y, int(p > 0.5), p)
    
    import matplotlib.pyplot as plt, time
    fig, ax = plt.subplots(nrows=1, ncols=1)
    plt.ylim(0, 0.5)
    ax.plot(list(range(len(errors))), errors)
    fig.savefig('fig/%d.png' % int(time.time()))
    plt.close(fig)
