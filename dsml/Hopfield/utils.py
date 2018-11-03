import numpy as np

#=============================================================================
# Utilities

class Data:
    def __init__(self, path=None, shape=None, data=None):
        if path is not None and shape is not None:
            with open(path) as f:
                data, temp = [], []
                for line in f.readlines():
                    if len(temp) < shape[0]:
                        func = lambda c: 1 if c != ' ' else -1
                        line = list(map(func, line[:-1]))
                        line = line + [-1] * (shape[1] - len(line))
                        line = line[:shape[1]]
                        temp.append(line)
                    else:
                        data.append(np.array(temp).ravel())
                        temp.clear()
                if len(temp) == shape[0]:
                    data.append(np.array(temp).ravel())
            self.data = np.array(data)
            self.shape = shape
        elif data is not None and shape is not None:
            self.data = data
            self.shape = shape
    
    def show(self, n=0):
        import matplotlib.pyplot as plt
        plt.imshow(self.data[n].reshape(*self.shape))
        plt.show()
    
    def noise(self, p):
        data = np.array(self.data)
        for pattern in data:
            dim = len(pattern)
            i = np.random.choice(range(dim), int(p * dim), replace=False)
            pattern[i] *= -1
        return Data(data=data, shape=self.shape)
    
    def basic():
        return (
            Data(path='datasets/Basic_Training.txt', shape=(13, 9)),
            Data(path='datasets/Basic_Testing.txt', shape=(13, 9))
        )
    
    def bonus():
        return (
            Data(path='datasets/Bonus_Training.txt', shape=(10, 10)),
            Data(path='datasets/Bonus_Testing.txt', shape=(10, 10))
        )



def plot_all(train, test, result, shape, title=''):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(train.shape[0], 3)

    for i, data in enumerate(zip(train, test, result)):
        for j, x in enumerate(data):
            ax[i, j].imshow(x.reshape(*shape))
            ax[i, j].axis('off')

    fig.suptitle(title + ' (train|test|result)', size=16)
    plt.show()
    