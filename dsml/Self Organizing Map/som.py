import numpy as np

#=============================================================================
# Utilities

class Dataset(object):
    def __init__(self, path):
        with open(path) as f:
            self.lines = f.readlines()
        self.x, self.y = [], []
        for line in self.lines:
            *x, y = line.split()
            x, y = list(map(float, x)), int(y)
            self.x.append(x)
            self.y.append(y)
        self.x = np.array(self.x, dtype=float)
        self.x = (self.x - self.x.mean(0)) / self.x.var(0)
        self.y = np.array(self.y, dtype=int)
        self.n = len(self.x[0])
        self.c = len(set(self.y))
    
    def epoch(self, random=False):
        data = list(zip(self.x, self.y))
        yield from np.random.permutation(data) if random else data


def nearest(p, m):
    min_dis, min_arg = np.Inf, (-1, -1) 
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            d = p - m[i, j]
            dis = sum(d * d)
            if dis < min_dis:
                min_dis = dis
                min_arg = (i, j)
    return m[min_arg]


#=============================================================================
# SOM

class SOM:
    def __init__(self, config):
        print('SOM config:', config)

        size, limit = config['size'], config['limit']
        self.mat = np.stack(np.mgrid[:size[0], :size[1]], 2).astype(float) \
                    * (limit[1] - limit[0]) / max(size) + limit[0]
        self.dataset = Dataset(config['dataset'])
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.tou1, self.tou2 = config['tou']
        self.sigma0 = config['sigma']
        self.eta0 = config['eta']
        self.t = 0

        if self.batch_size < 0 or self.batch_size > len(self.dataset.x):
            raise Exception('Invalid batch size, expect in range (0, {})'.format(len(self.dataset.x)))
    

    def step(self):
        m = len(self.dataset.x)
        
        if self.t < self.epochs * m:
            print('\repoch %d, t = %d' % (self.t // m, self.t), end='', flush=True)

            if self.batch_size == 0:
                self._update(self.dataset.x)
                self.t += m

            else:
                indeces = np.random.choice(m, size=self.batch_size, replace=False)
                xs = self.dataset.x[indeces]
                self._update(xs)
                self.t += self.batch_size
    
    
    def _update(self, X):
        sigma = self.sigma0 * np.exp(- self.t / self.tou1)
        eta = self.eta0 * np.exp(- self.t / self.tou2)

        cal_d = lambda x: nearest(x, self.mat) - self.mat
        cal_pi = lambda d: np.exp(- np.sum(d ** 2, 2) / (2 * sigma ** 2))
        cal_delta = lambda a: eta * np.expand_dims(a[0], 2) * (a[1] - self.mat)

        self.mat += sum(map(cal_delta, zip(map(cal_pi, map(cal_d, X)), X)))
