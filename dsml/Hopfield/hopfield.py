import numpy as np

#=============================================================================
# Hopfield

class Hopfield:
    def __init__(self):
        pass


    def fit(self, data):
        assert len(data.shape) == 2
        m, n = data.shape
        self.w = (data.T @ data - m * np.eye(n)) / n
        self.t = self.w.sum(0)


    def predict(self, x, max_iter=100):
        assert len(x.shape) == 2 and x.shape[-1] == self.w.shape[0]
        x = np.array(x)
        active = list(range(x.shape[0]))
        for i in range(max_iter):
            if len(active) == 0:
                break
            print('iter {}: active: {}'.format(i, active))
            temp = np.array(x)
            x[active] = np.sign(x[active] @ self.w - self.t)
            for a, p, q in zip(active, x[active], temp[active]):
                if all(p == q):
                    active.remove(a)

        return x
