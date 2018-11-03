import numpy as np



class Activation():
    def __init__(self, f):
        self.f = f

    def __call__(self, x):
        return self.f(x, False)

    def d(self, x):
        return self.f(x, True)

    def __repr__(self):
        return self.f.__name__


@Activation
def Linear(x, d=False):
    if not d:
        return x
    return 1

@Activation
def ReLU(x, d=False):
    if not d:
        x = np.array(x)
        x[x < 0] = 0
        return x
    return (x > 0).astype(np.float32)

@Activation
def Sigmoid(x, d=False):
    if not d:
        return 1 / (1 + np.exp(-x))
    return Sigmoid(x) * (1 - Sigmoid(x))

@Activation
def Swish(x, d):
    temp = np.exp(-x)
    if not d:
        return x / (1 + temp)
    return x * temp / (1 + temp) ** 2 + 1 / (1 + temp)




class Network():
    def __init__(self, dim_in, **kargs):
        self.dim_in = dim_in
        self.config = kargs
        self.layers = []

    def forward(self, a):
        a = np.array(a, dtype=np.float32)
        assert len(a.shape) == 2
        assert a.shape[1] == self.layers[0].dim_in
        for layer in self.layers:
            a = layer.forward(a)
        return a

    def backward(self, pEpa):
        pEpa = np.array(pEpa, dtype=np.float32)
        assert len(pEpa.shape) == 2
        assert pEpa.shape[1] == self.layers[-1].size
        for layer in self.layers[::-1]:
            pEpa = layer.backward(pEpa)
    
    def push(self, layer, **kwargs):
        config = self.config
        if not self.layers:
            config['dim_in'] = self.dim_in
        else:
            config['dim_in'] = self.layers[-1].size
        config.update(kwargs)
        layer.init(config)
        self.layers.append(layer)
    
    def reset():
        for layer in self.layers:
            layer.init()

    def __repr__(self):
        rows = ['    ' + repr(layer) for layer in self.layers]
        rows.insert(0, '    ' + repr(self.config))
        return 'Network(\n{}\n)'.format(',\n'.join(rows))

    def deep(dim_in, layer_sizes, eta=0.01, decay=0.999):
        nn = Network(
            dim_in,
            activation = Swish,
            eta = eta,
            decay = decay
        )
        for size in layer_sizes:
            nn.push(Layer(size))
        nn.layers[-1].activation = Sigmoid
        return nn
    
    def shallow(dim_in, layer_sizes, eta=0.01, decay=0.999):
        nn = Network(
            dim_in,
            activation = Sigmoid,
            eta = eta,
            decay = decay
        )
        for size in layer_sizes:
            nn.push(Layer(size))
        return nn



class Layer(object):
    def __init__(self, size, **kwargs):
        self.config = {
            'size': size,
            'dim_in': None,
            'bias': 1,
            'activation': ReLU,
            'eta': 0.01,
            'decay': 0.999
        }
        self.use_config(kwargs)
    
    def forward(self, i):
        bias = np.ones([i.shape[0], 1]) * self.bias
        self.x = np.append(bias, i, axis=1)
        self.z = self.x @ self.w
        return self.activation(self.z)

    def backward(self, pEpa):
        papz = self.activation.d(self.z)
        pEpz = pEpa * papz
        pzpw = self.x.T
        pEpw = pzpw @ pEpz
        self.update(pEpw)
        return (pEpz @ self.w.T)[:, 1:]
    
    def update(self, grad):
        self.G += (grad ** 2).sum()
        self.w -= self.r * grad / np.sqrt(self.G)
        self.r *= self.decay

    def init(self, config):
        self.use_config(config)
        assert self.dim_in is not None
        self.w = np.random.rand(self.dim_in + 1, self.size) - 0.5
        self.r = self.eta
        self.G = 0.0001 # adagrad

    def use_config(self, config={}):
        for key in self.config:
            if key in config:
                self.config[key] = config[key]
            setattr(self, key, self.config[key])

    def __repr__(self):
        return 'Layer{}'.format({
            'size': self.size,
            'dim_in': self.dim_in,
            'bias': self.bias,
            'activation': self.activation,
            'eta': self.eta,
            'decay': self.decay
        })