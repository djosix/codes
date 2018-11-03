import numpy as np
import tensorflow as tf


class DenseNetwork(object):
    def __init__(self, config):
        self.num_inputs = config['inputs']
        self.num_outputs = config['outputs']
        self.layers_config = config['layers']
        self.optimizer_config = config['optimizer']
        self._build()
    
    def _build(self):
        self.x = x = tf.placeholder(tf.float32, [None, 2])
        self.y = tf.placeholder(tf.float32, [None, 1])
        for units, activation in self.layers_config:
            x = tf.layers.dense(x, units, activation=activation)
        self.f = x
        self.loss = tf.losses.sigmoid_cross_entropy(
            self.y, self.f, reduction=tf.losses.Reduction.MEAN)
        Optimizer = getattr(tf.train, self.optimizer_config['name'])
        self.optimizer = Optimizer(**self.optimizer_config['config'])
        self.optimize = self.optimizer.minimize(self.loss)
        self.output = tf.nn.sigmoid(self.f)

    def predict(self, x):
        return self.output.eval({self.x: x})
    
    def fit(self, x, y, epochs=100, verbose=False):
        for _ in range(epochs):
            self.optimize.run({
                self.x: x,
                self.y: y
            })
            if verbose:
                print(self.loss.eval({
                    self.x: x,
                    self.y: y
                }))

    def error(self, x, y):
        return self.loss.eval({self.x: x, self.y: y})

if __name__ == '__main__':
    m = DenseNetwork({
        'inputs': 2,
        'outputs': 1,
        'layers': [
            (4, tf.nn.relu),
            (4, tf.nn.relu),
            (2, tf.nn.relu),
            (1, None) # sigmoid cross entropy
        ],
        'optimizer': {
            'name': 'AdamOptimizer',
            'config': {
                'learning_rate': 0.01
            }
        }
    })

    session = tf.InteractiveSession()
    session.run(tf.global_variables_initializer())

    trX = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]

    trY = [
        [0],
        [1],
        [1],
        [0]
    ]


    for i in range(100):
        m.fit(trX, trY, epochs=100)
        e = m.error(trX, trY)
        a = sum((m.predict(trX) > 0.5) == trY) / len(trY)
        print('epoch {}: error={} accuracy={}'.format(i * 100, e, a))
        if a == 1: break
