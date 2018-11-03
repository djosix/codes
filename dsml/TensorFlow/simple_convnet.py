import numpy as np
import tensorflow as tf

class ConvNet(object):
    def __init__(self):
        self.x = x = tf.placeholder(tf.float32, shape=[None, 784])
        self.y = y = tf.placeholder(tf.float32, shape=[None, 10])
        self.keep_prob = tf.placeholder(tf.float32)
        x = tf.reshape(x, [-1, 28, 28, 1])      # to 2D
        x = tf.nn.relu(self._conv(x, 5, 16, 2)) # conv1
        x = tf.nn.relu(self._conv(x, 5, 16, 2)) # conv2
        x = tf.reshape(x, [-1, 7 * 7 * 16])     # flatten
        x = tf.nn.relu(self._dense(x, 1024))    # fc1
        x = tf.nn.dropout(x, self.keep_prob)    # dropout
        logits = self._dense(x, 10)   # fc2
        self.h = h = tf.nn.softmax(logits)
        # self.err = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=h))
        self.err = tf.losses.softmax_cross_entropy(y, logits, reduction=tf.losses.Reduction.MEAN)
        # self.err = -tf.reduce_sum(y * tf.log(h))
        optimizer = tf.train.AdamOptimizer()
        self.opt = optimizer.minimize(self.err)

    def _conv(self, x, filter_size, n_layers, stride=2):
        w = tf.Variable(tf.random_normal([
            filter_size,
            filter_size,
            int(x.shape[-1]),
            n_layers
        ], 0, 0.01))
        b = tf.Variable(tf.random_normal([n_layers], 0, 0.01))
        conv = tf.nn.conv2d(input=x,
                            filter=w,
                            strides=[1, stride, stride, 1],
                            padding='SAME')
        return conv + b

    def _dense(self, x, n_units):
        n_dim = int(x.shape[-1])
        w = tf.Variable(tf.random_normal([n_dim, n_units], 0, 0.01))
        b = tf.Variable(tf.random_normal([n_units], 0, 0.01))
        return x @ w + b

    def fit(self, x, y, keep_prob=0.5):
        self.opt.run({
            self.x: x,
            self.y: y,
            self.keep_prob: keep_prob
        })

    def predict(self, x, keep_prob=1.):
        return self.h.eval({
            self.x: x,
            self.keep_prob: keep_prob
        })

    def accuracy(self, x, y, keep_prob=1.):
        h = self.h.eval({
            self.x: x,
            self.y: y,
            self.keep_prob: keep_prob
        })
        corr = sum(np.argmax(h, axis=1) == np.argmax(y, axis=1))
        return corr / len(h)

    def error(self, x, y, keep_prob=1.):
        return self.err.eval({
            self.x: x,
            self.y: y,
            self.keep_prob: keep_prob
        })

if __name__ == '__main__':
    import tensorflow.examples.tutorials.mnist.input_data as input_data
    mnist = input_data.read_data_sets('/tmp/MNIST', one_hot=True)

    sess = tf.InteractiveSession()

    m = ConvNet()
    sess.run(tf.global_variables_initializer())

    for i in range(10000):
        for j in range(mnist.train.num_examples // 100):
            trX, trY = mnist.train.next_batch(100, shuffle=False)
            m.fit(trX, trY)
        trX, trY = mnist.train.images, mnist.train.labels
        vaX, vaY = mnist.validation.images, mnist.validation.labels
        trE = m.error(trX, trY)
        trA = m.accuracy(trX, trY)
        vaE = m.error(vaX, vaY)
        vaA = m.accuracy(vaX, vaY)
        print('epoch %d: trE=%.5f trA=%.5f vaE=%.5f vaA=%.5f'
              % (i, trE, trA, vaE, vaA))

