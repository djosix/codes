import tensorflow as tf
import numpy as np



def build_autoencoder(dims=[784, 512, 256, 64]):
    W = []
    for n_inputs, n_units in zip(dims[:-1], dims[1:]):
        W += [tf.Variable(tf.random_uniform(
            [n_inputs, n_units],
            -1. / np.sqrt(n_inputs),
            1. / np.sqrt(n_inputs) ))]
    
    def encoder(x):
        for w, n_units in zip(W, dims[1:]):
            b = tf.Variable(tf.zeros([n_units]))
            x = tf.nn.tanh(x @ w + b)
        return x
    
    def decoder(x):
        for w, n_units in zip(W[::-1], dims[-2::-1]):
            b = tf.Variable(tf.zeros([n_units]))
            x = tf.nn.tanh(x @ tf.transpose(w) + b)
        return x
    
    def autoencoder(x):
        return decoder(encoder(x))

    return autoencoder, encoder, decoder



if __name__ == '__main__':
    import tensorflow.examples.tutorials.mnist.input_data as input_data
    mnist = input_data.read_data_sets('/tmp/MNIST', one_hot=True)

    session = tf.InteractiveSession()

    autoencoder, encoder, decoder = build_autoencoder([784, 128, 64])

    x = tf.placeholder(tf.float32, [None, 784])
    _x = autoencoder(x)

    adam = tf.train.AdamOptimizer()
    loss = tf.reduce_sum((_x - x) ** 2)
    optimize = adam.minimize(loss)

    session.run(tf.global_variables_initializer())

    for i in range(100):
        l = loss.eval({x: mnist.train.images})
        print('epoch %d: loss=%f' % (i, l))
        for _ in range(mnist.train.num_examples // 100):
            trX, _ = mnist.train.next_batch(100)
            optimize.run({x: trX})
