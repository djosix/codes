import tensorflow as tf
import numpy as np


def simple_rnn(input_dim, output_dim, state_dim, batch_size, seq_len):
    w1 = tf.Variable(tf.random_normal(
        [input_dim + state_dim, state_dim], 0, 0.001))
    b1 = tf.Variable(tf.zeros([1, state_dim]))
    w2 = tf.Variable(tf.random_normal(
        [state_dim, output_dim], 0, 0.001))
    b1 = tf.Variable(tf.zeros([1, output_dim]))

    def rnn_func(x, state):
        x_state = tf.concat([x, state], 1)
        return tf.nn.tanh(x_state @ w1 + b1)

    def rnn_over_time(x, state):
        tf.reshape(state, [1, state_dim])
        state = tf.tile(state, [batch_size, 1])
        outputs = []
        for i in range(seq_len):
            state = rnn_func(x[:, i, :], state)
            outputs += [state @ w2 + b1]
        return outputs, state

    return rnn_over_time


if __name__ == '__main__':
    x = tf.placeholder(tf.float32, [None, 10, 1])
    init_state = tf.placeholder(tf.float32, [1, 2])

    outputs, final_state = simple_rnn(1, 1, 2, 2000, 10)(x, init_state)

    losses = [tf.losses.sigmoid_cross_entropy(x[:, i, :], outputs[i])
              for i in range(10)]
    loss = tf.reduce_mean(losses)
    optimize = tf.train.AdamOptimizer().minimize(loss)

    corrects = [tf.cast(tf.equal(x[:, i, :] > 0.5, outputs[i] > 0), tf.float32)
                for i in range(10)]
    accuracy = tf.reduce_mean(corrects)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # make a batch of size 2000
    X = np.random.choice(2, [20, 10, 1]).astype(np.float32)
    X = np.tile(X, [100, 1, 1])
    
    for i in range(1000):
        l = loss.eval({x: X, init_state: [[0, 0]]})
        a = accuracy.eval({x: X, init_state: [[0, 0]]})
        print('iter %d: loss=%.5f accuracy=%.5f' % (i, l, a))
        optimize.run({x: X, init_state: [[0, 0]]})
        if a > .999999:
            break
    
