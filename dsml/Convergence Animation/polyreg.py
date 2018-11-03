import tensorflow as tf
import numpy as np
from seaborn import plt

trX = np.linspace(-3, 3, 10)
trY = np.sin(trX / 2) + np.random.normal(0, 0.1, trX.shape)

# normalization
trX = (trX - np.mean(trX)) / np.std(trX)
trY = (trY - np.mean(trY)) / np.std(trY)

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
b = tf.Variable(tf.random_normal([1]))

h = b + 0
for i in range(1, 5):
    w = tf.Variable(tf.random_normal([1]))
    h += tf.multiply((x ** i), w)

loss = tf.reduce_sum(tf.pow(h - y, 2)) / 9

adam = tf.train.AdamOptimizer(0.01).minimize(loss)


i = 0

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    lp = 0
    while True:
        sess.run(adam, feed_dict={x: trX, y: trY})
        lc = sess.run(loss, feed_dict={x: trX, y: trY})
        # print(lc)
        if np.abs(lp - lc) < 1e-5:
            break
        lp = lc
        

        i += 1
        if i % 200 != 0:
            continue

        fig, ax = plt.subplots(1, 1)
        ax.set_ylim([-2, 2])
        ax.scatter(trX, trY, color='red')
        ax.plot(trX, sess.run(h, feed_dict={x: trX}), color='green')
        fig.show()
        plt.show()

