from lazy_game import Game, GameStart, pg
import tensorflow as tf
import numpy as np

#=====================================================
# training set
n_dim = 4
n_x = 20
error = 0.3
# optimization
learning_rate = 1
optimizer = tf.train.AdamOptimizer
# AdagradOptimizer AdadeltaOptimizer GradientDescentOptimizer
# AdamOptimizer RMSPropOptimizer FtrlOptimizer
#=====================================================

def dot(g, pos, r=1):
    pg.draw.ellipse(g, (100, 100, 100),
                    [pos[0]-r, pos[1]-r, r*2, r*2])

def line(g, a, b):
    pg.draw.line(g, (0,0,0), a, b)

trX = np.linspace(-3, 3, n_x)
trY = np.sin(trX) + np.random.normal(0, error, trX.shape)
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
b = tf.Variable(tf.random_normal([1]))
h = b + 0
for i in range(1, n_dim):
    w = tf.Variable(tf.random_normal([1]))
    h += tf.multiply((x ** i), w)
loss = tf.reduce_sum(tf.pow(h - y, 2)) / 9

opt = optimizer(learning_rate).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

def draw(g):
    sess.run(opt, feed_dict={x: trX, y: trY})
    H = sess.run(h, feed_dict={x: trX})
    p = (0, 0)
    for i, (xx, yy, hh) in enumerate(zip(trX, trY, H)):
        d = ((xx + 3) * 640 / 6, yy * 240 / 3 + 240)
        dot(g, d, 6)
        c = ((xx + 3) * 640 / 6, hh * 240 / 3 + 240)
        if i > 0:
            line(g, p, c)
        p = c

GameStart(name="Test", draw=draw)