import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/tmp/MNIST', one_hot=True)


x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

h = tf.nn.softmax(x @ W + b)
y = tf.placeholder(tf.float32, [None, 10])

#cross_entropy = tf.reduce_sum(h * tf.log(y), reduction_indices=[1])
#loss = tf.reduce_mean(cross_entropy)

_ = tf.nn.softmax_cross_entropy_with_logits(logits=h, labels=y)

cross_entropy = tf.reduce_sum(_)
loss = tf.reduce_mean(cross_entropy)

learning_rate = 0.05
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_step = optimizer.minimize(loss)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(h, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

import matplotlib.pyplot as plt


accs = []
for _ in range(1000):
    batch_x, batch_y = mnist.train.next_batch(100)
    sess.run(train_step,
             {x: batch_x, y: batch_y})
    acc = sess.run(accuracy, {x: mnist.test.images, y: mnist.test.labels})
    accs += [acc]
    print(acc)

plt.title('Accuracy')
plt.plot(range(1000), accs)
