# tensor: constant, Variable, placeholder

import tensorflow as tf

node1 = tf.constant(3, tf.float32)
node2 = tf.constant(4, tf.float32)
print(node1, node2)

sess = tf.Session()
print(sess.run([node1, node2]))

#node3 = node1 + node2
node3 = tf.add(node1, node2)
print(node3)

print(sess.run(node3))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b
print(a, b, adder_node)
print(sess.run(adder_node, {a: 10, b: 20}))
print(sess.run(adder_node, {a: [1, 2], b: [-1, -2]}))
print(sess.run(adder_node,
               {a: [[1, 2], [3, 4]], b: [[-1, -2], [-3, -4]]}))
