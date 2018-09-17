import tensorflow as tf

node1 = tf.constant(3)
node2 = tf.constant(2)
node3 = tf.add(node1, node2)

sess = tf.Session()

print("Sum : ", sess.run(node3))

sess.close()