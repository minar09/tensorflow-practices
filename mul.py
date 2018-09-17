import tensorflow as tf

node1 = tf.constant(3, dtype=tf.int32)
node2 = tf.constant(2, dtype=tf.int32)
node3 = tf.multiply(node1, node2)

with tf.Session() as sess:
	print("Product : ", sess.run(node3))

