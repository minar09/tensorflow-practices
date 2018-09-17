import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

x_data = [[1, 1], [2, 2], [3, 3]]
y_data = [[1], [2], [3]]
w = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
x = tf.placeholder(tf.float32, shape=[None, 2])
y = tf.placeholder(tf.float32, shape=[None, 1])

hypothesis = tf.add(tf.matmul(x, w), b)

cost = tf.reduce_mean(tf.square(hypothesis - y))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

with tf.Session() as sess:

	sess.run(tf.global_variables_initializer())

	for step in range(2001):
		cost_val, w_val, b_val, _ = sess.run([cost, w, b, train], feed_dict={x: x_data, y: y_data})
		
		if step % 200 == 0:
			print(step, cost_val, w_val, b_val)
		