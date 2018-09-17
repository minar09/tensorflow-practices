import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

w = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
x = tf.placeholder(tf.float32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])

hypothesis = x * w + b

cost = tf.reduce_mean(tf.square(hypothesis - y))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for step in range(2001):
	cost_val, w_val, b_val, _ = sess.run([cost, w, b, train], feed_dict={x: [1, 2, 3, 4, 5], y: [1, 2, 3, 4, 5]})
	
	if step % 200 == 0:
		print(step, cost_val, w_val, b_val)
		