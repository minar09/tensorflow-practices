#### Softmax classifier ###


# Hide the warning messages about CPU/GPU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import libraries
import tensorflow as tf

# Define training data
x_data = [[1, 2, 1, 1], [2, 1, 3, 2], [3, 1, 3, 4], [4, 1, 5, 5], [1, 7, 5, 5], [1, 2, 5, 6], [1, 6, 6, 6], [1, 7, 7, 7]]
y_data = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]

# Define placeholder
x = tf.placeholder("float", shape=[None, 4])
y = tf.placeholder("float", shape=[None, 3])

nb_classes = 3

# Define Weight and Bias for linear regression
w = tf.Variable(tf.random_normal([4, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# Define hypothesis, cost and optimization functions
hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)
cost = tf.reduce_mean( - tf.reduce_sum(y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Launch graph/Define session 
with tf.Session() as sess:

	sess.run(tf.global_variables_initializer())

	for step in range(2001):
		sess.run(optimizer, feed_dict={x: x_data, y: y_data})
	
		if step % 400 == 0:
			print(step, sess.run(cost, feed_dict={x: x_data, y: y_data}))
			
	# Testing and One-hot encoding
	a = sess.run(hypothesis, feed_dict={x: [[1, 11, 7, 9]]})
	print(a, sess.run(tf.argmax(a, 1)))
	
	b = sess.run(hypothesis, feed_dict={x: [[1, 3, 4, 3]]})
	print(b, sess.run(tf.argmax(b, 1)))
	
	c = sess.run(hypothesis, feed_dict={x: [[1, 1, 0, 1]]})
	print(c, sess.run(tf.argmax(c, 1)))
	
	all = sess.run(hypothesis, feed_dict={x: [[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1]]})
	print(all, sess.run(tf.argmax(all, 1)))
	
	
		