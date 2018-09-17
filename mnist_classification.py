#### MNIST classification ###


# Hide the warning messages about CPU/GPU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import libraries
import tensorflow as tf
import time
import numpy as np

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

# Download/Read MNIST
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Hide the warning messages about deprecations of MNIST data read
tf.logging.set_verbosity(old_v)

# Initialize parameters
t1 = time.time()
num_steps = 5000
batch_size = 128
display_step = 500

n_hidden_1 = 256
n_hidden_2 = 256
n_hidden_3 = 256
num_input = 784
num_classes = 10

# Define placeholder
x = tf.placeholder("float", [None, num_input])
y = tf.placeholder("float", [None, num_classes])

# Define Weight and Bias for linear regression

weights = {
	'h1' : tf.Variable(tf.random_normal([num_input, n_hidden_1])),
	'h2' : tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
	'h3' : tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
	'out' : tf.Variable(tf.random_normal([n_hidden_1, num_classes]))
}

biases = {
	'b1' : tf.Variable(tf.random_normal([n_hidden_1])),
	'b2' : tf.Variable(tf.random_normal([n_hidden_2])),
	'b3' : tf.Variable(tf.random_normal([n_hidden_3])),
	'out' : tf.Variable(tf.random_normal([num_classes]))
}

# Initialize the model
def mlp(x):
	l1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
	l2 = tf.nn.relu(tf.add(tf.matmul(l1, weights['h2']), biases['b2']))
	l3 = tf.nn.relu(tf.add(tf.matmul(l2, weights['h3']), biases['b3']))
	lout = tf.add(tf.matmul(l3, weights['out']), biases['out'])
	return lout

# Define hypothesis, cost and optimization functions
logits = mlp(x)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

prediction = tf.nn.softmax(logits)
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Launch graph/Initialize session 
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	
	for step in range(1, num_steps+1):
		batch_train_images, batch_train_labels = mnist.train.next_batch(batch_size)
		sess.run(optimizer, feed_dict={x: batch_train_images, y: batch_train_labels})
	
		if step % display_step == 0 or step == 1:
			print("Step " + str(step) + " out of " + str(num_steps))
			
	print("Optimization finished!")
	t2 = time.time()
	print("Testing accuracy: ", sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})*100, "%")

print("Learning time: " + str(t2-t1) + " seconds")
