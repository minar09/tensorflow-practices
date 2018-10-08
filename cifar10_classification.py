
# Hide the warning messages about CPU/GPU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# import libraries
import tensorflow as tf
import numpy as np
from keras.datasets.cifar10 import load_data
from matplotlib import pyplot as plt


if __name__ == "__main__":

    # Load data and initialize variables
    (x_train, y_train), (x_test, y_test) = load_data()

    # one hot encoding
    y_train_onehot = np.eye(10)[y_train]
    y_test_onehot = np.eye(10)[y_test]
    #print(y_train_onehot.shape)

    y_train_onehot = np.squeeze(y_train_onehot)
    y_test_onehot = np.squeeze(y_test_onehot)
    #print(y_train_onehot)

    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    y = tf.placeholder(tf.float32, shape=[None, 10])

    # Network architecture
    x_image = x
    w_conv1 = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 32], stddev=0.01))
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
    h_conv1 = tf.nn.sigmoid(tf.nn.conv2d(x, w_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    w_conv2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 32, 64], stddev=0.01))
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
    h_conv2 = tf.nn.sigmoid(tf.nn.conv2d(h_pool1, w_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    h_conv2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 64])
    w_fc1 = tf.Variable(tf.truncated_normal(shape=[8 * 8 * 64, 384], stddev=0.01))
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[384]))
    h_fc1 = tf.nn.sigmoid(tf.matmul(h_conv2_flat, w_fc1) + b_fc1)

    w_fc2 = tf.Variable(tf.truncated_normal(shape=[384, 10], stddev=0.01))
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
    logits = tf.matmul(h_fc1, w_fc2) + b_fc2
    y_pred = tf.nn.softmax(logits)

    # Functions
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits))
    optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Tensorboard
    accuracy_history = tf.placeholder(tf.float32)
    accuracy_history_summary = tf.summary.scalar('accuracy_history', accuracy_history)
    merged_history = tf.summary.merge_all()



    # Run the network
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        writer_train = tf.summary.FileWriter('./board/hist/train',sess.graph)
        writer_test = tf.summary.FileWriter('./board/hist/test',sess.graph)
        writer_loss = tf.summary.FileWriter('./board/hist/loss',sess.graph)

        total_epoch = 10
        for e in range(total_epoch):

            total_size = x_train.shape[0]
            batch_size = 128

            loss_list = []
            train_accuracy_list = []

            for i in range(int(total_size / batch_size)):

                # batch load
                batch_x = x_train[i*batch_size:(i+1)*batch_size]
                batch_y = y_train_onehot[i*batch_size:(i+1)*batch_size]

                # train
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

                # logging
                train_accuracy = accuracy.eval(feed_dict={x: batch_x, y: batch_y})
                loss_print = loss.eval(feed_dict={x: batch_x, y: batch_y})
                train_accuracy_list.append(train_accuracy)
                loss_list.append(loss_print)

                writer_train.add_summary(sess.run(merged_history, feed_dict={accuracy_history: np.mean(train_accuracy_list)}),e)
                writer_loss.add_summary(sess.run(merged_history, feed_dict={accuracy_history: np.mean(loss_list)}),e)

            test_total_size = x_test.shape[0]
            test_batch_size = 128

            test_accuracy_list = []
            for i in range(int(test_total_size / test_batch_size)):
                # test batch load
                test_batch_x = x_test[i*test_batch_size:(i+1)*test_batch_size]
                test_batch_y = y_test_onehot[i*test_batch_size:(i+1)*test_batch_size]

                # logging
                test_accuracy = accuracy.eval(feed_dict={x: test_batch_x, y: test_batch_y})
                test_accuracy_list.append(test_accuracy)

                writer_test.add_summary(sess.run(merged_history, feed_dict={accuracy_history: np.mean(test_accuracy_list)}),e)

            print("Epoch:", e, "Train accuracy:", np.mean(train_accuracy_list)*100, "Loss:",np.mean(loss_list), "Test accuracy:", np.mean(test_accuracy_list)*100)





