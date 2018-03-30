#!usb/bin/python
# -*- coding:utf-8 -*-
"""

Description:
    A simple demo of classification using Tensorflow

Dependencies:
    tensorflow: 1.1.0
    matplotlib
    numpy

Date:
    2018-03-29
"""

__author__ = 'Alex Cheen'

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# region overall settings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
np.random.seed(1)
tf.set_random_seed(1)
# endregion

# region data generation

metal_data = np.ones((100, 2))   # shape (100, 2)
data_0 = np.random.normal(2*metal_data, 1)
data_1 = np.random.normal(-2*metal_data, 1)

data = np.vstack((data_0, data_1))

label = np.array([0]*100+[1]*100)

org_plt = plt.subplot(121)
org_plt.set_title('original data')
cls_plt = plt.subplot(122)


org_plt.scatter(data[:, 0], data[:, 1], c=label, s=15, cmap='RdYlGn')
# plt.show()

# endregion

# region NN structure define
input_data = tf.placeholder(tf.float32, data.shape)  # shape (200,2)
input_label = tf.placeholder(tf.int32, label.shape)   # shape (200,1)

# define structure
hidden_layer = tf.layers.dense(input_data, 4, tf.nn.relu)
output = tf.layers.dense(hidden_layer, 2)

# loss function
loss = tf.losses.sparse_softmax_cross_entropy(
    labels=input_label, logits=output)
opt = tf.train.GradientDescentOptimizer(learning_rate=0.02)

accuracy = tf.metrics.accuracy(
    labels=tf.squeeze(input_label),
    predictions=tf.argmax(output, axis=1),)[1]

train_op = opt.minimize(loss)
init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

# endregion

# region NN training and visualized progress
with tf.Session() as sess:
    sess.run(init_op)

    plt.ion()

    for step in range(500):
        _, acc, pred = sess.run([train_op, accuracy, output],
                                {input_data: data, input_label: label})
        if step % 5 == 0:

            cls_plt.cla()
            cls_plt.set_title('tf classification')
            cls_plt.scatter(data[:, 0], data[:, 1], c=pred.argmax(1),
                            s=20, lw=0, cmap='RdYlGn')
            cls_plt.text(-1.5, -4, 'Accuracy=%.2f%%' % (acc*100), fontdict={'size': 20, 'color': 'red'})

            plt.pause(0.5)
    plt.ioff()

plt.show()

# endregion

