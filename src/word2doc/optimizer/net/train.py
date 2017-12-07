#!/usr/bin/env python

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from word2doc.optimizer import pre_process

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w_h, w_o):
    h = tf.nn.sigmoid(tf.matmul(X, w_h)) # this is a basic mlp, think 2 stacked logistic regressions
    return tf.matmul(h, w_o) # note that we dont take the softmax at the end because our cost fn does that for us


def load_data(path):
    squad = np.load(path)
    squad_dict = np.ndarray.tolist(squad)

    labels = list()
    data = list()
    for question, data_dict in squad_dict.items():
        label = data_dict['label']
        docs = data_dict['docs']
        doc_names = list()
        doc_scores = list()

        for name, scores in docs.items():
            doc_names.append(name)
            doc_scores += scores

        labels.append(label)
        doc_data = {
            'names': doc_names,
            'scores': doc_scores
        }

        data.append(doc_data)

    return labels, data


def train():

    trX, trY = load_data('data/squad/dev-v1.1/1-queries.npy')
    teX, teY = trX, trY

    # trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

    X = tf.placeholder("float", [None, 784])
    Y = tf.placeholder("float", [None, 10])

    w_h = init_weights([784, 625]) # create symbolic variables
    w_o = init_weights([625, 10])

    py_x = model(X, w_h, w_o)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y)) # compute costs
    train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct an optimizer
    predict_op = tf.argmax(py_x, 1)

    # Launch the graph in a session
    with tf.Session() as sess:
        # you need to initialize all variables
        tf.global_variables_initializer().run()

        for i in range(100):
            for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
                sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
            print(i, np.mean(np.argmax(teY, axis=1) ==
                             sess.run(predict_op, feed_dict={X: teX})))
