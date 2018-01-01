#!/usr/bin/env python

from __future__ import print_function

import tensorflow as tf
import numpy as np
import random

from tqdm import tqdm
from word2doc.util import constants
from word2doc.util import logger


class OptimizerNet:

    def __init__(self):
        self.logger = logger.get_logger()

        # Parameters
        self.learning_rate = 0.001
        self.training_epochs = 15
        self.batch_size = 20
        self.display_step = 1

        # Network Parameters
        self.n_hidden_1 = 100   # 1st layer number of neurons
        self.n_hidden_2 = 1000  # 2nd layer number of neurons
        self.n_input = 20     # 20 values from pre-processing
        self.n_classes = 5    # one of 5 classes

        # tf Graph input
        self.X = tf.placeholder(tf.float32, [None, self.n_input])
        self.Y = tf.placeholder(tf.int32, [None, self.n_classes])

        # Store layers weight & bias
        self.weights = {
            'h1': tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2])),
            'out': tf.Variable(tf.random_normal([self.n_hidden_2, self.n_classes]))
        }
        self.biases = {
            'b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([self.n_hidden_2])),
            'out': tf.Variable(tf.random_normal([self.n_classes]))
        }

    def load_data(self, path):
        self.logger.info("Load data..")

        squad = np.load(path)
        squad_dict = np.ndarray.tolist(squad)

        labels = list()
        scores = list()
        with tqdm(total=len(squad_dict)) as pbar:
            for question, data_dict in tqdm(squad_dict.items()):
                label = data_dict['label']
                docs = data_dict['docs']
                doc_scores = list()

                counter = 0
                doc_id = -1
                for name, scores_local in docs.items():
                    if name == label:
                        doc_id = counter
                    else:
                        counter += 1

                    doc_scores += [float(s) for s in scores_local]

                if not doc_id == -1:
                    # Make sure all score arrays have 5 docs
                    while not len(doc_scores) == 20:
                        doc_scores += [-1000.0, -1000.0, -1000.0, -1000.0]

                    # 1-hot encode
                    one_hot = [0, 0, 0, 0, 0]
                    one_hot[doc_id] = 1

                    # Add to arrays
                    labels.append(one_hot)
                    scores.append(doc_scores)

                pbar.update()

        return scores, labels

    def scramble_data(self, x, y):

        shuffled = list(zip(x, y))

        random.shuffle(shuffled)

        scrambled = []
        for tuple in shuffled:
            train = list(map(lambda b: np.ndarray.tolist(b), np.array_split(tuple[0], 5)))
            tuple_zip = list(zip(train, tuple[1]))
            random.shuffle(tuple_zip)
            tuple_zip = list(zip(*tuple_zip))
            tuple_zip[0] = [item for sublist in tuple_zip[0] for item in sublist]
            tuple_zip[1] = list(tuple_zip[1])
            scrambled.append(tuple_zip)

        x, y = zip(*scrambled)

        return x, y

    def multilayer_perceptron(self, x):
        """Create model"""

        # Hidden fully connected layer with 20 neurons
        layer_1 = tf.add(tf.matmul(x, self.weights['h1']), self.biases['b1'])

        # Hidden fully connected layer with 20 neurons
        layer_2 = tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2'])

        # Output fully connected layer with a neuron for each class
        out_layer = tf.matmul(layer_2, self.weights['out']) + self.biases['out']

        return out_layer

    def train(self):
        # Construct model
        logits = self.multilayer_perceptron(self.X)

        # Load data
        train_x, train_y = self.load_data(constants.get_squad_train_queries_path())
        test_x, test_y = self.load_data(constants.get_squad_dev_queries_path())

        # Scramble data
        train_x, train_y = self.scramble_data(train_x, train_y)

        # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_op = optimizer.minimize(loss_op)

        # Initializing the variables
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            # Training cycle
            for epoch in range(self.training_epochs):
                avg_cost = 0.
                total_batch = int(len(train_x) / self.batch_size)

                # Loop over all batches
                with tqdm(total=total_batch) as pbar:
                    for i in tqdm(range(total_batch)):
                        batch_x = train_x[self.batch_size * i:self.batch_size * (i + 1)]

                        batch_y = train_y[self.batch_size * i:self.batch_size * (i + 1)]

                        # Run optimization op (backprop) and cost op (to get loss value)
                        _, c = sess.run([train_op, loss_op], feed_dict={self.X: batch_x, self.Y: batch_y})

                        # Compute average loss
                        avg_cost += c / total_batch

                        pbar.update()

                # Display logs per epoch step
                if epoch % self.display_step == 0:
                    print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))

            print("Optimization Finished!")

            # Test model
            pred = tf.nn.softmax(logits)  # Apply softmax to logits
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(self.Y, 1))

            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print("Accuracy:", accuracy.eval({self.X: test_x, self.Y: test_y}))
