#!/usr/bin/env python

from __future__ import print_function

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


class OptimizerNet:

    def __init__(self):
        # Parameters
        self.learning_rate = 0.001
        self.training_epochs = 15
        self.batch_size = 50
        self.display_step = 1

        # Network Parameters
        self.n_hidden_1 = 20 # 1st layer number of neurons
        self.n_hidden_2 = 20 # 2nd layer number of neurons
        self.n_input = 20 # MNIST data input (img shape: 28*28)
        self.n_classes = 5 # MNIST total classes (0-9 digits)

        # tf Graph input
        self.X = tf.placeholder("float", [None, self.n_input])
        self.Y = tf.placeholder("float", [None, self.n_classes])

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
        squad = np.load(path)
        squad_dict = np.ndarray.tolist(squad)

        labels = list()
        scores = list()
        for question, data_dict in squad_dict.items():
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
                labels.append(doc_id)
                scores.append(doc_scores)

        return scores, labels

    def create_tensors(self):
        x, y = self.load_data('data/squad/dev-v1.1/1-queries.npy')

        # Create score tensor
        scores = tf.data.Dataset.from_tensor_slices(x)
        scores_batch = scores.batch(self.batch_size)
        scores_iterator = scores_batch.make_one_shot_iterator()

        return scores_iterator, y

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
        train_x, train_y = self.load_data('data/squad/dev-v1.1/1-queries.npy')

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
                for i in range(total_batch):
                    batch_x = train_x.get_next()
                    batch_y = None

                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c = sess.run([train_op, loss_op], feed_dict={self.X: batch_x, self.Y: batch_y})

                    # Compute average loss
                    avg_cost += c / total_batch

                # Display logs per epoch step
                if epoch % self.display_step == 0:
                    print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
            print("Optimization Finished!")

            # Test model
            pred = tf.nn.softmax(logits)  # Apply softmax to logits
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(self.Y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print("Accuracy:", accuracy.eval({self.X: mnist.test.images, self.Y: mnist.test.labels}))
