import os
import random
import time

import keras
import numpy as np
import prettytable
import tensorflow as tf
from keras import optimizers
import keras.backend as K

from keras.layers.core import Dense, Activation
from word2doc.optimizer.net.APrioriLayer import APrioriLayer
from keras.models import Sequential

from word2doc.optimizer.tensorboard import LoggableTensorBoard
from word2doc.util import constants
from word2doc.util import logger


class Word2Doc:
    def __init__(self):
        self.logger = logger.get_logger()

        self.hyper_params = {
            'TIME': '',
            'TRAINING PARAMS': '',
            'loss_func': 'categorical_crossentropy',
            'optimizer': 'sgd',
            'epochs': 1,
            'batch_size_train': 8,
            'batch_size_test': 8,
            'n_input': 4096,
            '': '',
            'LEARNING RATE': '',
            ' ': '',
            'HIDDEN LAYER 1': '',
            'n_h1': 300,
            'h1_activation': 'relu',
            '  ': '',
            'OUTPUT LAYER': '',
            'n_classes': 5316192,
            'out_activation': 'softmax',
        }

    def log_hyper_params(self, id):
        self.hyper_params['TIME'] = id
        table = prettytable.PrettyTable(['Hyper Parameter', 'Value'])

        for key, val in self.hyper_params.items():
            table.add_row([key, val])

        self.logger.info(table)

    def create_run_log(self, id):
        path = os.path.join(constants.get_tensorboard_path(), id)
        os.makedirs(path)
        return path

    def sampled_softmax(weights, biases, y_true, y_pred, num_sampled, num_classes):
        return K.sampled_softmax(weights, biases, y_true, y_pred, num_sampled, num_classes)

    def load_data(self, path):
        self.logger.info('Load ' + path)

        data = np.load(path)

        labels = list()
        titles = dict()
        embeddings = list()
        context = list()
        for data_dict in data:

            doc_id = data_dict['doc_index']
            titles[doc_id] = data_dict['doc_title']
            ctx = data_dict['doc_window']

            e = data_dict['pivot_embeddings']

            for emb in e:
                labels.append(doc_id)
                embeddings.append(emb)

                neg_ctx = self.negative_samples(data, ctx, 1)
                context.append(neg_ctx)

        return labels, embeddings, context, titles

    def negative_samples(self, data, doc_context, num):
        negative_samples = []

        while len(negative_samples) < num:
            sample = random.choice(data)

            if sample['doc_index'] not in doc_context:
                negative_samples.append(sample['pivot_embeddings'][0])

        return negative_samples

    def shuffle_data(self, x, y, c):
        shuffled = list(zip(x, y, c))
        random.shuffle(shuffled)

        x, y, c = zip(*shuffled)

        return x, y, c

    def normalize_data(self, x, num_docs):
        norm_data = []

        # Go through every input for net
        for elem in np.asarray(x):

            # Dissect input into values for each document, so we get an array
            train = list(map(lambda b: np.ndarray.tolist(b), np.array_split(elem, num_docs)))

            # Normalize columns in array
            for j in range(0, len(train[0])):
                norm_list = list()

                # Extract column into norm_list
                for i in range(0, len(train)):
                    norm_list.append(train[i][j])

                # Normalize
                norm_list = np.asarray(norm_list)
                std = norm_list.std()

                # Std. is 0 -> all values are the same, so they will all be set to 0 (because of subtraction of mean)
                if std < 0.00001:
                    norm_list = [float(0)] * len(train)
                else:
                    norm_list = (norm_list - norm_list.mean()) / std

                # Put column back together
                for i in range(0, len(train)):
                    train[i][j] = norm_list[i]

            # Flatten list again and add to general result
            norm_data.append([item for sublist in train for item in sublist])

        return norm_data

    def model(self):
        start_time = time.time()
        self.logger.info('Compiling Model ... ')

        model = Sequential()

        # Hidden layer 1
        model.add(Dense(self.hyper_params['n_h1'], input_dim=self.hyper_params['n_input']))
        model.add(Activation(self.hyper_params['h1_activation']))

        # Output layer
        model.add(Dense(self.hyper_params['n_classes']))
        model.add(Activation(self.hyper_params['out_activation']))

        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.logger.info('Model compiled in {0} seconds'.format(time.time() - start_time))
        return model

    def train(self):
        # Load data
        labels, embeddings, context, titles = self.load_data(os.path.join(constants.get_word2doc_dir(), '1-wpp.npy'))
        # train_x, train_y = self.load_data(os.path.join(constants.get_word2doc_dir(), '1-wpp.npy'))

        # Shuffle data
        self.logger.info('Shuffling data..')
        embeddings, labels, context = self.shuffle_data(embeddings, labels, context)
        self.logger.info('Done shuffling data.')

        # Set up model
        model = self.model()
        model.summary()

        # Set up tensorboard
        id = str(int(round(time.time())))
        tbCallback = LoggableTensorBoard(log_dir=self.create_run_log(id),
                                         histogram_freq=0,
                                         write_graph=True,
                                         write_images=True,
                                         custom_log_func=None)

        # Train model
        self.logger.info('Training model with hyper params:')
        self.log_hyper_params(id)

        embeddings = np.asarray(embeddings)
        labels = np.array(labels)

        model.fit(embeddings, labels,
                  epochs=self.hyper_params['epochs'],
                  batch_size=self.hyper_params['batch_size_train'],
                  verbose=2,
                  callbacks=[tbCallback])

        # score = model.evaluate(test_x, test_y, batch_size=self.hyper_params['batch_size_test'])

        #self.logger.info('Score: [loss, accuracy]: {0}'.format(score))

    def custom_log_func(self, tensorboard, epoch, logs=None):

        # Add learning rate
        optimizer = tensorboard.model.optimizer
        lr = keras.backend.eval(tf.cast(optimizer.lr, tf.float64) * (
            1.0 / (1.0 + tf.cast(optimizer.iterations, tf.float64) * tf.cast(optimizer.decay, tf.float64))))

        return {
            "learning_rate": lr
        }

