import os
import random
import time

import keras
import numpy as np
import prettytable
import tensorflow as tf
from keras import optimizers
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from tqdm import tqdm

from word2doc.optimizer.tensorboard import XTensorBoard
from word2doc.util import constants
from word2doc.util import logger


class OptimizerNet:
    def __init__(self):
        self.logger = logger.get_logger()

        self.hyper_params = {
            'TIME': '',
            'TRAINING PARAMS': '',
            'loss_func': 'categorical_crossentropy',
            'optimizer': 'sgd',
            'epochs': 100,
            'batch_size_train': 8,
            'batch_size_test': 8,
            'n_input': 20,
            '': '',
            'LEARNING RATE': '',
            '          ': '',
            'HIDDEN LAYER 1': '',
            'n_h1': 1300,
            'h1_activation': 'relu',
            'h1_dropout': 0.4,
            ' ': '',
            'HIDDEN LAYER 2': '',
            'n_h2': 1100,
            'h2_activation': 'relu',
            'h2_dropout': 0.4,
            '  ': '',
            '   ': '',
            'HIDDEN LAYER 3': '',
            'n_h3': 900,
            'h3_activation': 'relu',
            'h3_dropout': 0.4,
            '    ': '',
            '     ': '',
            'HIDDEN LAYER 4': '',
            'n_h4': 700,
            'h4_activation': 'relu',
            'h4_dropout': 0.4,
            '      ': '',
            '       ': '',
            'HIDDEN LAYER 5': '',
            'n_h5': 500,
            'h5_activation': 'relu',
            'h5_dropout': 0.4,
            '        ': '',
            'HIDDEN LAYER 6': '',
            'n_h6': 300,
            'h6_activation': 'relu',
            'h6_dropout': 0.4,
            '         ': '',
            'OUTPUT LAYER': '',
            'n_classes': 5,
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

    def load_data(self, path):
        self.logger.info('Load ' + path)

        error_counter = 0

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
                else:
                    error_counter += 1

                pbar.update()

        return scores, labels

    def scramble_data(self, x, y, num_docs):
        shuffled = list(zip(x, y))
        random.shuffle(shuffled)

        scrambled = []
        for elem in shuffled:
            train = list(map(lambda b: np.ndarray.tolist(b), np.array_split(elem[0], num_docs)))
            tuple_zip = list(zip(train, elem[1]))
            random.shuffle(tuple_zip)
            tuple_zip = list(zip(*tuple_zip))
            tuple_zip[0] = [item for sublist in tuple_zip[0] for item in sublist]
            tuple_zip[1] = list(tuple_zip[1])
            scrambled.append(tuple_zip)

        x, y = zip(*scrambled)

        return np.asarray(x), np.asarray(y)

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
        model.add(Dropout(self.hyper_params['h1_dropout']))

        # Hidden layer 2
        model.add(Dense(self.hyper_params['n_h2']))
        model.add(Activation(self.hyper_params['h2_activation']))
        model.add(Dropout(self.hyper_params['h2_dropout']))

        # Hidden layer 3
        model.add(Dense(self.hyper_params['n_h3']))
        model.add(Activation(self.hyper_params['h3_activation']))
        model.add(Dropout(self.hyper_params['h3_dropout']))

        # Hidden layer 4
        model.add(Dense(self.hyper_params['n_h4']))
        model.add(Activation(self.hyper_params['h4_activation']))
        model.add(Dropout(self.hyper_params['h4_dropout']))

        # Hidden layer 5
        model.add(Dense(self.hyper_params['n_h5']))
        model.add(Activation(self.hyper_params['h5_activation']))
        model.add(Dropout(self.hyper_params['h5_dropout']))

        # Hidden layer 6
        model.add(Dense(self.hyper_params['n_h6']))
        model.add(Activation(self.hyper_params['h6_activation']))
        model.add(Dropout(self.hyper_params['h6_dropout']))

        # Output layer
        model.add(Dense(self.hyper_params['n_classes']))
        model.add(Activation(self.hyper_params['out_activation']))

        opt = optimizers.SGD(lr=0.12, decay=1e-2, momentum=0.9, nesterov=True)
        model.compile(loss=self.hyper_params['loss_func'], optimizer=opt, metrics=['accuracy'])
        self.logger.info('Model compiled in {0} seconds'.format(time.time() - start_time))
        return model

    def train(self):
        # Load data
        train_x, train_y = self.load_data(constants.get_squad_train_queries_path())
        test_x, test_y = self.load_data(constants.get_squad_dev_queries_path())

        self.logger.info('Label distribution of training data (before shuffle):')
        self.logger.info(self.label_distribution(train_y))

        # data_gen = DataGenerator()
        # self.logger.info('Generating random data')
        # train_x = data_gen.generate_data(40000, 5)
        # test_x = data_gen.generate_data(6000, 5)
        # self.logger.info('Done.')

        # Normalize data
        self.logger.info('Normalizing data for 0 mean and unit variance..')
        train_x = self.normalize_data(train_x, 5)
        test_x = self.normalize_data(test_x, 5)
        self.logger.info('Done normalizing.')

        # Generate labels (for generated data only)
        # self.logger.info('Generating labels for random data')
        # train_y = data_gen.generate_labels(train_x, 5)
        # test_y = data_gen.generate_labels(test_x, 5)
        # self.logger.info('Done.')

        # Scramble data
        self.logger.info('Scrambling data..')
        train_x, train_y = self.scramble_data(train_x, train_y, 5)
        # train_x, train_y = np.asarray(train_x), np.asarray(train_y)
        # test_x, test_y = self.scramble_data(test_x, test_y)
        test_x, test_y = np.asarray(test_x), np.asarray(test_y)
        self.logger.info('Done scrambling data.')

        # Do a brief sanity check of data
        self.logger.info('Label distribution of training data (after shuffle):')
        self.logger.info(self.label_distribution(train_y))
        self.logger.info('Label distribution of test data (no shuffle):')
        self.logger.info(self.label_distribution(test_y))

        # Set up model
        model = self.model()

        # Set up tensorboard
        id = str(int(round(time.time())))
        tbCallback = XTensorBoard(log_dir=self.create_run_log(id),
                                 histogram_freq=0,
                                 write_graph=True,
                                 write_images=True,
                                 custom_log_func=self.custom_log_func)

        # Train model
        self.logger.info('Training model with hyper params:')
        self.log_hyper_params(id)

        model.fit(train_x, train_y,
                  epochs=self.hyper_params['epochs'],
                  batch_size=self.hyper_params['batch_size_train'],
                  validation_data=(test_x, test_y),
                  verbose=2,
                  callbacks=[tbCallback])

        score = model.evaluate(test_x, test_y, batch_size=self.hyper_params['batch_size_test'])

        self.logger.info('Score: [loss, accuracy]: {0}'.format(score))

    def label_distribution(self, label_data):
        freq = [0] * len(label_data[0])

        for label in label_data:
            freq[int(np.argmax(label))] += 1

        s = sum(freq)
        return list(map(lambda i: float(i) / float(s), freq))

    def custom_log_func(self, tensorboard, epoch, logs=None):

        # Add learning rate
        optimizer = tensorboard.model.optimizer
        lr = keras.backend.eval(tf.cast(optimizer.lr, tf.float64) * (
        1.0 / (1.0 + tf.cast(optimizer.iterations, tf.float64) * tf.cast(optimizer.decay, tf.float64))))

        return {
            "learning_rate": lr
        }
