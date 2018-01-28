import os
import random
import time

import keras
import numpy as np
import prettytable
import tensorflow as tf
from keras import optimizers
from keras.layers.core import Dense, Dropout, Activation
from word2doc.optimizer.net.APrioriLayer import APrioriLayer
from keras.models import Sequential
from tqdm import tqdm

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
            'epochs': 10,
            'batch_size_train': 8,
            'batch_size_test': 8,
            'n_input': 4096,
            '': '',
            'LEARNING RATE': '',
            ' ': '',
            'HIDDEN LAYER 1': '',
            'n_h1': 300,
            'h1_activation': 'relu',
            'h1_dropout': 0.4,
            '  ': '',
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

    def model(self, aprioris):
        start_time = time.time()
        self.logger.info('Compiling Model ... ')

        model = Sequential()

        # Hidden layer 1
        model.add(Dense(self.hyper_params['n_h1'], input_dim=self.hyper_params['n_input']))
        model.add(Activation(self.hyper_params['h1_activation']))
        #model.add(Dropout(self.hyper_params['h1_dropout']))

        # Hidden layer 2
        model.add(Dense(self.hyper_params['n_h2']))
        model.add(Activation(self.hyper_params['h2_activation']))
        #model.add(Dropout(self.hyper_params['h2_dropout']))

        # Hidden layer 3
        model.add(Dense(self.hyper_params['n_h3']))
        model.add(Activation(self.hyper_params['h3_activation']))
        #model.add(Dropout(self.hyper_params['h3_dropout']))

        # Output layer
        model.add(Dense(self.hyper_params['n_classes']))
        model.add(Activation(self.hyper_params['out_activation']))

        model.add(APrioriLayer(aprioris))

        #opt = optimizers.SGD(lr=0.12, decay=1e-2, momentum=0.9, nesterov=True)
        opt = optimizers.Adam(lr=0.001)
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

        # succ, err, acc = self.check_data(train_x, train_y, 5)
        s_tr_x, s_tr_y = self.sort_data(train_x, train_y)
        s_te_x, s_te_y = self.sort_data(test_x, test_y)
        # a = self.avg_data(s_t_x[0], 5)
        # b = self.avg_data(s_t_x[1], 5)
        # c = self.avg_data(s_t_x[2], 5)
        # d = self.avg_data(s_t_x[3], 5)
        # e = self.avg_data(s_t_x[4], 5)

        # del s_tr_x[0]
        # del s_tr_y[0]
        # del s_te_x[0]
        # del s_te_y[0]

        # train_x = [item for sublist in s_tr_x for item in sublist]
        # train_y = [item for sublist in s_tr_y for item in sublist]
        # test_x = [item for sublist in s_te_x for item in sublist]
        # test_y = [item for sublist in s_te_y for item in sublist]

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
        aprioris = self.label_distribution(test_y)
        self.logger.info(aprioris)

        # Set up model
        model = self.model(aprioris)

        # Set up tensorboard
        id = str(int(round(time.time())))
        tbCallback = LoggableTensorBoard(log_dir=self.create_run_log(id),
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

        # model_part = keras.Model(input=[model.layers[0]], output=[model.layers[6]])
        # model_part.compile(loss='categorical_crossentropy', optimizer='sgd')
        # model_part.fit(X_train, y_train, nb_epoch=2)

        score = model.evaluate(test_x, test_y, batch_size=self.hyper_params['batch_size_test'])
        # wo_ap = self.intermediate_output(model, 5, test_x[0:1])
        # w_ap = model.predict(test_x[0:1])

        self.logger.info('Score: [loss, accuracy]: {0}'.format(score))

    def label_distribution(self, label_data):
        freq = [0] * len(label_data[0])

        for label in label_data:
            freq[int(np.argmax(label))] += 1

        s = sum(freq)
        return list(map(lambda i: float(i) / float(s), freq))

    def sort_data(self, x, y):
        data = list(zip(x, y))
        s_x = [[], [], [], [], []]
        s_y = [[], [], [], [], []]

        for elem in data:
            index = int(np.argmax(elem[1]))
            if index == 0:
                s_x[0].append(elem[0])
                s_y[0].append(elem[1])
            elif index == 1:
                s_x[1].append(elem[0])
                s_y[1].append(elem[1])
            elif index == 2:
                s_x[2].append(elem[0])
                s_y[2].append(elem[1])
            elif index == 3:
                s_x[3].append(elem[0])
                s_y[3].append(elem[1])
            elif index == 4:
                s_x[4].append(elem[0])
                s_y[4].append(elem[1])

        return s_x, s_y

    def intermediate_output(self, model, layer_num, data):
        intermediate_layer_model = keras.Model(inputs=model.input, outputs=model.layers[layer_num].output)
        return intermediate_layer_model.predict(data)

    def avg_data(self, x, num_docs):
        avg_data = []

        # Go through every input for net
        for elem in np.asarray(x):

            # Dissect input into values for each document, so we get an array
            train = list(map(lambda b: np.ndarray.tolist(b), np.array_split(elem, num_docs)))

            l_2 = [0, 0, 0, 0, 0]

            # Normalize columns in array
            for j in range(0, len(train[0])):
                mean_list = list()

                # Extract column into mean_list
                for i in range(0, len(train)):
                    mean_list.append(train[i][j])

                # Calculate mean
                mean_list = np.asarray(mean_list)
                l_2[j] = mean_list.max() - mean_list.min()

            # Flatten list again and add to general result
            avg_data.append(l_2)

        return avg_data

    def check_data(self, x, y, num_docs):
        data = list(zip(x, y))

        succ = 0
        err = 0

        # Go through every input for net
        for elem in np.asarray(data):

            # Dissect input into values for each document, so we get an array
            train = list(map(lambda b: np.ndarray.tolist(b), np.array_split(elem[0], num_docs)))

            l = []
            # Normalize columns in array
            for i in range(0, len(train)):
                l.append(train[i][0])

            if elem[1][np.argmax(l)] == 1:
                succ += 1
            else:
                err += 1

        return succ, err, (float(succ) / float(succ + err))

    def custom_log_func(self, tensorboard, epoch, logs=None):

        # Add learning rate
        optimizer = tensorboard.model.optimizer
        lr = keras.backend.eval(tf.cast(optimizer.lr, tf.float64) * (
            1.0 / (1.0 + tf.cast(optimizer.iterations, tf.float64) * tf.cast(optimizer.decay, tf.float64))))

        return {
            "learning_rate": lr
        }

