import time
import random
import keras
import prettytable
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop

from tqdm import tqdm
from word2doc.util import constants
from word2doc.util import logger


class OptimizerNet:

    def __init__(self):
        self.logger = logger.get_logger()

        self.hyper_params = {
            'TRAINING PARAMS': '',
            'epochs': 20,
            'batch_size_train': 256,
            'batch_size_test': 16,
            'n_input': 20,
            '': '',
            'HIDDEN LAYER 1': '',
            'n_h1': 500,
            'h1_activation': 'relu',
            'h1_dropout': 0.4,
            ' ': '',
            'HIDDEN LAYER 2': '',
            'n_h2': 300,
            'h2_activation': 'relu',
            'h2_dropout': 0.4,
            '  ': '',
            'OUTPUT LAYER': '',
            'n_classes': 5,
            'out_activation': 'softmax',
            
        } 

    def log_hyper_params(self):
        table = prettytable.PrettyTable(['Hyper Parameter', 'Value'])

        for key, val in self.hyper_params.items():
            table.add_row([key, val])
            
        self.logger.info(table)

    def load_data(self, path):
        self.logger.info('Load ' + path)

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

        return np.asarray(x), np.asarray(y)

    def model(self):
        start_time = time.time()
        self.logger.info('Compiling Model ... ')

        model = Sequential()

        # Hidden layer 1
        model.add(Dense(self.hyper_params['n_h1'], input_dim=self.hyper_params['n_input']))
        model.add(BatchNormalization())
        model.add(Activation(self.hyper_params['h1_activation']))
        model.add(Dropout(self.hyper_params['h1_dropout']))

        # Hidden layer 2
        model.add(Dense(self.hyper_params['n_h2']))
        model.add(BatchNormalization())
        model.add(Activation(self.hyper_params['h2_activation']))
        model.add(Dropout(self.hyper_params['h2_dropout']))

        # Output layer
        model.add(Dense(self.hyper_params['n_classes']))
        model.add(Activation(self.hyper_params['out_activation']))

        rms = RMSprop()
        model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])
        self.logger.info('Model compield in {0} seconds'.format(time.time() - start_time))
        return model

    def train(self):
        # Load data
        train_x, train_y = self.load_data(constants.get_squad_train_queries_path())
        test_x, test_y = self.load_data(constants.get_squad_dev_queries_path())

        # Scramble data
        self.logger.info('Scrambling data..')
        train_x, train_y = self.scramble_data(train_x, train_y)
        test_x, test_y = self.scramble_data(test_x, test_y)
        self.logger.info('Done scrambling data.')

        # Set up model
        model = self.model()

        # Set up tensorboard
        tbCallback = keras.callbacks.TensorBoard(log_dir=constants.get_logs_dir(),
                                                 histogram_freq=0,
                                                 write_graph=True,
                                                 write_images=True)
        # Train model
        self.logger.info('Training model with hyper params:')
        self.log_hyper_params()

        model.fit(train_x, train_y,
                  epochs=self.hyper_params['epochs'],
                  batch_size=self.hyper_params['batch_size_train'],
                  validation_data=(test_x, test_y),
                  verbose=2,
                  callbacks=[tbCallback])

        score = model.evaluate(test_x, test_y, batch_size=self.hyper_params['batch_size_test'])

        self.logger.info('Score: [loss, accuracy]: {0}'.format(score))
