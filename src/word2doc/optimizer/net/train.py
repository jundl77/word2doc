import time
import random
import keras
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop

from tqdm import tqdm
from word2doc.util import constants
from word2doc.util import logger


class TrainKeras:

    def __init__(self):
        self.logger = logger.get_logger()

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

        return np.asarray(x), np.asarray(y)

    def model(self):
        start_time = time.time()
        self.logger.info('Compiling Model ... ')
        model = Sequential()
        model.add(Dense(500, input_dim=20))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.4))

        model.add(Dense(300))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Dense(5))
        model.add(Activation('softmax'))

        rms = RMSprop()
        model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])
        self.logger.info('Model compield in {0} seconds'.format(time.time() - start_time))
        return model

    def train(self, epochs=20, batch=256):
        # Load data
        train_x, train_y = self.load_data(constants.get_squad_train_queries_path())
        test_x, test_y = self.load_data(constants.get_squad_dev_queries_path())

        # Scramble data
        train_x, train_y = self.scramble_data(train_x, train_y)
        test_x, test_y = self.scramble_data(test_x, test_y)

        # Set up model
        model = self.model()

        # Set up tensorboard
        tbCallback = keras.callbacks.TensorBoard(log_dir=constants.get_logs_dir(),
                                                 histogram_freq=0,
                                                 write_graph=True,
                                                 write_images=True)
        # Train model
        self.logger.info('Training model...')
        model.fit(train_x, train_y,
                  epochs=epochs,
                  batch_size=batch,
                  validation_data=(test_x, test_y),
                  verbose=2,
                  callbacks=[tbCallback])

        score = model.evaluate(test_x, test_y, batch_size=16)

        self.logger.info("Network's test score [loss, accuracy]: {0}".format(score))
