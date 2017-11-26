import json
import os
import numpy as np

from word2doc.util import constants
from word2doc.util import logger


class OptimizerPreprocessor:

    def __init__(self, model):
        self.model = model
        self.logger = logger.get_logger()
        return

    def preprocess(self):
        self.logger.info('Preprocessing training and test data for optimizer')

        if os.path.isfile(constants.get_optimizer_data_path()):
            self.logger.info('Exisiting optimizier data found, loading into memory..')
            return self.__load_data()

        squad_train = self.__preprocess_squad(constants.get_squad_train_path())
        squad_test = self.__preprocess_squad(constants.get_squad_train_path())

        data = {
            'train': squad_train,
            'test': squad_test
        }

        self.logger.info('Saving optimizer preprocessed data...')
        self.__save_data(data)
        self.logger.info('Done.')

        return [squad_train, squad_test]

    def __preprocess_squad(self, path):
        squad = json.load(open(path))
        data = squad['data']

        doc_titles = np.array([])
        queries = {}

        self.logger.info('Gathering labels and queries from squad...')
        for doc in data:
            title = doc['title']
            title = title.replace('_', ' ')
            doc_titles = np.append(doc_titles, title)
            doc_titles = np.unique(doc_titles)
            paragraphs = doc['paragraphs']

            for par in paragraphs:
                qas = par['qas']

                for qa in qas:
                    question = qa['question']
                    index = np.where(doc_titles == title)[0][0]
                    # Run through model
                    queries[question] = {
                        'label_index': index,
                        'docs': self.model.calculate_rankings(question)
                    }

        self.logger.info('Done.')

        return {
            'labels': doc_titles,
            'queries': queries,
        }

    def __save_data(self, data):
        np.savez(constants.get_optimizer_data_path(), **data)

    def __load_data(self):
        with np.load(constants.get_optimizer_data_path()) as data:
            train = data['train']
            test = data['test']
        return [train, test]
