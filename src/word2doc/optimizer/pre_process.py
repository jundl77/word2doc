
import json
import os
import numpy as np

from tqdm import tqdm
from word2doc.util import constants
from word2doc.util import logger


class OptimizerPreprocessor:

    def __init__(self, model):
        self.model = model
        self.logger = logger.get_logger()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    def create_bins(self):
        self.logger.info('Load data and split it into files')

        self.__create_bins_squad(constants.get_squad_train_path())
        self.__create_bins_squad(constants.get_squad_dev_path())

    def __create_bins_squad(self, path):

        self.logger.info('Creating bins for ' + path)

        # Define path to bin folder
        base_path = os.path.splitext(path)[0]

        # Create folder for bins if it does not exist already
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        else:
            self.logger.info('Bins already exist, stopping.')
            return

        # Load data
        squad = json.load(open(path))
        data_split = np.array_split(squad['data'], constants.get_number_workers())

        # Save data to bins
        counter = 1
        for b in data_split:
            name = os.path.join(base_path, str(counter) + '.npy')
            np.save(name, b)
            counter += 1

    def pre_process_squad(self, path, bin_id):
        # Define path to bin folder
        bin_dir_path = os.path.splitext(path)[0]
        bin_path = os.path.join(bin_dir_path, str(bin_id) + '.npy')

        # Load bin data
        bin_data = np.load(bin_path)

        queries = {}

        self.logger.info('Gathering labels and queries from squad...')
        with tqdm(total=len(bin_data)) as pbar:
            for doc in tqdm(bin_data):
                title = doc['title']
                title = title.replace('_', ' ')
                paragraphs = doc['paragraphs']

                for par in paragraphs:
                    qas = par['qas']

                    for qa in qas:
                        question = qa['question']

                        # Run through model
                        queries[question] = {
                            'label': title,
                            'docs': self.model.calculate_rankings(question)
                        }

                pbar.update()

        self.logger.info('Done with bin ' + str(bin_id))

        name = os.path.join(bin_dir_path, str(bin_id) + '-queries.npy')
        np.save(name, queries)

    def merge_bins(self):
        self.__merge_bins_squad(constants.get_squad_train_path())
        self.__merge_bins_squad(constants.get_squad_dev_path())

    def __merge_bins_squad(self, path):
        # Define path to bin folder
        bin_dir_path = os.path.splitext(path)[0]

        base_dir = os.path.dirname(path)
        file_name = os.path.splitext(base_dir)[0]

        data = {}
        # Load data from bins
        for i in range(1, constants.get_number_workers()) :
            bin_path = os.path.join(bin_dir_path, str(i) + '-queries.npy')
            bin_data = np.load(bin_path)

            # Append bin to all data
            data.update(bin_data)

        name = os.path.join(base_dir, file_name + '-queries.npy')
        np.save(name, data)