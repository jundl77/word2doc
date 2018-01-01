
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

    def create_bins(self, num_bins):
        self.logger.info('Load data and split it into files')

        self.__create_bins_squad(constants.get_squad_train_path(), num_bins)
        self.__create_bins_squad(constants.get_squad_dev_path(), num_bins)

    def __create_bins_squad(self, path, num_bins):

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
        data_split = np.array_split(squad['data'], num_bins)

        # Save data to bins
        counter = 1
        for b in data_split:
            name = os.path.join(base_path, str(counter) + '.npy')
            np.save(name, b)
            counter += 1

    def pre_process_squad(self, path, bin_id):

        # Bin ids should start with 0, so add 1 (because of slurm queue)
        #bin_id += 21

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

                        docs = self.model.calculate_rankings(question)

                        # Run through model
                        if docs is not None:
                            queries[question] = {
                                'label': title,
                                'docs': docs
                            }

                pbar.update()

        self.logger.info('Done with bin ' + str(bin_id))

        name = os.path.join(bin_dir_path, str(bin_id) + '-queries.npy')
        np.save(name, queries)

    def merge_bins(self, num_bins):
        self.logger.info('Merging bins..')
        self.__merge_bins_squad(constants.get_squad_train_path(), num_bins)
        self.__merge_bins_squad(constants.get_squad_dev_path(), num_bins)

    def __merge_bins_squad(self, path, num_bins):
        self.logger.info('Merging bins in ' + path)

        # Define path to bin folder
        bin_dir_path = os.path.splitext(path)[0]

        data = {}
        # Load data from bins
        for i in range(1, num_bins) :
            bin_path = os.path.join(bin_dir_path, str(i) + '-queries.npy')
            bin_data = np.load(bin_path)
            squad_dict = np.ndarray.tolist(bin_data)

            # Append bin to all data
            data.update(squad_dict)

        self.logger.info('Done.')
        name = bin_dir_path + '-queries.npy'

        self.logger.info('Saving to ' + name)
        np.save(name, data)
        self.logger.info('Saved.')
