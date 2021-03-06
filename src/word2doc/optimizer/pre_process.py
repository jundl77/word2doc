import json
import os
import numpy as np

from tqdm import tqdm
from random import shuffle
from random import randint
from word2doc.util import constants
from word2doc.util import logger
from word2doc.keywords import rake_extractor
from word2doc import retriever
from word2doc.retriever.doc_db import DocDB
from word2doc.labels.extractor import LabelExtractor
from word2doc.embeddings import infersent
from word2doc.util.analytics import Analytics


class SquadPreprocessor:

    def __init__(self, model):
        self.model = model
        self.logger = logger.get_logger()
        self.rake = rake_extractor.RakeExtractor()

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

    def pre_process(self, path, bin_id):

        # Define path to bin folder
        bin_path = os.path.join(path, str(bin_id) + '.npy')

        # Load bin data
        bin_data = np.load(bin_path)

        queries = {}  # Only the kept queries
        seen_questions = []

        query_error_count = 0
        query_duplicate_count = 0
        total_queries_processed = 0

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
                        real_q = question

                        # Take most relevant part of query, at least for training
                        question = self.rake.extract(question.lower())[0]

                        # Make sure there are not duplicates (not even in substrings)
                        res = list(filter(lambda q: q in question, seen_questions))
                        if len(res) > 0:
                            query_duplicate_count += 1
                            continue

                        # Process the question
                        scores, doc = self.model.calculate_rankings(question, label=title)
                        total_queries_processed += 1

                        if doc is not None:
                            # Preempted
                            seen_questions.append(question)
                            continue

                        if scores is None or title not in scores:
                            # Label was not retrieved
                            query_error_count += 1
                        else:
                            # Remove any queries that are more specific (current is substring of an existing query)
                            res = list(filter(lambda q: question in q, queries))
                            if len(res) > 0:
                                for q in res:
                                    query_duplicate_count += 1
                                    del queries[q]

                            # Mark as seen
                            seen_questions.append(question)

                            # Run through model
                            queries[question] = {
                                'label': title,
                                'docs': scores
                            }

                pbar.update()

        self.logger.info('Done with bin ' + str(bin_id))

        # Update analytics model
        analytics = self.model.get_analytics()
        analytics.queries_processed(total_queries_processed, query_duplicate_count, query_error_count)
        analytics.save_to_file('squad_analytics_bin' + str(bin_id))

        name = os.path.join(path, str(bin_id) + '-queries.npy')
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


class Word2DocPreprocessor:

    def __init__(self):
        self.doc_db = DocDB(constants.get_db_path())
        self.logger = logger.get_logger()
        self.analytics = Analytics()
        self.ranker = retriever.get_class('tfidf')(tfidf_path=constants.get_retriever_model_path())
        self.rake = rake_extractor.RakeExtractor()
        self.extractor = LabelExtractor(constants.get_db_path())
        self.infersent = infersent.get_infersent()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    def pre_process(self, bin_id, num_bins):

        doc_titles = self.doc_db.get_doc_ids()

        data = list()
        invalid_count = 0
        freq_error_count = 0
        counter = 0

        bin_size = round(len(doc_titles) / num_bins)
        index_start = bin_size * (bin_id - 1)
        index_end = index_start + bin_size

        # Make sure that any remaining objects are not ignored at the end by an imperfect bin size
        if len(doc_titles) - index_end < bin_size:
            index_end = len(doc_titles)

        doc_titles_bin = doc_titles[index_start:index_end]

        self.logger.info('Creating pivot embeddings for docs...')
        with tqdm(total=len(doc_titles_bin)) as pbar:
            for doc in tqdm(doc_titles_bin):

                index = index_start + counter

                clean_title = ''.join(e for e in doc.lower() if e.isalnum() or e == ' ')

                # Make sure we don't have a title that has only special chars
                if clean_title == '':
                    invalid_count += 1
                    counter += 1
                    continue

                # Get intro paragraph of document
                intro_par = self.extractor.extract_label(doc)
                sentences = intro_par.split('.')
                topic_sent = sentences[0]
                rest_par = '.'.join(sentences[1:])

                # Get up to to 6 keywords from doc, including title. These will be the 'context' words around which the
                # actual document will pivot
                pivots = [doc]
                pivots = pivots + self.rake.extract(topic_sent)[:2]
                pivots = pivots + self.rake.extract(rest_par)[:6 - len(pivots)]

                pivots = self.__remove_duplicates_str(pivots)

                # Encode pivots as word embeddings using infersent
                pivots_embedding = list(map(lambda p: self.infersent.encode(p), pivots))

                # Get closest docs
                num_docs = 10
                try:
                    doc_names, doc_scores = self.ranker.closest_docs(doc, num_docs)
                    doc_window = list(map(lambda d: self.__safe_index(doc_titles, d), doc_names))

                    # Make sure correct doc is always there
                    if index not in doc_window:
                        freq_error_count += 1
                        # doc_window[len(doc_window) - 1] = index

                    shuffle(doc_window)
                except RuntimeError:
                    doc_window = [-1] * num_docs

                # Append word embeddings to data. Index describes correct document id (index 0 -> doc 0)
                data.append({
                    'doc_index': index,
                    'doc_title': doc,
                    'pivot_embeddings': pivots_embedding,
                    'doc_window': doc_window
                })

                counter += 1
                pbar.update()

        self.logger.info("Invalid count: " + str(invalid_count))

        # Update analytics model
        self.analytics.docs_processed(counter, freq_error_count, invalid_count)
        self.analytics.save_to_file('word2doc_analytics_bin' + str(bin_id))

        # Save to file
        self.logger.info("Done creating pivot embeddings, saving to file..")
        name = os.path.join(constants.get_word2doc_dir(), str(bin_id) + '-wpp.npy')
        np.save(name, data)
        self.logger.info("Saved to file.")

    def __remove_duplicates_str(self, list):
        output = []
        seen = set()
        for value in list:
            clean_value = ''.join(e for e in value.lower() if e.isalnum() or e == ' ')

            if clean_value not in seen:
                output.append(value)
                seen.add(clean_value)

        return output

    def __remove_duplicates_int(self, list):
        output = []
        seen = set()
        for value in list:

            if value not in seen:
                output.append(value)
                seen.add(value)

        return output

    def __safe_index(self, data, elem):
        try:
            return data.index(elem)
        except ValueError:
            return -1

    def merge_bins(self, num_bins):
        self.logger.info('Merging bins..')
        self.__merge_bins(constants.get_word2doc_dir(), num_bins)

    def __merge_bins(self, path, num_bins):
        self.logger.info('Merging bins in ' + path)

        data = {}
        # Load data from bins
        for i in range(1, num_bins):
            bin_path = os.path.join(path, str(i) + '-wpp.npy')
            bin_data = np.load(bin_path)
            list_data = np.ndarray.tolist(bin_data)

            # Append bin to all data
            if len(data) == 0:
                data = list_data
            else:
                data = np.append(data, list_data)

        self.logger.info('Done.')
        name = os.path.join(path, 'word2doc-pp.npy')

        self.logger.info('Saving to ' + name)
        np.save(name, data)
        self.logger.info('Saved.')



