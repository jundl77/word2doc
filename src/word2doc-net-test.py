#!/usr/bin/env python3
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Interactive mode for the tfidf retriever module."""

import argparse
import code
import prettytable
from random import randint
from word2doc.retriever.doc_db import DocDB
import pprint
import json
import locale
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

from word2doc import model
from word2doc.util import logger
from word2doc.util import init_project
from word2doc.optimizer.net.word2doc import Word2Doc


class Word2DocTest:

    def __init__(self):
        self.doc_db = DocDB(constants.get_db_path())
        self.logger = logger.get_logger()
        self.analytics = Analytics()
        self.ranker = retriever.get_class('tfidf')(tfidf_path=constants.get_retriever_model_path())
        self.rake = rake_extractor.RakeExtractor()
        self.extractor = LabelExtractor(constants.get_db_path())
        self.infersent = infersent.get_infersent()
        self.doc_titles = self.doc_db.get_doc_ids()
        self.word2doc = Word2Doc()

    def run_test(self):
        self.logger.info("Starting test.")
        succ_counter = 0
        err_counter = 0
        total_counter = 0
        test_data = list()
        log = list()

        _, _, old_context, titles = self.word2doc.load_train_data(os.path.join(constants.get_word2doc_dir(), '3-wpp.npy'))

        test_data_old = np.load(os.path.join(constants.get_word2doc_dir(), 'word2doc-test-bin-3.npy'))

        while total_counter < 200:
            try:
                # Choose random document
                index = randint(0, len(titles) - 1)
                doc = titles[index]

                for old_doc in test_data_old:
                    if old_doc["doc_title"] == doc:
                        continue

                print("Run no. " + str(total_counter))
                if total_counter > 0:
                    acc = float(succ_counter) / float(total_counter)
                    print("Accuracy: " + str(acc))
                print(u' '.join(("Document: ", doc)).encode('utf-8'))
                intro_par = self.extractor.extract_label(doc)
                print("Intro paragraph:")
                print(intro_par)

                skip = input("Skip? (y\\n)")
                if skip == "y":
                    continue

                valid = False
                while not valid:
                    category = input("Enter a category (1: normal, 2: typo, 3: abbreviation, 4: number): ")
                    try:
                        category = int(category)
                        valid = True
                    except ValueError:
                        print("Invalid category.")

                keyword = input("Create a keyword:\n")
                data = self.pre_process(keyword)
                # self.logger.info("Evaluating " + keyword + "..")
                # res, data = self.eval(data, old_context)
                #
                # if res is None:
                #     continue
                #
                # if res:
                #     prompt = False
                #     correct = False
                #
                #     while not prompt:
                #         answer = input("Is the response correct? (y\\n)")
                #
                #         if answer == "y":
                #             succ_counter += 1
                #             print("Saved success.")
                #             correct = True
                #             prompt = True
                #         elif answer == "n":
                #             err_counter += 1
                #             print("Saved error.")
                #             correct = False
                #             prompt = True
                #         else:
                #             print("Wrong input, try again (y\\n)")
                # else:
                #     err_counter += 1
                #     correct = False
                #     self.logger.info("Saved error.")

                test_data.append({
                    'doc_index': index,
                    'doc_title': doc,
                    'category': category,
                    'query': keyword,
                    'pivot_embeddings': data['pivot_embedding'],
                    'doc_window': data['doc_window']
                })

                # log.append({
                #     "Tested document": doc,
                #     "User keyword": keyword,
                #     "Number of context docs removed": data['n_docs_rm'],
                #     "Correct": correct
                # })
                total_counter += 1

                # Save test data
                name = os.path.join(constants.get_word2doc_dir(), 'word2doc-test-400.npy')
                np.save(name, test_data)
            except:
                print("Error, continuing")
                continue

        acc = float(succ_counter) / float(total_counter)
        log.append({
            "Total docs tested": total_counter,
            "Number of correct docs:": succ_counter,
            "Number of incorrect docs": err_counter,
            "Accuracy": acc
        })

        # Save test data
        name = os.path.join(constants.get_word2doc_dir(), 'word2doc-test-bin-3.npy')
        np.save(name, test_data)

        # Save log
        with open(os.path.join(constants.get_logs_dir(), 'word2doc_test_3' + '.json'), 'w') as fp:
            json.dump(log, fp, sort_keys=True, indent=4)

        self.logger.info("Accuracy over 200 randomly chosen documents: " + str(acc))

    def eval(self, data, old_context):
        embb = data['pivot_embedding']
        ctx = data['doc_window']

        ctx, n_docs_rm = self.__normalize_context(old_context, ctx)
        pp = pprint.PrettyPrinter(indent=2)

        if ctx is None:
            print("Not enough context docs found, skipping.")
            return None, None

        data['n_docs_rm'] = n_docs_rm
        data['doc_window'] = ctx

        docs = self.word2doc.predict([embb], [ctx])
        print("Documents found:")
        pp.pprint(docs)
        return True, data

    def pre_process(self, keyword,):
        # Encode pivots as word embeddings using infersent
        pivots_embedding = self.infersent.encode(keyword)

        # Get closest docs
        num_docs = 10
        try:
            doc_names, doc_scores = self.ranker.closest_docs(keyword, num_docs)
            doc_window = list(map(lambda d: self.__safe_index(self.doc_titles, d), doc_names))

            shuffle(doc_window)
        except RuntimeError:
            doc_window = [-1] * num_docs

        data = {
            'keyword': keyword,
            'pivot_embedding': pivots_embedding,
            'doc_window': doc_window
        }

        return data

    def __remove_duplicates_str(self, list):
        output = []
        seen = set()
        for value in list:
            clean_value = ''.join(e for e in value.lower() if e.isalnum() or e == ' ')

            if clean_value not in seen:
                output.append(value)
                seen.add(clean_value)

        return output

    def __safe_index(self, data, elem):
        try:
            return data.index(elem)
        except ValueError:
            return -1

    def __normalize_context(self, old_ctx, new_ctx):
        old_ctx = list(old_ctx)
        counter = 0
        i = 0
        mapping = {}
        for context in old_ctx:
            for num in context:
                if num not in mapping:
                    mapping[num] = i
                    i += 1

        ctx = list()
        for context in new_ctx:
            if context in mapping:
                ctx.append(mapping[context])
            else:
                counter += 1

        if len(ctx) < 1:
            return None

        # Filling up contexts with duplicates
        for i in range(0, 10 - len(ctx)):
            index = randint(0, len(ctx) - 1)
            ctx.append(ctx[index])

        self.logger.info("Removed " + str(counter) + " unindexed documents from context")

        return ctx, counter


# Start test here
init_project.init(1)
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

test = Word2DocTest()
test.run_test()
