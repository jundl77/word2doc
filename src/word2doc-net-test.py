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
        log = dict()

        _, _, old_context, titles = self.word2doc.load_data(os.path.join(constants.get_word2doc_dir(), '3-wpp.npy'))

        while total_counter < 100:
            # Choose random document
            index = randint(0, len(titles) - 1)
            doc = titles[index]

            print("Run no. " + str(total_counter))
            print("Document: " + doc)

            intro_par = self.extractor.extract_label(doc)
            print("Intro paragraph:")
            print(intro_par)

            skip = input("Skip? (y\\n)")
            if skip == "y":
                continue

            keyword = input("Create a keyword:\n")
            self.logger.info("Evaluating " + keyword + "..")
            res = self.eval(keyword, old_context, titles)

            if res is None:
                continue

            if res:
                answer = input("Is the response correct? (y\\n)")

                prompt = False
                while not prompt:
                    if answer == "y":
                        succ_counter += 1
                        print("Saved success.")
                        prompt = True
                    elif answer == "n":
                        err_counter += 1
                        print("Saved error.")
                        prompt = True
                    else:
                        print("Wrong input, try again (y\\n)")
            else:
                err_counter += 1
                self.logger.info("Saved error.")


            log += {
                "Tested document": doc,
                "User keyword": keyword
            }
            total_counter += 1

        acc = float(succ_counter) / float(total_counter)
        self.logger.info("Accuracy over 200 randomly chosen documents: " + str(acc))

    def eval(self, keyword, old_context, titles):
        data = self.pre_process(keyword)
        embb = data['pivot_embedding']
        ctx = data['doc_window']

        ctx = self.__normalize_context(old_context, ctx)

        if ctx is None:
            print("Not enough context docs found, skipping.")
            return None

        pred = self.word2doc.predict([embb], [ctx])

        if pred[0][0] in titles:
            self.logger.info("Closest document: " + titles[pred[0][0]])
            return True
        else:
            self.logger.info("Document not found.")
            return False

    def pre_process(self, keyword):
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

        if len(ctx) < 5:
            return None

        # Filling up contexts with duplicates
        for i in range(0, 10 - len(ctx)):
            index = randint(0, len(ctx) - 1)
            ctx.append(ctx[index])

        self.logger.info("Removed " + str(counter) + " unindexed documents from context")

        return ctx


# Start test here
init_project.init(1)
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

test = Word2DocTest()
test.run_test()
