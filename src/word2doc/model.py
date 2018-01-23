#!/usr/bin/env python3
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Interactive mode for the tfidf retriever module."""

import numpy as np
from functools import reduce

from word2doc import retriever
from word2doc.util import logger
from word2doc.util import constants
from word2doc.util import analytics
from word2doc.labels import extractor
from word2doc.embeddings import infersent
from word2doc.references import reference_graph_builder
from word2doc.keywords import rake_extractor


class Model:

    def __init__(self, db_path, tfidf_model):
        db_class = retriever.get_class('sqlite')

        self.logger = logger.get_logger()
        self.analytics = analytics.Analytics()

        self.process_db = db_class(db_path)
        self.ranker = retriever.get_class('tfidf')(tfidf_path=tfidf_model)

        self.infersent = infersent.get_infersent()
        self.rake = rake_extractor.RakeExtractor()
        self.lables = {}

    def get_analytics(self):
        return self.analytics

    def process(self, query):
        return self.calculate_rankings(query, 10)

    def calculate_rankings(self, query, k=10, label=None):

        try:
            doc_names, doc_scores = self.ranker.closest_docs(query, k)
        except RuntimeError:
            return None, None

        if label is not None and label not in doc_names:
            # Label was not among found docs, so for training purposes this is useless
            return None, None

        # Get candidate labels for candidate doc
        e = extractor.LabelExtractor(constants.get_db_path())

        # Filter out more specific docs with reference tree
        doc_names = self.__filter_reference(query, doc_names)

        chosen_doc = None

        # Perform sentence embeddings
        title_scores = []
        keyword_scores = []
        for i in range(len(doc_names)):
            label = e.extract_label(doc_names[i])
            self.lables[doc_names[i]] = label

            # Embedding title
            score_title = self.infersent.compare_sentences(query, doc_names[i])
            title_scores.append(score_title)

            # Embedding top three keywords
            keywords = self.rake.extract(label.lower())[:3]
            keyword_embeddings = list(map(lambda w: self.infersent.compare_sentences(query, w), keywords))

            if len(keyword_embeddings) == 0:
                keyword_scores.append(0)
            else:
                tuples = zip(keyword_embeddings, [0.43, 0.33, 0.23])
                tuple_product = list(map(lambda x: x[0] * x[1], tuples))
                keyword_scores.append(reduce(lambda x, y: x + y, tuple_product))

            # Preempt search because a title matches
            if score_title > 0.95:
                self.analytics.preempted_search()
                chosen_doc = doc_names[i]
                break

        scores = {}

        # Normalize data to zero mean and unit variance
        doc_scores, doc_norm_error_count = self.__normalize_list(doc_scores.tolist())
        title_scores, title_norm_error_count = self.__normalize_list(title_scores)
        keyword_scores, keyword_norm_error_count = self.__normalize_list(keyword_scores)

        for i in range(len(title_scores)):
            scores[doc_names[i]] = [doc_scores[i], title_scores[i], keyword_scores[i]]

        return scores, chosen_doc

    def __filter_reference(self, query, doc_names):
        """Filter out more specific docs with reference tree"""

        rgraph_builder = reference_graph_builder.ReferencesGraphBuilder()
        ref_graph = rgraph_builder.build_references_graph(doc_names)
        filtered_docs = rgraph_builder.filter_titles(query, doc_names, ref_graph, self.infersent)

        # Update analytics
        self.analytics.reference_graph_analytics(doc_names, filtered_docs)

        return filtered_docs

    def __normalize_list(self, l):
        """Normalize data to 0 mean and unit variance (calculate z score)"""

        zero_count = 0

        # Normalize
        norm_list = np.asarray(l)
        std = norm_list.std()

        # Std. is 0 -> all values are the same, so they will all be set to 0 (because of subtraction of mean)
        if std < 0.00001:
            zero_count += 1
            norm_list = [float(0)] * len(l)
        else:
            norm_list = (norm_list - norm_list.mean()) / std

        return norm_list, zero_count
