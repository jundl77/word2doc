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

    def calculate_rankings(self, query, k=10):
        query = self.rake.extract(query.lower())[0]  # Take most relevant part of query, at least for training

        try:
            doc_names, doc_scores = self.ranker.closest_docs(query, k)
        except RuntimeError:
            return None

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
                keyword_scores.append(reduce(lambda x, y: x + y, keyword_embeddings) / len(keyword_embeddings))

            # Preempt search because a title matches
            if score_title > 0.95:
                chosen_doc = doc_names[i]
                break

        scores = {}

        # Normalize data to zero mean and unit variance
        doc_scores = self.__normalize_list(doc_scores.tolist())
        title_scores = self.__normalize_list(title_scores)
        keyword_scores = self.__normalize_list(keyword_scores)

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
        """Normalize data to 0 mean and unit variance"""

        mean = np.mean(l)
        var = np.var(l)

        # If variance is 0, set variance to 1 (e.g. list only contains same numbers)
        if var < 0.001:
            var = 1

        return list(map(lambda x: (x - mean) / var, l))

