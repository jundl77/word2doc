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
        return self.calculate_rankings(query, 5)

    def calculate_rankings(self, query, k=5):
        doc_names, doc_scores = self.ranker.closest_docs(query, k)

        # Get candidate labels for candidate doc
        e = extractor.LabelExtractor(constants.get_db_path())
        self.logger.info('Docs found: ' + str(doc_names))

        # Filter out more specific docs with reference tree
        doc_names = self.__filter_reference(query, doc_names)

        # Perform sentence embeddings
        embedding_scores = []
        title_scores = []
        keyword_scores = []
        for i in range(len(doc_names)):
            label = e.extract_label(doc_names[i])
            self.lables[doc_names[i]] = label

            # Embedding label
            score = self.infersent.compare_sentences(query, label)
            embedding_scores.append(score)

            # Embedding title
            score_title = self.infersent.compare_sentences(query, doc_names[i])
            title_scores.append(score_title)

            # Embedding keywords
            keywords = self.rake.extract(label.lower())
            keyword_embeddings = list(map(lambda w: self.infersent.compare_sentences(query, w), keywords))

            # keyword_embeddings = reject_outliers(keyword_embeddings)
            if len(keyword_embeddings) == 0:
                keyword_scores.append(0)
            else:
                keyword_scores.append(reduce(lambda x, y: x + y, keyword_embeddings) / len(keyword_embeddings))

        scores = {}
        for i in range(len(doc_names)):
            scores[doc_names[i]] = [doc_scores[i], embedding_scores[i], title_scores[i], keyword_scores[i]]

        return scores

    def __filter_reference(self, query, doc_names):
        """Filter out more specific docs with reference tree"""

        self.logger.info('Filter out more specific docs with reference tree')
        rgraph_builder = reference_graph_builder.ReferencesGraphBuilder()
        ref_graph = rgraph_builder.build_references_graph(doc_names)
        filtered_docs = rgraph_builder.filter_titles(query, doc_names, ref_graph, self.infersent)
        self.logger.info('Docs kept: ' + str(filtered_docs))

        # Update analytics
        self.analytics.reference_graph_analytics(doc_names, filtered_docs)

        return filtered_docs

    def __reject_outliers(self, data, m = 2.):
        d = np.abs(data - np.median(data))
        mdev = np.median(d)
        s = d/mdev if mdev else 0.
        return data[s<m]

