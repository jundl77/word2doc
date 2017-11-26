#!/usr/bin/env python3
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Interactive mode for the tfidf retriever module."""

import argparse
import code
import sys
import prettytable
import numpy as np
from functools import reduce

from word2doc import retriever
from word2doc.util import logger
from word2doc.labels import extractor
from word2doc.util import constants
from word2doc.embeddings import infersent
from word2doc.references import reference_graph_builder
from word2doc.keywords import rake_extractor

logger = logger.get_logger()

parser = argparse.ArgumentParser()
parser.add_argument('db_path', type=str, help='/path/to/db')
parser.add_argument('--model', type=str, default=None)
args = parser.parse_args()

sys.path.append('/Users/julianbrendl/Projects/bachelor-thesis/word2doc/src/word2doc/embeddings/infersent/')
logger.info('Initializing ranker...')
ranker = retriever.get_class('tfidf')(tfidf_path=args.model)


# ------------------------------------------------------------------------------
# Drop in to interactive
# ------------------------------------------------------------------------------

PROCESS_DB = None
INFERSENT = None
RAKE = None
LABELS = None


def init(db_path):
    global PROCESS_DB, INFERSENT, RAKE, LABELS
    db_class = retriever.get_class('sqlite')
    PROCESS_DB = db_class(db_path)
    INFERSENT = infersent.get_infersent()
    RAKE = rake_extractor.RakeExtractor()
    LABELS = {}


def process(query, k=5):
    global LABELS, INFERSENT, RAKE
    doc_names, doc_scores = ranker.closest_docs(query, k)

    # Get candidate labels for candidate doc
    e = extractor.LabelExtractor(constants.get_db_path())
    print('Docs found: ' + str(doc_names))

    # Filter out more specific docs with reference tree
    print('Filter out more specific docs with reference tree')
    rgraph_builder = reference_graph_builder.ReferencesGraphBuilder()
    ref_graph = rgraph_builder.build_references_graph(doc_names)
    doc_names = list(map(lambda child: child, ref_graph.get_children()))
    print('Docs kept: ' + str(doc_names))

    # Perform sentence embeddings
    max_score = -1000
    embedding_scores = []
    title_scores = []
    keyword_scores = []
    index = 0
    for i in range(len(doc_names)):
        label = e.extract_label(doc_names[i])
        LABELS[doc_names[i]] = label

        # Embedding label
        score = INFERSENT.compare_sentences(query.lower(), label.lower())
        embedding_scores.append(score)

        # Embedding title
        score_title = INFERSENT.compare_sentences(query.lower(), doc_names[i].lower())
        title_scores.append(score_title)

        # Embedding keywords
        keywords = RAKE.extract(label.lower())
        keyword_embeddings = list(map(lambda w: INFERSENT.compare_sentences(query.lower(), w), keywords))
        print('\n\nDocument: ' + doc_names[i])
        list(filter(lambda k: print("Keyword: " + str(k[0]) + " --- Score: " + str(k[1])), list(zip(keywords, keyword_embeddings))))
        # keyword_embeddings = reject_outliers(keyword_embeddings)
        keyword_scores.append(reduce(lambda x, y: x + y, keyword_embeddings) / len(keyword_embeddings))

        if score > max_score:
            max_score = score
            index = i


    table = prettytable.PrettyTable(
        ['Rank', 'Doc Id', 'Doc Score', 'Label Embedding Score', 'Title Embedding Score', 'Keyword Embedding Score']
    )
    for i in range(len(doc_names)):
        table.add_row([i + 1, doc_names[i], '%.5g' % doc_scores[i], embedding_scores[i], title_scores[i], keyword_scores[i]])
    print(table)

    print("Best document based on sentence embedding:")
    print(doc_names[index])


def get_doc(doc_id):
    global PROCESS_DB
    return PROCESS_DB.get_doc_text(doc_id)


def get_label(doc_id):
    global LABELS
    return LABELS[doc_id]


def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]


banner = """
Interactive TF-IDF Retriever
>> process(question, k=5)
>> get_doc(doc_id)
>> get_label(doc_id)
>> usage()
"""


def usage():
    print(banner)


init(args.db_path)
code.interact(banner=banner, local=locals())