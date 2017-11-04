#!/usr/bin/env python3
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Interactive mode for the tfidf retriever module."""

import argparse
import code
import prettytable

from word2doc import retriever
from word2doc.util import logger

logger = logger.get_logger()


parser = argparse.ArgumentParser()
parser.add_argument('db_path', type=str, help='/path/to/db')
parser.add_argument('--model', type=str, default=None)
args = parser.parse_args()

logger.info('Initializing ranker...')
ranker = retriever.get_class('tfidf')(tfidf_path=args.model)


# ------------------------------------------------------------------------------
# Drop in to interactive
# ------------------------------------------------------------------------------

PROCESS_DB = None


def init(db_path):
    global PROCESS_DB
    db_class = retriever.get_class('sqlite')
    PROCESS_DB = db_class(db_path)


def process(query, k=5):
    doc_names, doc_scores = ranker.closest_docs(query, k)
    table = prettytable.PrettyTable(
        ['Rank', 'Doc Id', 'Doc Score']
    )
    for i in range(len(doc_names)):
        table.add_row([i + 1, doc_names[i], '%.5g' % doc_scores[i]])
    print(table)


def get_doc(doc_id):
    global PROCESS_DB
    doc = PROCESS_DB.get_doc_text(str(doc_id))
    print(doc)


banner = """
Interactive TF-IDF Retriever
>> process(question, k=5)
>> get_doc(doc_id)
>> usage()
"""


def usage():
    print(banner)


init(args.db_path)
code.interact(banner=banner, local=locals())