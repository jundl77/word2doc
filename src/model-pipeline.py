#!/usr/bin/env python3
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import prettytable

from word2doc import retriever
from word2doc.labels import extractor
from word2doc.util import constants
from word2doc.util import logger
from word2doc.util import init_project

logger = logger.get_logger()
TFIDF_DOC_N = 5


# ------------------------------------------------------------------------------
# Data pipeline that builds the processes the data for the retriever.
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('query', type=str, help='The keyword that you want an explanation for.')
    args = parser.parse_args()

    # Init project
    init_project.init()
    save_path = constants.get_db_path()
    logger.info('Initializing ranker...')
    ranker = retriever.get_class('tfidf')(tfidf_path=constants.get_retriever_model_path())

    # Get tf-idf document list (candidate docs are in doc_names)
    doc_names, doc_scores = ranker.closest_docs(args.query, TFIDF_DOC_N)
    table = prettytable.PrettyTable(
        ['Rank', 'Doc Id', 'Doc Score']
    )
    for i in range(len(doc_names)):
        table.add_row([i + 1, doc_names[i], '%.5g' % doc_scores[i]])
    print(table)

    # Get candidate labels for candidate doc
    e = extractor.LabelExtractor(constants.get_db_path())
    labels = []
    for i in range(len(doc_names)):
        labels[i] = e.extract_label(doc_names[i])

    # Choose doc based on word-sentence embeddings
    #TODO: Implement
