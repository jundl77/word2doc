#!/usr/bin/env python3
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import math

from word2doc.retriever import build_db
from word2doc.retriever import build_tfidf
from word2doc.util import constants
from word2doc.util import logger
from word2doc.util import init_project

logger = logger.get_logger()


# ------------------------------------------------------------------------------
# Data pipeline that builds the processes the data for the retriever.
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help='/path/to/data')
    parser.add_argument('--preprocess', type=str, default=None,
                        help=('File path to a python module that defines '
                              'a `preprocess` function'))
    parser.add_argument('--ngram', type=int, default=2,
                        help=('Use up to N-size n-grams '
                              '(e.g. 2 = unigrams + bigrams)'))
    parser.add_argument('--hash-size', type=int, default=int(math.pow(2, 24)),
                        help='Number of buckets to use for hashing ngrams')
    parser.add_argument('--tokenizer', type=str, default='simple',
                        help=("String option specifying tokenizer type to use "
                              "(e.g. 'corenlp')"))
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of CPU processes (for tokenizing, etc)')
    args = parser.parse_args()

    # Init project
    init_project.init()
    save_path = constants.get_db_path()

    # Build database
    logger.info('Building database...')
    build_db.store_contents(
        args.data_path, save_path, args.preprocess, args.num_workers
    )

    # Calculate tfidf data
    logger.info('Counting words...')
    count_matrix, doc_dict = build_tfidf.get_count_matrix(
        args, 'sqlite', {'db_path': args.db_path}
    )

    logger.info('Making tfidf vectors...')
    tfidf = build_tfidf.get_tfidf_matrix(count_matrix)

    logger.info('Getting word-doc frequencies...')
    freqs = build_tfidf.get_doc_freqs(count_matrix)

    # Save to disk
    build_tfidf.save_tfidf(args, tfidf, freqs)

    logger.info('Done.')
