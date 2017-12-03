#!/usr/bin/env python3
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import math
import os

from word2doc.wikiextractor import wiki_extractor
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
    parser.add_argument('data_path', type=str, help='/path/to/wikidump')
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
    init_project.init(args.num_workers)
    save_path = constants.get_db_path()

    # Extract text from wikipedia dump
    wiki_extractor.extract_wiki(args.data_path, output=constants.get_wiki_extract_path(), json=True, references=True)

    # Build database if it does not already exist
    if not os.path.isfile(save_path):
        logger.info('No database found. Building database...')
        build_db.store_contents(args.data_path, save_path, args.preprocess)
    else:
        logger.info('Existing database found. Using database.')

    # Calculate tfidf data
    logger.info('Counting words...')
    count_matrix, doc_dict = build_tfidf.get_count_matrix(
        args, 'sqlite', {'db_path': save_path}
    )

    logger.info('Making tfidf vectors...')
    tfidf = build_tfidf.get_tfidf_matrix(count_matrix)

    logger.info('Getting word-doc frequencies...')
    freqs = build_tfidf.get_doc_freqs(count_matrix)

    # Save to disk
    build_tfidf.save_tfidf(args, tfidf, freqs, doc_dict)

    logger.info('Done.')
