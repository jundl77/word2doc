#!/usr/bin/env python3
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""A script to build the tf-idf document matrices for retrieval."""

import os.path
from collections import Counter
from functools import partial
from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize

import numpy as np
import scipy.sparse as sp

from word2doc import retriever
from word2doc import tokenizers
from word2doc.util import logger

logger = logger.get_logger()

# ------------------------------------------------------------------------------
# Multiprocessing functions
# ------------------------------------------------------------------------------

DOC2IDX = None
PROCESS_TOK = None
PROCESS_DB = None


def init(tokenizer_class, db_class, db_opts):
    global PROCESS_TOK, PROCESS_DB
    PROCESS_TOK = tokenizer_class()
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)
    PROCESS_DB = db_class(**db_opts)
    Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)


def fetch_text(doc_id):
    global PROCESS_DB
    return PROCESS_DB.get_doc_text(doc_id)


def tokenize(text):
    global PROCESS_TOK
    return PROCESS_TOK.tokenize(text)


# ------------------------------------------------------------------------------
# Build article --> word count sparse matrix.
# ------------------------------------------------------------------------------


def count(ngram, hash_size, doc_id):
    """Fetch the text of a document and compute hashed ngrams counts."""
    global DOC2IDX
    row, col, data = [], [], []
    # Tokenize
    tokens = tokenize(retriever.utils.normalize(fetch_text(doc_id)))

    # Get ngrams from tokens, with stopword/punctuation filtering.
    ngrams = tokens.ngrams(n=ngram, uncased=True, filter_fn=retriever.utils.filter_ngram)

    # Hash ngrams and count occurences
    counts = Counter([retriever.utils.hash(gram, hash_size) for gram in ngrams])

    # Return in sparse matrix data format.
    row.extend(counts.keys())
    col.extend([DOC2IDX[doc_id]] * len(counts))
    data.extend(counts.values())
    return row, col, data


def get_count_matrix(args, db, db_opts):
    """Form a sparse word to document count matrix (inverted index).
    M[i, j] = # times word i appears in document j.
    """
    # Map doc_ids to indexes
    global DOC2IDX
    db_class = retriever.get_class(db)
    with db_class(**db_opts) as doc_db:
        doc_ids = doc_db.get_doc_ids()
    DOC2IDX = {doc_id: i for i, doc_id in enumerate(doc_ids)}

    # Setup worker pool
    tok_class = tokenizers.get_class(args.tokenizer)
    workers = ProcessPool(
        args.num_workers,
        initializer=init,
        initargs=(tok_class, db_class, db_opts)
    )

    # Compute the count matrix in steps (to keep in memory)
    #
    # row = doc keys
    # col = doc ids
    # data = hashed ngrams
    logger.info('Mapping...')
    row, col, data = [], [], []

    # Check if a mapping is present, else calculate it
    mapping_present = False
    base_dir = '/Users/julianbrendl/Projects/bachelor-thesis/word2doc/data/wikipedia/temp' # TODO: fix paths
    filename = 'docs.npz'
    mapping_path = os.path.join(base_dir, filename)

    if os.path.isfile(mapping_path):
        logger.info('Exisiting mapping found, loading into memory..')
        mapping_present = True
        row, col, data = retriever.utils.load_mapping(mapping_path)
        logger.info('Loaded mapping into memory.')
    else:
        logger.info('No exisiting mapping found, creating new mapping..')
        step = max(int(len(doc_ids) / 10), 1)
        batches = [doc_ids[i:i + step] for i in range(0, len(doc_ids), step)]
        _count = partial(count, args.ngram, args.hash_size)
        for i, batch in enumerate(batches):
            logger.info('-' * 25 + 'Batch %d/%d' % (i + 1, len(batches)) + '-' * 25)
            for b_row, b_col, b_data in workers.imap_unordered(_count, batch):
                row.extend(b_row)
                col.extend(b_col)
                data.extend(b_data)
        workers.close()
        workers.join()
        logger.info('Creating new mapping.')

    # Save mapping if none was found before
    if not mapping_present:
        logger.info('Saving mapping..')
        retriever.utils.save_mapping(base_dir, filename, row, col, data)
        logger.info('Saved.')

    logger.info('Creating sparse matrix...')
    count_matrix = sp.csr_matrix(
        (data, (row, col)), shape=(args.hash_size, len(doc_ids))
    )
    count_matrix.sum_duplicates()
    return count_matrix, (DOC2IDX, doc_ids)


# ------------------------------------------------------------------------------
# Transform count matrix to different forms.
# ------------------------------------------------------------------------------


def get_tfidf_matrix(cnts):
    """Convert the word count matrix into tfidf one.
    tfidf = log(tf + 1) * log((N - Nt + 0.5) / (Nt + 0.5))
    * tf = term frequency in document
    * N = number of documents
    * Nt = number of occurences of term in all documents
    """
    Ns = get_doc_freqs(cnts)
    idfs = np.log((cnts.shape[1] - Ns + 0.5) / (Ns + 0.5))
    idfs[idfs < 0] = 0
    idfs = sp.diags(idfs, 0)
    tfs = cnts.log1p()
    tfidfs = idfs.dot(tfs)
    return tfidfs


def get_doc_freqs(cnts):
    """Return word --> # of docs it appears in."""
    binary = (cnts > 0).astype(int)
    freqs = np.array(binary.sum(1)).squeeze()
    return freqs
