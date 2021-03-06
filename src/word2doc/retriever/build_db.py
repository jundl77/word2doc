#!/usr/bin/env python3
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""A script to read in and store documents in a sqlite database."""

import importlib.util
import json
import os
import sqlite3
from multiprocessing import Pool as ProcessPool

from tqdm import tqdm

from . import utils
from word2doc.util import logger
from word2doc.util import constants

logger = logger.get_logger()


# ------------------------------------------------------------------------------
# Import helper
# ------------------------------------------------------------------------------


PREPROCESS_FN = None


def init(filename):
    global PREPROCESS_FN
    if filename:
        PREPROCESS_FN = import_module(filename).preprocess


def import_module(filename):
    """Import a module given a full path to the file."""
    spec = importlib.util.spec_from_file_location('doc_filter', filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------------------
# Store corpus.
# ------------------------------------------------------------------------------


def iter_files(path):
    """Walk through all files located under a root path."""
    if os.path.isfile(path):
        yield path
    elif os.path.isdir(path):
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                yield os.path.join(dirpath, f)
    else:
        raise RuntimeError('Path %s is invalid' % path)


def get_contents(filename):
    """Parse the contents of a file. Each line is a JSON encoded document."""
    global PREPROCESS_FN
    documents = []

    with open(filename, encoding='latin1') as f:
        for line in f:
            # Parse document

            doc = dict()
            try:
                doc = json.loads(line)
            except ValueError:
                logger.error('Unable to parse: ' + filename) # For files like OSXs .DS_Store
                continue

            # Handle references
            if 'references' not in doc:
                doc['references'] = []

            # Maybe preprocess the document with custom function
            if PREPROCESS_FN:
                doc = PREPROCESS_FN(doc)

            # Skip if it is empty or None
            if not doc:
                continue

            # Add the document
            documents.append((utils.normalize(doc['id']), doc['url'], json.dumps(doc['references']), doc['text']))
    return documents


def store_contents(data_path, save_path, preprocess):
    """Preprocess and store a corpus of documents in sqlite.
    Args:
        data_path: Root path to directory (or directory of directories) of files
          containing json encoded documents (must have `id` and `text` fields).
        save_path: Path to output sqlite db.
        preprocess: Path to file defining a custom `preprocess` function. Takes
          in and outputs a structured doc.
    """

    num_workers = constants.get_number_workers()

    logger.info('Reading into database...')
    conn = sqlite3.connect(save_path)
    c = conn.cursor()
    c.execute("CREATE TABLE documents (id PRIMARY KEY, url, refs, text);")

    workers = ProcessPool(num_workers, initializer=init, initargs=(preprocess,))
    files = [f for f in iter_files(data_path)]
    count = 0
    with tqdm(total=len(files)) as pbar:
        for elements in tqdm(workers.imap_unordered(get_contents, files)):
            count += len(elements)
            c.executemany("INSERT INTO documents VALUES (?,?,?,?)", elements)
            pbar.update()
    logger.info('Read %d docs.' % count)
    logger.info('Committing...')
    conn.commit()
    conn.close()
