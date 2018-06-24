#!/usr/bin/env python3
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Interactive mode for the tfidf retriever module."""

import argparse
import code
import prettytable
import locale

from word2doc import model
from word2doc.util import logger
from word2doc.util import init_project

logger = logger.get_logger()
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

parser = argparse.ArgumentParser()
parser.add_argument('db_path', type=str, help='/path/to/db')
parser.add_argument('--model', type=str, default=None)
args = parser.parse_args()


# ------------------------------------------------------------------------------
# Drop in to interactive
# ------------------------------------------------------------------------------

MODEL = None


def init(db_path, model_path):
   global MODEL
   init_project.init(1)
   MODEL = model.Model(db_path, model_path)


def calculate_rankings(query, k=10):
    global MODEL

    scores, doc = MODEL.calculate_rankings(query, k)
    table = prettytable.PrettyTable(
        ['Doc Id', 'Doc Score', 'Title Embedding Score', 'Keyword Embedding Score']
    )

    for t, s in scores.items():
        table.add_row([t, '%.5g' % s[0], s[1], s[2]])
    print(table)

    if doc is not None:
        print("Preempted search because found matching title.")
        print("Doc retrieved: " + doc)

    analytics = MODEL.get_analytics()
    #analytics.save_to_file()
    analytics.report()


banner = """
Interactive TF-IDF Retriever
>> calculate_rankings(question, k=10)
>> usage()
"""


def usage():
    print(banner)


init(args.db_path, args.model)
code.interact(banner=banner, local=locals())