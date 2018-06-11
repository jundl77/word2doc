#!/usr/bin/env python3
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Interactive mode for the tfidf retriever module."""

import argparse
import numpy as np
from tqdm import tqdm

from word2doc import model
from word2doc.util import logger
from word2doc.util import init_project

logger = logger.get_logger()

parser = argparse.ArgumentParser()
parser.add_argument('db_path', type=str, help='/path/to/db')
parser.add_argument('--model', type=str, default=None)
args = parser.parse_args()


# ------------------------------------------------------------------------------
# Drop in to interactive
# ------------------------------------------------------------------------------

MODEL = None

top10_accuracy = 0
top1_accuracy = 0
total = 0


def init(db_path, model_path):
    global MODEL
    init_project.init(1)
    MODEL = model.Model(db_path, model_path)


def calculate_rankings(query, title):
    global MODEL, top10_accuracy, top1_accuracy, total

    scores, doc = MODEL.calculate_rankings(query, 10)

    first = True
    for t, s in scores.items():
        if t == title and first:
            top1_accuracy += 1
            top10_accuracy += 1
            first = False
        elif t == title:
            top10_accuracy += 1

    total += 1


init(args.db_path, args.model)
test_data = np.load("data/word2doc/word2doc-test-bin-3.npy")

with tqdm(total=len(test_data)) as pbar:
    for d in tqdm(test_data):
        calculate_rankings(d["query"], d["doc_title"])
        pbar.update()

print("Done")
result_10 = float(top10_accuracy) / float(total)
result_1 = float(top1_accuracy) / float(total)

print("Top 1 accuracy:" + str(result_1))
print("Top 10 accuracy:" + str(result_10))
