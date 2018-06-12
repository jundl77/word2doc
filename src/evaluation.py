#!/usr/bin/env python3
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Interactive mode for the tfidf retriever module."""

import argparse
import numpy as np
from tqdm import tqdm
from random import randint
import os


from word2doc import model
from word2doc.util import logger
from word2doc.util import constants
from word2doc.util import init_project
from word2doc.labels.extractor import LabelExtractor
from word2doc.optimizer.net.word2doc import Word2Doc
from word2doc.retriever.doc_db import DocDB

logger = logger.get_logger()

parser = argparse.ArgumentParser()
parser.add_argument('db_path', type=str, help='/path/to/db')
parser.add_argument('--model', type=str, default=None)
args = parser.parse_args()


MODEL = None

top10_accuracy = 0
top5_accuracy = 0
top1_accuracy = 0
total = 0

APS = 0
doc_db = DocDB(constants.get_db_path())
doc_titles = doc_db.get_doc_ids()

w2d = Word2Doc()


def init(db_path, model_path):
    global MODEL
    init_project.init(1)
    MODEL = model.Model(db_path, model_path)


def calculate_rankings_accuracy_fb(query, title):
    global MODEL, top10_accuracy, top5_accuracy, top1_accuracy, total

    scores, doc = MODEL.calculate_rankings(query, 5)

    first = True
    i = 0
    for t, s in scores.items():
        if t == title and first:
            top1_accuracy += 1
            top5_accuracy += 1
            top10_accuracy += 1
        elif t == title and i < 5:
            top5_accuracy += 1
            top10_accuracy += 1
        elif t == title:
            top10_accuracy += 1

        first = False
        i += 1

    total += 1


def calculate_rankings_accuracy_word2doc(embb, ctx, titles, target_title):
    global MODEL, top10_accuracy, top5_accuracy, top1_accuracy, total

    docs = w2d.predict([embb], [ctx])
    docs = list(map(lambda x: titles[x], docs[0]))

    first = True
    i = 0
    for t in docs:
        if t == target_title and first:
            top1_accuracy += 1
            top5_accuracy += 1
            top10_accuracy += 1
        elif t == target_title and i < 5:
            top5_accuracy += 1
            top10_accuracy += 1
        elif t == target_title:
            top10_accuracy += 1

        first = False
        i += 1

    total += 1


def calculate_rankings_map_word2doc(query, label, embb, ctx, titles):
    global MODEL, APS, total, w2d
    extractor = LabelExtractor(constants.get_db_path())

    print("Run no. " + str(total))

    docs = w2d.predict([embb], [ctx])
    docs = list(map(lambda x: titles[x], docs[0]))

    ap = list()

    for t in docs:
        print(u' '.join(("QUERY: ", query)).encode('utf-8'))
        print(u' '.join(("LABEL: ", label)).encode('utf-8'))
        print(u' '.join(("TEST: ", t)).encode('utf-8'))
        try:
            intro_par = extractor.extract_label(t)
        except:
            intro_par = "ERROR"

        print("Intro paragraph:")
        print(intro_par)
        print("\n")

        if t == label:
            print("-------------- Found label. ------------------")
            ap.append(calculate_ap(ap, 1))
        else:
            valid = False
            while not valid:
                rel = input("Is the document relevant? (y\\n)")
                if rel == "y":
                    ap.append(calculate_ap(ap, 1))
                    valid = True
                elif rel == "n":
                    ap.append(calculate_ap(ap, 0))
                    valid = True
                else:
                    print("Invalid input.")

    APS += float(sum(ap)) / float(len(ap))
    total += 1


def calculate_rankings_map_fb(query, label):
    global MODEL, APS, total
    extractor = LabelExtractor(constants.get_db_path())

    print("Run no. " + str(total))

    scores, doc = MODEL.calculate_rankings(query, 5)

    ap = list()

    for t, s in scores.items():
        print(u' '.join(("QUERY: ", query)).encode('utf-8'))
        print(u' '.join(("LABEL: ", label)).encode('utf-8'))
        print(u' '.join(("TEST: ", t)).encode('utf-8'))
        try:
            intro_par = extractor.extract_label(t)
        except:
            intro_par = "ERROR"

        print("Intro paragraph:")
        print(intro_par)
        print("\n")

        if t == label:
            print("-------------- Found label.motion-next) ------------------")
            ap.append(calculate_ap(ap, 1))
        else:
            valid = False
            while not valid:
                rel = input("Is the document relevant? (y\\n)")
                if rel == "y":
                    ap.append(calculate_ap(ap, 1))
                    valid = True
                elif rel == "n":
                    ap.append(calculate_ap(ap, 0))
                    valid = True
                else:
                    print("Invalid input.")

    APS += float(sum(ap)) / float(len(ap))
    total += 1


def calculate_ap(prev_ap, rel):
    if len(prev_ap) == 0:
        return rel

    result = sum(prev_ap) + rel
    return float(result) / float(len(prev_ap) + 1)


def test_map_w2d():
    global APS, total, w2d
    init(args.db_path, args.model)

    base_train_data = "72t-wpp.npy"
    test_data_path = "word2doc-test-400_normal.npy"

    test_data = np.load(os.path.join(constants.get_word2doc_dir(), test_data_path))

    target, embeddings, context, titles = w2d.load_train_data(os.path.join(constants.get_word2doc_dir(), base_train_data))
    target, embeddings, context_test, titles_test = w2d.load_test_data(os.path.join(constants.get_word2doc_dir(), test_data_path))
    context = w2d.normalize_test_context(context, context_test)
    embeddings, target, context = w2d.filter_test_data(embeddings, target, context)
    target = w2d.normalize_test_labels(target, titles, titles_test)

    for emb, tar, ctx in zip(embeddings, target, context):
        d = list(filter(lambda x: (x['pivot_embeddings'] == emb).all(), test_data))[0]

        if not total == 0:
            print("Current MAP: " + str(float(APS) / float(total)))
        calculate_rankings_map_word2doc(d['query'], d['doc_title'], emb, ctx, titles)

    print("FINAL MAP SCORE: " + str(float(APS) / float(total)))


def test_accuracy_w2d():
    global top10_accuracy, top5_accuracy, top1_accuracy, total, w2d

    init(args.db_path, args.model)

    base_train_data = "72t-wpp.npy"
    test_data_path = "word2doc-test-400_normal.npy"

    test_data = np.load(os.path.join(constants.get_word2doc_dir(), test_data_path))

    target, embeddings, context, titles = w2d.load_train_data(os.path.join(constants.get_word2doc_dir(), base_train_data))
    target, embeddings, context_test, titles_test = w2d.load_test_data(os.path.join(constants.get_word2doc_dir(), test_data_path))
    context = w2d.normalize_test_context(context, context_test)
    embeddings, target, context = w2d.filter_test_data(embeddings, target, context)
    target = w2d.normalize_test_labels(target, titles, titles_test)

    for emb, tar, ctx in zip(embeddings, target, context):
        d = list(filter(lambda x: (x['pivot_embeddings'] == emb).all(), test_data))[0]
        calculate_rankings_accuracy_word2doc(emb, ctx, titles, d["doc_title"])

    print("Done")
    result_10 = float(top10_accuracy) / float(total)
    result_5 = float(top5_accuracy) / float(total)
    result_1 = float(top1_accuracy) / float(total)

    print("Top 1 accuracy:" + str(result_1))
    print("Top 5 accuracy:" + str(result_5))
    print("Top 10 accuracy:" + str(result_10))


def test_map_fb():
    global APS, total
    init(args.db_path, args.model)
    test_data = np.load("data/word2doc/word2doc-test-400_numb.npy")

    for d in test_data:
        if not total == 0:
            print("Current MAP: " + str(float(APS) / float(total)))
        calculate_rankings_map_fb(d["query"], d["doc_title"])

    print("FINAL MAP SCORE: " + str(float(APS) / float(total)))


def test_accuracy_fb():
    global top10_accuracy, top5_accuracy, top1_accuracy, total

    init(args.db_path, args.model)
    test_data = np.load("data/word2doc/word2doc-test-400_normal.npy")

    with tqdm(total=len(test_data)) as pbar:
        for d in tqdm(test_data):
            calculate_rankings_accuracy_fb(d["query"], d["doc_title"])
            pbar.update()

    print("Done")
    result_10 = float(top10_accuracy) / float(total)
    result_5 = float(top5_accuracy) / float(total)
    result_1 = float(top1_accuracy) / float(total)

    print("Top 1 accuracy:" + str(result_1))
    print("Top 5 accuracy:" + str(result_5))
    print("Top 10 accuracy:" + str(result_10))


#test_accuracy_w2d()
test_map_w2d()