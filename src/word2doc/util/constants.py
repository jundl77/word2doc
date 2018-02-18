#!/usr/bin/env python3
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Saves constants"""

import os

# General
PROJECT_ROOT_DIR = ''
DATA_DIR_NAME = 'data'
LOGS_DIR_NAME = 'logs'
N_WORKERS = 1

# Wikipedia extraction path
WIKI_EXTRACT_NAME = 'wiki'

# Wikipedia inverted index
DOCS_DB_NAME = 'docs.db'
RETRIEVER_MODEL_NAME = 'r_model.npz'
RETRIEVER_MODEL_META_NAME = 'r_model_meta.json'

# Optimizer data
OPTIMIZER_DATA_NAME = 'opt_data.npz'

# TensorFlow
TENSORBOARD_NAME = 'tensorboard'

# SQuAD data
SQUAD_DIR_NAME = 'squad'
SQUAD_DEV_NAME = 'dev-v1.1.json'
SQUAD_TRAIN_NAME = 'train-v1.1.json'
SQUAD_DEV_QUERIES_NAME = 'dev-v1.1-queries.npy'
SQUAD_TRAIN_QUERIES_NAME = 'train-v1.1-queries.npy'

# Word2Doc data
WORD2DOC_DIR_NAME = 'word2doc'

# GloVe dir
GLOVE_DIR_PATH = 'GloVe'
GLOVE_840B_300D_NAME = 'glove.840B.300d.txt'

# Infersent data
INFERSENT_DIR_NAME = 'infersent'
INFERSENT_MODEL_NAME = 'infersent.allnli.pickle'


# ------------------------------------------------------------------------------
# Number of workers
# ------------------------------------------------------------------------------

def get_number_workers():
    return N_WORKERS


def set_number_workers(n):
    global N_WORKERS
    N_WORKERS = n
    return N_WORKERS


# ------------------------------------------------------------------------------
# General paths
# ------------------------------------------------------------------------------

def set_root_dir(path):
    global PROJECT_ROOT_DIR
    PROJECT_ROOT_DIR = path


def get_root_dir():
    return PROJECT_ROOT_DIR


def get_data_dir():
    return os.path.join(get_root_dir(), DATA_DIR_NAME)


def get_logs_dir():
    return os.path.join(get_root_dir(), LOGS_DIR_NAME)


# ------------------------------------------------------------------------------
# Logging paths
# ------------------------------------------------------------------------------

def get_tensorboard_path():
    return os.path.join(get_logs_dir(), TENSORBOARD_NAME)


# ------------------------------------------------------------------------------
# Data-set paths
# ------------------------------------------------------------------------------

def get_wiki_extract_path():
    return os.path.join(get_data_dir(), WIKI_EXTRACT_NAME)


def get_db_path():
    return os.path.join(get_data_dir(), DOCS_DB_NAME)


def get_retriever_model_path():
    return os.path.join(get_data_dir(), RETRIEVER_MODEL_NAME)


def get_retriever_model_meta_path():
    return os.path.join(get_data_dir(), RETRIEVER_MODEL_META_NAME)


def get_glove_dir():
    return os.path.join(get_data_dir(), GLOVE_DIR_PATH)


def get_glove_840b_300d_path():
    return os.path.join(get_glove_dir(), GLOVE_840B_300D_NAME)


def get_infersent_dir_path():
    return os.path.join(get_data_dir(), INFERSENT_DIR_NAME)


def get_infersent_model_path():
    return os.path.join(get_infersent_dir_path(), INFERSENT_MODEL_NAME)


# ------------------------------------------------------------------------------
# Optimizer paths
# ------------------------------------------------------------------------------

def get_optimizer_data_path():
    return os.path.join(get_data_dir(), OPTIMIZER_DATA_NAME)


def get_squad_dir():
    return os.path.join(get_data_dir(), SQUAD_DIR_NAME)


def get_squad_dev_path():
    return os.path.join(get_squad_dir(), SQUAD_DEV_NAME)


def get_squad_train_path():
    return os.path.join(get_squad_dir(), SQUAD_TRAIN_NAME)


def get_squad_dev_queries_path():
    return os.path.join(get_squad_dir(), SQUAD_DEV_QUERIES_NAME)


def get_squad_train_queries_path():
    return os.path.join(get_squad_dir(), SQUAD_TRAIN_QUERIES_NAME)


def get_word2doc_dir():
    return os.path.join(get_data_dir(), WORD2DOC_DIR_NAME)
