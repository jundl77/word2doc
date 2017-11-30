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
ANALYTICS_FILE_NAME = 'analytics.json'
N_WORKERS = 1

# Wikipedia inverted index
DOCS_DB_NAME = 'docs.db'
RETRIEVER_MODEL_NAME = 'r_model.npz'
RETRIEVER_MODEL_META_NAME = 'r_model_meta.json'

# Optimizer data
OPTIMIZER_DATA_NAME = 'opt_data.npz'

# SQuAD data
SQUAD_DIR_NAME = 'squad'
SQUAD_DEV_NAME = 'dev-v1.1.json'
SQUAD_TRAIN_NAME = 'train-v1.1.json'


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
# Analytics paths
# ------------------------------------------------------------------------------

def get_analytics_path():
    return os.path.join(get_logs_dir(), ANALYTICS_FILE_NAME)


# ------------------------------------------------------------------------------
# Retriever paths
# ------------------------------------------------------------------------------

def get_db_path():
    return os.path.join(get_data_dir(), DOCS_DB_NAME)


def get_retriever_model_path():
    return os.path.join(get_data_dir(), RETRIEVER_MODEL_NAME)


def get_retriever_model_meta_path():
    return os.path.join(get_data_dir(), RETRIEVER_MODEL_META_NAME)


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
