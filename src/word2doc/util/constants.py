#!/usr/bin/env python3
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Saves constants"""

import os

PROJECT_ROOT_DIR = ''
DATA_DIR_NAME = 'data'
DOCS_DB_NAME = 'docs.db'
RETRIEVER_MODEL_NAME = 'r_model.npz'
RETRIEVER_MODEL_META_NAME = 'r_model_meta.json'


def set_root_dir(path):
    global PROJECT_ROOT_DIR
    PROJECT_ROOT_DIR = path


def get_root_dir():
    global PROJECT_ROOT_DIR
    return PROJECT_ROOT_DIR


def get_data_dir():
    return os.path.join(get_root_dir(), DATA_DIR_NAME)


def get_db_path():
    return os.path.join(get_data_dir(), DOCS_DB_NAME)


def get_retriever_model_path():
    return os.path.join(get_data_dir(), RETRIEVER_MODEL_NAME)


def get_retriever_model_meta_path():
    return os.path.join(get_data_dir(), RETRIEVER_MODEL_META_NAME)
