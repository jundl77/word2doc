#!/usr/bin/env python3
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Saves constants"""

import os

PROJECT_ROOT_DIR = ""


def set_root_dir(path):
    global PROJECT_ROOT_DIR
    PROJECT_ROOT_DIR = path


def get_root_dir():
    global PROJECT_ROOT_DIR
    return PROJECT_ROOT_DIR


def get_data_dir():
    return os.path.join(get_root_dir(), 'data')


def get_db_path():
    return os.path.join(get_data_dir(), 'docs.db')

