#!/usr/bin/env python3
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Initialize the entire project."""

import os
import sys
import locale

from word2doc.util import constants


def init(num_workers):
    init_constants()
    init_file_structure()
    init_sys_path()

    if num_workers is None:
        num_workers = 1

    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    constants.set_number_workers(num_workers)


# ------------------------------------------------------------------------------
# Init constants
# ------------------------------------------------------------------------------


def init_constants():
    init_project_root()


def init_project_root():
    cur_dir = os.path.dirname(os.path.abspath(__file__))

    # Set to word2doc directory
    root_dir = os.path.abspath(os.path.join(cur_dir, os.pardir))

    # Set to src directory
    root_dir = os.path.abspath(os.path.join(root_dir, os.pardir))

    # Set to project root
    root_dir = os.path.abspath(os.path.join(root_dir, os.pardir))

    constants.set_root_dir(root_dir)


# ------------------------------------------------------------------------------
# Init file structure
# ------------------------------------------------------------------------------


def init_file_structure():
    # Create data folder
    if not os.path.exists(constants.get_data_dir()):
        os.makedirs(constants.get_data_dir())

    # Create logs folder
    if not os.path.exists(constants.get_logs_dir()):
        os.makedirs(constants.get_logs_dir())

    # Create TensorBoard folder
    if not os.path.exists(constants.get_tensorboard_path()):
        os.makedirs(constants.get_tensorboard_path())


# ------------------------------------------------------------------------------
# Init file structure
# ------------------------------------------------------------------------------


def init_sys_path():
    sys.path.append(os.path.join(constants.get_root_dir(), 'src/word2doc/embeddings/infersent/'))
