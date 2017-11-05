#!/usr/bin/env python3
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Initialize the entire project."""

import os
from word2doc.util import constants


def init():
    init_constants()
    init_file_structure()


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
