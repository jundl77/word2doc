#!/usr/bin/env python3
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from word2doc.util import constants
from logging.handlers import TimedRotatingFileHandler

# Set up logger
log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
root_logger = logging.getLogger("Rotating Log")

file_handler = TimedRotatingFileHandler("{0}/{1}.log".format(constants.get_logs_dir(), 'word2doc'), when="d", interval=1, backupCount=100)
file_handler.setFormatter(log_formatter)
root_logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
root_logger.addHandler(console_handler)
root_logger.setLevel(logging.INFO)


def get_logger():
    return root_logger
