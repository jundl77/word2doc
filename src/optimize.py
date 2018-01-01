#!/usr/bin/env python3
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse

from word2doc.util import constants
from word2doc.util import logger
from word2doc.util import init_project
from word2doc.optimizer import pre_process
from word2doc.model import Model
from word2doc.optimizer.net.train import OptimizerNet

logger = logger.get_logger()


# ------------------------------------------------------------------------------
# Data pipeline that builds the processes the data for the retriever.
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('type', type=str, default=None,
                        help='create-bins, do-squad, merge-bins')
    parser.add_argument('--path', type=str, default=None,
                        help='path to the data to process')
    parser.add_argument('--bin-id', type=int, default=None,
                        help='Id of the bin to process')
    parser.add_argument('--num-bins', type=int, default=None,
                        help='Number of bins')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of CPU processes')
    args = parser.parse_args()

    # Init project
    init_project.init(args.num_workers)

    # Pre-processor
    if args.type == 'create-bins':
        pre_processor = pre_process.OptimizerPreprocessor(None)
        pre_processor.create_bins(args.num_bins)
    elif args.type == 'do-squad':
        m = Model(constants.get_db_path(), constants.get_retriever_model_path())
        pre_processor = pre_process.OptimizerPreprocessor(m)
        pre_processor.pre_process_squad(args.path, args.bin_id)
    elif args.type == 'merge-bins':
        pre_processor = pre_process.OptimizerPreprocessor(None)
        pre_processor.merge_bins(args.num_bins)

    # Neural net
    elif args.type == 'train':
        net = OptimizerNet()
        net.train()

