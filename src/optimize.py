#!/usr/bin/env python3
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import locale

from word2doc.util import constants
from word2doc.util import logger
from word2doc.util import init_project
from word2doc.optimizer import pre_process
from word2doc.model import Model
from word2doc.optimizer.net.train import OptimizerNet
from word2doc.optimizer.net.word2doc import Word2Doc

logger = logger.get_logger()


def pre_process_loop(args, pre_processor):
    #loop_interval = int(args.num_bins / args.num_workers)
    #loop_offset = (args.bin_id - 1) * loop_interval
    #start_bin = 1 + loop_offset
    #end_bin = loop_interval + loop_offset

    #for i in range(start_bin, end_bin + 1):
    #    pre_processor.pre_process(i, args.num_bins)

    if args.bin_id == 1:
        pre_processor.pre_process(10, args.num_bins)
    elif args.bin_id == 9:
        pre_processor.pre_process(13, args.num_bins)


def handle_model_type(args):
    if args.model_type == 'squad':
        # Pre-processor
        if args.model_action == 'create-bins':
            pre_processor = pre_process.SquadPreprocessor(None)
            pre_processor.create_bins(args.num_bins)
        elif args.model_action == 'pre-process':
            m = Model(constants.get_db_path(), constants.get_retriever_model_path())
            pre_processor = pre_process.SquadPreprocessor(m)
            pre_processor.pre_process(args.path, args.bin_id)
        elif args.model_action == 'merge-bins':
            pre_processor = pre_process.SquadPreprocessor(None)
            pre_processor.merge_bins(args.num_bins)

        # Neural net
        elif args.model_action == 'train':
            net = OptimizerNet()
            net.train()

    elif args.model_type == 'word2doc':
        # Pre-processor
        if args.model_action == 'pre-process':
            pre_processor = pre_process.Word2DocPreprocessor()
            pre_process_loop(args, pre_processor)
        elif args.model_action == 'merge-bins':
            pre_processor = pre_process.Word2DocPreprocessor()
            pre_processor.merge_bins(args.num_bins)

        # Neural net
        elif args.model_action == 'train':
            net = Word2Doc()
            net.train()
        elif args.model_action == 'eval':
            net = Word2Doc()
            net.eval()


# ------------------------------------------------------------------------------
# Data pipeline that builds the processes the data for the retriever.
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_type', type=str, default=None,
                        help='squad, word2doc')
    parser.add_argument('model_action', type=str, default=None,
                        help='create-bins, pre-process, merge-bins, train')
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
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

    handle_model_type(args)



