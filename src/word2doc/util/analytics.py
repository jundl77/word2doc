#!/usr/bin/env python3
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import json

from bashplotlib.histogram import plot_hist
from word2doc.util import constants


class Analytics:

    def __init__(self):
        # Changes through
        self.rg_changes = []

        # Number of times a query was stopped from fully executing because the near perfect answer was found
        self.pp_query_preemption_count = -1

        # Calculates for how many queries the label was correctly found using the document retriever
        self.n_pp_queries = -1
        self.n_pp_queries_error = -1
        self.n_pp_queries_duplicate = -1
        self.pp_queries_accuracy = -1

    def preempted_search(self):
        self.pp_query_preemption_count += 1

    def queries_processed(self, total, n_duplicate, n_error):
        self.n_pp_queries = total
        self.n_pp_queries_error = n_error
        self.n_pp_queries_duplicate = n_duplicate
        self.pp_queries_accuracy = float(total - n_duplicate - n_error) / float(total - n_duplicate)

    def reference_graph_analytics(self, original_docs, filtered_docs):
        self.rg_changes.append(len(original_docs) - len(filtered_docs))

    def save_to_file(self, file_name):

        data = {
            'reference_graph_changes': self.rg_changes,
            'pp_query_preemption_count': self.pp_query_preemption_count,
            'n_pp_queries': self.n_pp_queries,
            'n_pp_queries_duplicate': self.n_pp_queries_duplicate,
            'n_pp_queries_error': self.n_pp_queries_error,
            'pp_queries_accuracy': self.pp_queries_accuracy
        }

        with open(os.path.join(constants.get_logs_dir(), file_name + '.json'), 'w') as fp:
            json.dump(data, fp, sort_keys=True, indent=4)

    def report(self):
        print('\n' * 2)
        print('=' * 80)
        print('=' + ' ' * 34 + 'ANALYTICS' + ' ' * 35 + '=')
        print('=' * 80)
        print('\n' * 1)

        print('Preemption count: ' + str(self.pp_query_preemption_count))
        print('\n')
        plot_hist(self.rg_changes,
                  height=10,
                  title='Docs filtered out based on reference',
                  xlab=True,
                  showSummary=True)

