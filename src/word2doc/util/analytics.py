#!/usr/bin/env python3
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json

from bashplotlib.histogram import plot_hist
from word2doc.util import constants


class Analytics:

    def __init__(self):
        self.rg_changes = []

    def reference_graph_analytics(self, original_docs, filtered_docs):
        self.rg_changes.append(len(original_docs) - len(filtered_docs))

    def save_to_file(self):

        data = {
            'reference_graph_changes': self.rg_changes
        }

        with open(constants.get_analytics_path(), 'w') as fp:
            json.dump(data, fp, sort_keys=True, indent=4)

    def report(self):
        print('\n' * 2)
        print('=' * 80)
        print('=' + ' ' * 34 + 'ANALYTICS' + ' ' * 35 + '=')
        print('=' * 80)
        print('\n' * 1)

        plot_hist(self.rg_changes,
                  height=10,
                  title='Docs filtered out based on reference',
                  xlab=True,
                  showSummary=True)

