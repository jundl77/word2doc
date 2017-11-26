#!/usr/bin/env python3
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from word2doc.retriever.doc_db import DocDB

from word2doc.util import logger

logger = logger.get_logger()

MIN_PP_LENGTH = 100


class LabelExtractor:

    def __init__(self, db_path=None):
        self.doc_db = DocDB(db_path)

    def extract_label(self, doc_id):
        """Extract the brief summary definition from a wikipedia article (which is the first paragraph)"""
        text = self.doc_db.get_doc_text(doc_id)
        p_array = text.split('\n\n')

        return self.__first_paragraph(p_array, 0)

    def __first_paragraph(self, p_array, index):
        """Sometimes first paragraph is title or subtitle etc. Make sure we get the first "true" paragraph"""

        if index >= len(p_array):
            return p_array[1].split('.')[0]

        if len(p_array[index]) < MIN_PP_LENGTH:
            return self.__first_paragraph(p_array, index + 1)

        # Get first sentence only
        return p_array[index].split('.')[0]






