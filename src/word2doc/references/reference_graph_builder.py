#!/usr/bin/env python3
# Copyright 2017-present
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import requests
import json
import re
from word2doc.references.reference_node import ReferenceNode


class ReferencesGraphBuilder:
    """Build a reference tree for a given set of documents.
       That means, find out what documents are referenced in the other documents"""

    def __init__(self):
        self.graph = ReferenceNode('#root')

    def build_references_graph(self, doc_titles):
        """Build reference graph for all documents in doc_titles"""

        for title in doc_titles:
            references = self.__extract_references(title)

            if self.graph.get_distant_child(title) is None:
                node = ReferenceNode(title)
                self.__distribute_references(node, references)
                self.graph.add_child(node)
            else:
                node = self.graph.get_distant_child(title)
                self.__distribute_references(node, references)

        return self.graph

    def __distribute_references(self, node, references):
        """Distribute references across nodes"""
        for ref in references:
            if self.graph.get_distant_child(ref) is None:
                child = ReferenceNode(ref)
            else:
                child = self.graph.get_distant_child(ref)
                if child.get_title() in self.graph.get_children() and child.get_distant_child(node.get_title()) is None:
                    self.graph.remove_child(child.get_title())

            node.add_child(child)

    def __extract_references(self, title):
        """Find out what references are in the document with the title 'title'"""
        text = self.__get_text(title)
        doc_regex = r'({{Main( article)?(\|[\w ]+)+}})'
        ref_regex = r'(\|([\w ]+))'
        matches = re.findall(doc_regex, text)
        doc_matches = list(map(lambda m: m[0], matches))

        matches = []
        for match in doc_matches:
            ref_matches = re.findall(ref_regex, match)
            matches += list(map(lambda m: m[1], ref_matches))

        return list(map(lambda m: m.strip(), matches))

    def __get_text(self, title):
        """Get the text for the document with the title 'title'"""
        doc_title = title.replace(" ", "_")
        request_url = 'https://en.wikipedia.org/w/api.php?action=query&titles=' + doc_title + \
                      '&prop=revisions&rvprop=content&format=json'
        response = requests.get(request_url)
        return json.dumps(response.json())
