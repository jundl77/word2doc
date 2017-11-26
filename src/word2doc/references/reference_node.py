#!/usr/bin/env python3
# Copyright 2017-present
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


class ReferenceNode:

    def __init__(self, title):
        self.title = title
        self.children = {}

    def get_title(self):
        return self.title

    def get_children(self):
        return self.children

    def get_child(self, title):
        return self.children[title]

    def remove_child(self, title):
        del self.children[title]

    def get_distant_child(self, title):
        if title in self.children:
            return self.get_child(title)
        else:
            for t, c in self.children.items():
                res = c.get_distant_child(title)
                if res is not None:
                    return res

        return None

    def add_child(self, node):
        self.children[node.get_title()] = node

    def print(self):
        self.__print_helper(0)

    def __print_helper(self, gen):
        print(" " * gen * 4 + "Document Title: " + self.title)
        print(" " * gen * 4 + "Children: ")
        for t, c in self.children.items():
            c.__print_helper(gen + 1)
