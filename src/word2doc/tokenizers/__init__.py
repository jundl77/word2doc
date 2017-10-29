#!/usr/bin/env python3
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

DEFAULTS = {
    'corenlp_classpath': os.getenv('CLASSPATH')
}


def set_default(key, value):
    global DEFAULTS
    DEFAULTS[key] = value


from word2doc.tokenizers.corenlp_tokenizer import CoreNLPTokenizer
from word2doc.tokenizers.regexp_tokenizer import RegexpTokenizer
from word2doc.tokenizers.simple_tokenizer import SimpleTokenizer

# Spacy is optional
try:
    from src.word2doc.tokenizers import SpacyTokenizer
except ImportError:
    pass


def get_class(name):
    if name == 'spacy':
        return SpacyTokenizer
    if name == 'corenlp':
        return CoreNLPTokenizer
    if name == 'regexp':
        return RegexpTokenizer
    if name == 'simple':
        return SimpleTokenizer

    raise RuntimeError('Invalid tokenizer: %s' % name)


def get_annotators_for_args(args):
    annotators = set()
    if args.use_pos:
        annotators.add('pos')
    if args.use_lemma:
        annotators.add('lemma')
    if args.use_ner:
        annotators.add('ner')
    return annotators


def get_annotators_for_model(model):
    return get_annotators_for_args(model.args)
