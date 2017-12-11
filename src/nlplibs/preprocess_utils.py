# coding: utf8
from __future__ import print_function

import codecs
import unicodedata
from nltk.tokenize.casual import TweetTokenizer


def tokenize(s):
    """
    Tokenization of the given text using TweetTokenizer delivered along with NLTK
    Args:
        s: tex
    Return:
        list of tokens
    """
    sentence_splitter = TweetTokenizer()
    tokens = sentence_splitter.tokenize(s)
    result = []
    for word in tokens:
        # the last "decode" function is because of Python3
        # http://stackoverflow.com/questions/2592764/what-does-a-b-prefix-before-a-python-string-mean
        w = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8').strip()
        # and add only if not empty (it happened in some data that there were empty tokens...)
        if w:
            result.append(w)
    return result
