# coding: utf8
from __future__ import print_function

import codecs
import unicodedata
from nltk.tokenize.casual import TweetTokenizer

def tokenize(s):
    """
    Tokenization of the given text using TweetTokenizer delivered along with NLTK
    :param s: text
    :return: list of tokens
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


class Example():
    """
    Example:
        an argument: (claim [str/List/json], reason)
        select: warrant (explain the reason of the argument)
        debate_title, debate_info
    """
    def __init__(self, example_dict):
        self.id = example_dict['id']
        self.claim = tokenize(example_dict['claim'])
        self.reason = tokenize(example_dict['reason'])
        self.warrant0 = tokenize(example_dict['warrant0'])
        self.warrant1 = tokenize(example_dict['warrant1'])
        self.title = tokenize(example_dict['title'])
        self.info  = tokenize(example_dict['info'])
        self.debate = tokenize(example_dict['debate'])
        self.label = int(example_dict['label'])

    def get_label(self):
        return self.label

    def get_claim(self):
        return self.claim

    def get_reason(self):
        return self.reason

    def get_warrant0(self):
        return self.warrant0

    def get_warrant1(self):
        return self.warrant1

    def get_title(self):
        return self.title

    def get_info(self):
        return self.info

    def get_id(self):
        return self.id

    def get_instance_string(self):
        instance_string = "{}\t{}\t{}\t{}\t{}".format(self.label,
                                                      ' '.join(self.warrant0), ' '.join(self.warrant1),
                                                      ' '.join(self.claim), ' '.join(self.reason))
        return instance_string

    def get_six(self, return_str=True):
        if return_str:
            return ' '.join(self.warrant0), ' '.join(self.warrant1), ' '.join(self.reason), \
                    ' '.join(self.claim), ' '.join(self.title), ' '.join(self.info)
        else:
            return self.warrant0, self.warrant1, self.reason, self.claim, self.title, self.info

    def get_all(self):
        return self.id, self.warrant0, self.warrant1, self.label, self.reason, self.claim, self.debate

    @staticmethod
    def load_data(file_path):
        """ return list of examples """
        examples = []
        with codecs.open(file_path, encoding='utf8') as f:
            #id	warrant0	warrant1	correctLabelW0orW1	reason	claim	debateTitle	debateInfo
            headline = f.readline()
            for line in f:
                example_dict = {}
                items = line.strip().split('\t')
                example_dict['id'] = items[0]
                example_dict['warrant0'] = items[1]
                example_dict['warrant1'] = items[2]
                example_dict['label'] = items[3]
                example_dict['reason'] = items[4]
                example_dict['claim'] = items[5]
                example_dict['title'] = items[6]
                example_dict['info'] = items[7]
                example_dict['debate'] = items[6] + ' ' + items[7]
                example = Example(example_dict)
                examples.append(example)
        return examples