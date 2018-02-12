# coding: utf8

"""
@author: rgtjf
@file: warrant_feature.py
@time: 2017/11/6 14:24
"""

from __future__ import print_function
from stst import Feature
from stst import dict_utils
import config


def check_negative_sentence(sent):
    """
    Return bool to tell whether the sentence is negative or not!
    :param sent:
    :return: bool
    """
    negation_terms = dict_utils.DictLoader().load_dict('negation', config.negation_term_file)
    negative_words = dict_utils.DictLoader().load_dict('negative', config.negative_word_file)
    is_negation = False
    for word in sent:
        if word in negation_terms: # or word in negative_words:
            is_negation = True
            break
    return is_negation


class Warrant_Feature(Feature):

    def extract(self, train_instance):
        # type = word / lemma / pos / ner, stopwords = True / False, lower = True / False
        word_type, stopwords, lower = 'lemma', False, True
        warrant0, warrant1, reason, claim, debate, negclaim = train_instance.get_six(type=word_type,
                                                                                    stopwords=stopwords,
                                                                                    lower=lower)

        warrant0 = check_negative_sentence(warrant0)
        warrant1 = check_negative_sentence(warrant1)
        reason = check_negative_sentence(reason)
        claim = check_negative_sentence(claim)

        features = [
                    warrant0, warrant1,
                    warrant0 == reason != claim,
                    warrant1 == reason != claim,
                    # _warrant0 == _claim, _warrant1 == _claim,
                    ]
        features = [float(x) for x in features]

        infos = ['TODO']

        return features, infos

if __name__ == '__main__':
    pass
