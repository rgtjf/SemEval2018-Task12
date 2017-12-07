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
        _warrant0 = train_instance.get_warrant0(type=word_type, stopwords=stopwords, lower=lower)
        _warrant1 = train_instance.get_warrant1(type=word_type, stopwords=stopwords, lower=lower)
        _reason = train_instance.get_reason(type=word_type, stopwords=stopwords, lower=lower)
        _claim = train_instance.get_claim(type=word_type, stopwords=stopwords, lower=lower)
        title = train_instance.get_title(type=word_type, stopwords=stopwords, lower=lower)
        info = train_instance.get_info(type=word_type, stopwords=stopwords, lower=lower)

        _warrant0 = train_instance._warrant0.split()
        _warrant1 = train_instance._warrant1.split()
        _reason = train_instance._reason.split()
        _claim = train_instance._claim.split()
        _title = train_instance._title.split()
        _info = train_instance._info.split()

        _warrant0 = check_negative_sentence(_warrant0)
        _warrant1 = check_negative_sentence(_warrant1)
        _reason = check_negative_sentence(_reason)
        _claim = check_negative_sentence(_claim)

        # features = [_warrant0 != _warrant1, _reason != _claim,
        #             _warrant0 != _reason, _warrant0 != _claim,
        #             _warrant1 != _reason, _warrant1 != _claim]

        features = [
                    _warrant0, _warrant1,
                    _warrant0 == _reason!=_claim,
                    _warrant1 == _reason!=_claim,
                    # _warrant0 == _claim, _warrant1 == _claim,
                    ]
        features = [float(x) for x in features]

        infos = ['TODO']

        return features, infos

if __name__ == '__main__':
    pass
