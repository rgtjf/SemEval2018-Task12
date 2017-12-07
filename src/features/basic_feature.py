# coding: utf8

"""
@author: rgtjf
@file: basic_feature.py
@time: 2017/11/6 16:51
"""

from __future__ import print_function

from stst import Feature
from stst import utils
import config

class BowFeature(Feature):

    def extract_information(self, train_instances):
        if self.is_training:
            sents = []
            for train_instance in train_instances:
                _warrant0 = train_instance._warrant0.split()
                _warrant1 = train_instance._warrant1.split()
                _reason = train_instance._reason.split()
                _claim = train_instance._claim.split()
                sents.append(_warrant0)
                sents.append(_warrant1)
                sents.append(_reason)
                sents.append(_claim)
            idf_dict = utils.idf_calculator(sents)
            # idf_dict = sorted(idf_dict.items(), key=lambda x: x[1], reverse=True)
            with utils.create_write_file(config.RESOURCE_DIR + '/idf_dict.txt') as fw:
                for key in idf_dict:
                    print('{}\t{}'.format(key, idf_dict[key]), file=fw)
            print(len(idf_dict))
        else:
            with utils.create_read_file(config.RESOURCE_DIR + '/idf_dict.txt') as fr:
                idf_dict = {}
                for line in fr:
                    line = line.strip().split('\t')
                    idf_dict[line[0]] = float(line[1])
        self.unigram_dict = idf_dict

    def extract(self, train_instance):
        _warrant0 = train_instance._warrant0.split()
        _warrant1 = train_instance._warrant1.split()
        _reason = train_instance._reason.split()
        _claim = train_instance._claim.split()

        _warrant0 = _warrant0 + _reason + _claim
        _warrant1 = _warrant1 + _reason + _claim

        self.vocab = utils.word2index(self.unigram_dict)
        feat0 = utils.vectorize(_warrant0, self.unigram_dict, self.vocab)
        feat1 = utils.vectorize(_warrant1, self.unigram_dict, self.vocab)
        infos = [len(self.unigram_dict), 'unigram']
        return feat0 + feat1, infos


if __name__ == '__main__':
    pass
