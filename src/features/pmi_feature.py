# coding: utf8
from __future__ import print_function

from stst import Feature
from stst import dict_utils
import config
from stst import utils


def co_gram(sa, sb):
    """
    >>> co_gram(['i', 't', 'u'], ['a', 'f']
    [('i', 'a'), ('i', 'f'), ('t', 'a'), ('t', 'f'), ('u', 'a'), ('u', 'f')]
    """
    gram = [(wa, wb) for wa in sa for wb in sb]
    return gram


def diffsents(sa, sb):
    """ tell the different part of a sentence pair
    Usage:
        la, ra, lb, rb = diffsents(warrant0, warrant1)
        diff_warrant0 = warrant0[la: ra+1]
        diff_warrant1 = warrant1[lb: rb+1]
    """
    m = len(sa)
    n = len(sb)
    la = lb = 0
    ra = m - 1
    rb = n - 1
    while la < m and lb < n:
        if sa[la] == sb[lb]:
            la += 1
            lb += 1
        else:
            break
    while ra >= 0 and rb >= 0:
        if sa[ra] == sb[rb]:
            ra -= 1
            rb -= 1
        else:
            break
    while la > ra or lb > rb:
        # la -= 1
        ra += 1
        # lb -= 1
        rb += 1
    if la == ra == m or lb == rb == n:
        la -= 1
        ra -= 1
        lb -= 1
        rb -= 1
    assert 0 <= la <= ra < m, "{}\t{}\t{}\t{}\t{}".format(m, la, ra, sa, sb)
    assert 0 <= lb <= rb < n, "{}\t{}\t{}\t{}\t{}".format(n, lb, rb, sb, sa)
    # sa[la, ra+1], sb[lb, rb+1]
    return la, ra, lb, rb


class BI_feature(Feature):
    """ warrant0 - others [(w0, w1), (w0, w1), etc]"""
    def extract_information(self, train_instances):
        if self.is_training:
            sents = []
            for train_instance in train_instances:
                warrant0, warrant1, reason, claim, title, info = train_instance.get_six(type='word')
                # obtain the diff part
                la, ra, lb, rb = diffsents(warrant0, warrant1)
                diff_warrant0 = warrant0[la: ra+1]
                diff_warrant1 = warrant1[lb: rb+1]
                # print('='*20)
                # print(warrant0)
                # print(warrant1)
                # print(diff_warrant0)
                # print(diff_warrant1)
                # append the co_gram: warrant0
                # sents.append(co_gram(diff_warrant0, warrant0))
                sents.append(co_gram(diff_warrant0, reason))
                sents.append(co_gram(diff_warrant0, claim))
                # sents.append(co_gram(diff_warrant0, title))
                # sents.append(co_gram(diff_warrant0, info))
                # append the co_gram: warrant1
                # sents.append(co_gram(diff_warrant1, warrant1))
                sents.append(co_gram(diff_warrant1, reason))
                sents.append(co_gram(diff_warrant1, claim))
                # sents.append(co_gram(diff_warrant1, title))
                # sents.append(co_gram(diff_warrant1, info))

            idf_dict = utils.idf_calculator(sents)
            # idf_dict = sorted(idf_dict.items(), key=lambda x: x[1], reverse=True)
            with utils.create_write_file(config.RESOURCE_DIR + '/bi_dict.txt') as fw:
                for key in idf_dict:
                    print('{}\t{}'.format(key, idf_dict[key]), file=fw)
            print('idf_dict length: ', len(idf_dict))
        else:
            with utils.create_read_file(config.RESOURCE_DIR + '/bi_dict.txt') as fr:
                idf_dict = {}
                for line in fr:
                    line = line.strip().split('\t')
                    idf_dict[line[0]] = float(line[1])
        self.bigram_dict = idf_dict

    def extract(self, train_instance):
        warrant0, warrant1, reason, claim, title, info = train_instance.get_six(type='word')
        # obtain the diff part
        la, ra, lb, rb = diffsents(warrant0, warrant1)
        diff_warrant0 = warrant0[la: ra + 1]
        diff_warrant1 = warrant1[lb: rb + 1]
        # append the co_gram: warrant0
        bi_warrant0 = []
        # bi_warrant0 += co_gram(diff_warrant0, warrant0)
        bi_warrant0 += co_gram(diff_warrant0, reason)
        bi_warrant0 += co_gram(diff_warrant0, claim)
        # bi_warrant0 += co_gram(diff_warrant0, title)
        # bi_warrant0 += co_gram(diff_warrant0, info)
        # append the co_gram: warrant1
        bi_warrant1 = []
        # bi_warrant1 += co_gram(diff_warrant1, warrant1)
        bi_warrant1 += co_gram(diff_warrant1, reason)
        bi_warrant1 += co_gram(diff_warrant1, claim)
        # bi_warrant1 += co_gram(diff_warrant1, title)
        # bi_warrant1 += co_gram(diff_warrant1, info)

        self.vocab = utils.word2index(self.bigram_dict)
        feat0 = utils.vectorize(bi_warrant0, self.bigram_dict, self.vocab)
        feat1 = utils.vectorize(bi_warrant1, self.bigram_dict, self.vocab)
        infos = [len(self.bigram_dict), 'bigram']
        return feat0 + feat1, infos


class PMI_Feature(Feature):

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
                    _warrant0 == _reason != _claim,
                    _warrant1 == _reason != _claim,
                    # _warrant0 == _claim, _warrant1 == _claim,
                    ]
        features = [float(x) for x in features]

        infos = ['TODO']

        return features, infos

if __name__ == '__main__':
    pass
