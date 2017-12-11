# coding: utf8

"""
@author: rgtjf
@file: task.py
@time: 2017/10/23 13:38
"""

from __future__ import print_function

import codecs
import json
import re
import traceback

import pyprind

import stst
from nlplibs import corenlp_utils
from stst import utils
from stst.dict_utils import DictLoader
from data.example import Example
from data.example import ParseExample

def preprocess(sent):
    """
    preprocess for one sent
    Args:
        sent: str
    Returns:
        sent: clean_sent, str
    """
    r1 = re.compile(r'\<([^ ]+)\>')
    r2 = re.compile(r'\$US(\d)')
    sent = sent.replace(u'’', "'")
    sent = sent.replace(u'``', '"')
    sent = sent.replace(u"''", '"')
    sent = sent.replace(u"´", "'")
    sent = sent.replace(u"—", ' ')
    sent = sent.replace(u"–", ' ')
    sent = sent.replace(u"-", " ")
    sent = sent.replace(u"/", " ")
    sent = r1.sub(r'\1', sent)
    sent = r2.sub(r'$\1', sent)
    return sent


def calc_avg_tokens(train_instances):
    """
    warrant0 / warrant1 / reason / claim / title / info
    """
    warrant0 = []
    warrant1 = []
    reason = []
    claim = []
    title = []
    info = []
    for train_instance in train_instances:
        warrant0.append(len(train_instance.get_warrant0()))
        warrant1.append(len(train_instance.get_warrant1()))
        reason.append(len(train_instance.get_reason()))
        claim.append(len(train_instance.get_claim()))
        title.append(len(train_instance.get_title()))
        info.append(len(train_instance.get_info()))

    print(sum(warrant0) / len(warrant0))
    print(sum(warrant1) / len(warrant1))
    print(sum(reason) / len(reason))
    print(sum(claim) / len(claim))
    print(sum(title) / len(title))
    print(sum(info) / len(info))


def load_parse_data(file_path, init=False):
    """
    Load data after Parse, like POS, NER, etc.
    Args:
        file_path:
        init: false, load from file;
            else init from corenlp

    Returns:
        parse_data: List of Example:class
    """

    ''' Pre-Define Write File '''
    parse_train_file = file_path.replace('data', 'generate/parse')
    parse_word_file = file_path.replace('data', 'generate/word')
    parse_lemma_file = file_path.replace('data', 'generate/lemma')
    parse_stopwords_lemma_file = file_path.replace('data', 'generate/stopwords/lemma')

    if init:

        nlp = corenlp_utils.StanfordNLP(server_url='http://localhost:9000')

        print(file_path)
        ''' Parse Data '''
        examples = Example.load_data(file_path)

        print('*' * 50)
        print("Parse Data, file_path=%s, n_line=%d\n" % (file_path, len(examples)))

        # idx = 0
        parse_data = []
        process_bar = pyprind.ProgPercent(len(examples))
        for example in examples:
            process_bar.update()

            id = example.get_id()
            label = example.get_label()
            parse_lst = [id, label]
            try:
                # warrant0 / warrant1 / reason / claim / title / info
                example_lst = example.get_six()

                for sent in example_lst:
                    parse_sent = nlp.parse(sent)
                    parse_lst.append(sent)
                    parse_lst.append(parse_sent)

            except Exception:
                print(example.get_id())
                traceback.print_exc()
                parse_lst = ("id label warrant0 warrant1 reason claim title info".split())

            parse_data.append(parse_lst)

        ''' Write Data to File '''
        f_parse = utils.create_write_file(parse_train_file)
        f_word = utils.create_write_file(parse_word_file)
        # id warrant0 warrant1 label reason claim title info
        f_lemma = utils.create_write_file(parse_lemma_file)
        f_stopwords_lemma = utils.create_write_file(parse_stopwords_lemma_file)

        for parse_example in parse_data:
            parse_sent = json.dumps(parse_example)  # list -> str
            parse_example = ParseExample(parse_example)  # list -> class

            id = parse_example.get_id()
            label = parse_example.get_label()

            for word_type, fw in zip(['word', 'lemma'], [f_word, f_lemma]):

                warrant0 = parse_example.get_warrant0(type=word_type, return_str=True)
                warrant1 = parse_example.get_warrant1(type=word_type, return_str=True)
                reason = parse_example.get_reason(type=word_type, return_str=True)
                claim = parse_example.get_claim(type=word_type, return_str=True)
                title = parse_example.get_title(type=word_type, return_str=True)
                info = parse_example.get_info(type=word_type, return_str=True)
                sent = '%s\t%s\t%s\t%d\t%s\t%s\t%s\t%s' % (id, warrant0, warrant1,
                                                           label, reason, claim, title, info)
                print(sent, file=fw)

            warrant0 = parse_example.get_warrant0(type='lemma', stopwords=True, return_str=True)
            warrant1 = parse_example.get_warrant1(type='lemma', stopwords=True, return_str=True)
            reason = parse_example.get_reason(type='lemma', stopwords=True, return_str=True)
            claim = parse_example.get_claim(type='lemma', stopwords=True, return_str=True)
            title = parse_example.get_title(type='lemma', stopwords=True, return_str=True)
            info = parse_example.get_info(type='lemma', stopwords=True, return_str=True)
            sent = '%s\t%s\t%s\t%d\t%s\t%s\t%s\t%s' % (id, warrant0, warrant1, label, reason, claim, title, info)
            print(sent, file=f_stopwords_lemma)

            print(parse_sent, file=f_parse)

        f_parse.close()
        f_word.close()
        f_lemma.close()
        f_stopwords_lemma.close()

    ''' Load Data from File '''
    print('*' * 50)
    parse_data = []
    with codecs.open(parse_train_file, 'r', encoding='utf8') as f:
        for line in f:
            # obtain the json object
            parse_sent = json.loads(line)
            # obtain the class
            parse_example = ParseExample(parse_sent)
            parse_data.append(parse_example)

    print("Load Data, file_path=%s  n_line=%d\n" % (file_path, len(parse_data)))
    return parse_data


if __name__ == '__main__':
    print('Train')
    train_file = '../data/train-full.txt'
    train_instances = Example.load_data(train_file)
    calc_avg_tokens(train_instances)

    print('Dev')
    dev_file = '../data/dev-full.txt'
    dev_instances = Example.load_data(dev_file)
    calc_avg_tokens(dev_instances)

    train_swap_file = '../data/train-w-swap-full.txt'
    train_instances = Example.load_data(train_swap_file)

    print(len(train_instances))
    print(train_instances[0].get_instance_string())

    for train_instance in train_instances[:10]:
        print(train_instance.get_instance_string())

    train_instances = load_parse_data(train_swap_file, init=True)
    train_instance = load_parse_data(train_file, init=True)
    dev_instances = load_parse_data(dev_file, init=True)

    # for idx in range(len(train_instances)):
    #     print(idx, end=',')
    #     if idx % 2 == 1:
    #         a = train_instances[idx-1]
    #         b = train_instances[idx]
    #
    #         if a.get_warrant0() == b.get_warrant1() and a.get_warrant1() == b.get_warrant0():
    #             if a.get_label() + b.get_label() == 1:
    #                 print('sc', end=',')
    #             else:
    #                 raise ValueError
    #         else:
    #             print(b.get_instance_string())
