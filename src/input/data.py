# coding: utf8
from __future__ import print_function

import codecs
import json
import re
import traceback
import os
import pyprind
import sys
sys.path.append('..')

import stst
import config
from nlplibs import corenlp_utils
from stst import utils
from stst.dict_utils import DictLoader
from input.example import Example
from input.example import ParseExample

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

    pairs = [('alwayst', 'always'), ('paretnt', 'parent'),
             ('financiall', 'financial'), # ("'s", "is"),
             ('candidante', 'candidate'), ('vaildate', 'validate'),
             ('locaiton', 'location'), ('sufficience', 'sufficiency'),
             ('enrol', 'enroll')]
    for pair in pairs:
        sent = re.sub(r'\b%s\b' % pair[0], pair[1], sent)
    pass
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


def load_data(file_path):
    """ return list of examples """
    examples = []
    with codecs.open(file_path, encoding='utf8') as f:
        # id	warrant0	warrant1	correctLabelW0orW1	reason	claim	debateTitle	debateInfo
        headline = f.readline()
        for line in f:
            example_dict = {}
            items = line.strip().split('\t')
            example_dict['id'] = items[0]
            example_dict['warrant0'] = preprocess(items[1])
            example_dict['warrant1'] = preprocess(items[2])
            example_dict['label'] = int(items[3])
            example_dict['reason'] = preprocess(items[4])
            example_dict['claim'] = preprocess(items[5])
            example_dict['title'] = preprocess(items[6])
            example_dict['info'] = preprocess(items[7])
            example_dict['debate'] = preprocess(items[6] + ' ' + items[7])
            example_dict['negclaim'] = preprocess(config.claim_dict[items[5]])
            examples.append(example_dict)
    return examples


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
    parse_train_file = file_path.replace('data/', 'generate/parse/')
    parse_word_file = file_path.replace('data/', 'generate/word/')
    parse_lemma_file = file_path.replace('data/', 'generate/lemma/')
    parse_stopwords_lemma_file = file_path.replace('data/', 'generate/stopwords/lemma/')

    if init or not os.path.exists(parse_train_file):

        ''' Define CoreNLP'''
        nlp = corenlp_utils.StanfordNLP(server_url='http://localhost:9000')

        ''' Read data '''
        print("Read Data from file: %s" % file_path)
        examples = load_data(file_path)

        ''' Parse data '''
        print('*' * 50)
        print("Parse Data to file: %s, n_line: %d\n" % (parse_train_file, len(examples)))
        parse_data = []
        process_bar = pyprind.ProgPercent(len(examples))
        for example in examples:
            process_bar.update()
            id = example['id']
            label = example['label']
            parse_lst = [id, label]
            try:
                # warrant0 / warrant1 / reason / claim / debate / negclaim
                example_lst = [example['warrant0'], example['warrant1'], example['reason'],
                               example['claim'], example['debate'], example['negclaim'],
                               example['title'], example['info']]
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
                warrant0, warrant1, reason, claim, debate, negclaim = parse_example.get_six(
                        return_str=True, type=word_type)
                sent = '%s\t%s\t%s\t%d\t%s\t%s\t%s\t%s' % (id, warrant0, warrant1,
                                                           label, reason, claim, debate, negclaim)
                print(sent, file=fw)

            warrant0, warrant1, reason, claim, debate, negclaim = parse_example.get_six(
                    return_str=True, type='lemma', stopwords=True)
            sent = '%s\t%s\t%s\t%d\t%s\t%s\t%s\t%s' % (id, warrant0, warrant1, label, reason,
                                                       claim, debate, negclaim)
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
    ROOT_DIR = '../../data'
    print('Train')
    train_file = ROOT_DIR + '/train-full.txt'
    train_instances = Example.load_data(train_file)
    calc_avg_tokens(train_instances)

    print('Dev')
    dev_file = ROOT_DIR + '/dev-full.txt'
    dev_instances = Example.load_data(dev_file)
    calc_avg_tokens(dev_instances)

    print('Test')
    test_file = ROOT_DIR + '/test-full-tmp.txt'
    test_instances = Example.load_data(test_file)
    calc_avg_tokens(test_instances)

    train_swap_file = ROOT_DIR + '/train-w-swap-full.txt'
    train_instances = Example.load_data(train_swap_file)

    print(len(train_instances))
    print(train_instances[0].get_instance_string())

    for train_instance in train_instances[:10]:
        print(train_instance.get_instance_string())

    train_instances = load_parse_data(train_swap_file, init=True)
    test_instances = load_parse_data(test_file, init=True)
    dev_instances = load_parse_data(dev_file, init=True)

    # train_instance = load_parse_data(train_file, init=True)
    # train_instance = load_parse_data(train_file, init=True)

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
