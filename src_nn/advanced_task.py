# coding: utf-8
from __future__ import print_function

import os
import pickle
import codecs
import six
import numpy as np

import data_utils
import utils
import config

import sys
sys.path.append('../src')
from input.example import Example, ParseExample

WORD_TYPE = 'lemma'


def merge_parse_sent(parse_sent_list):
    """
    only extend tokens
    """
    merge_parse_sent = parse_sent_list[0]
    for parse_sent in parse_sent_list[1:]:
        merge_parse_sent["sentences"][0]["tokens"] += parse_sent["sentences"][0]["tokens"]
    return merge_parse_sent


class Dataset(object):
    def __init__(self, file_name,
                 word_vocab,
                 max_sent_len,
                 num_class,
                 char_vocab=None,
                 max_word_len=None,
                 ):
        """
        return the formatted matrix, which is used as the input to deep learning models
        FORMAT:
        1. First translate to individual data
        self.{
            'instance_id_list': [all]
            'correct_label_w0_or_w1_list': [all]
            'warrant0': {
                'origin': [all]
                'word': [all]
                'len': [all]
                'pos': [all]
                'char': [all]
                'word_len': [all]
            }
            'diff_warrant0': {
                'word': [all]
                'len': [all]
                'pos': [all]
                'char': [all]
                'word_len': [all]
            }
        }
        Args: file_list:
              word_vocab:
        """
        self.examples = examples = ParseExample.load_data(file_name)

        instance_id_list = []
        correct_label_w0_or_w1_list = []
        warrant0_list = []
        warrant1_list = []
        reason_list = []
        claim_list = []
        negclaim_list = []
        debate_meta_data_list = []

        for example in examples:
            # warrant0, warrant1, reason, claim, title, info, negclaim
            warrant0, warrant1, reason, claim, title, info, negclaim = example.get_all()
            instance_id = example.get_id()
            correct_label_w0_or_w1 = example.get_label()
            correct_label_w0_or_w1 = data_utils.onehot_vectorize(correct_label_w0_or_w1, num_class)

            instance_id_list.append(instance_id)
            correct_label_w0_or_w1_list.append(correct_label_w0_or_w1)
            warrant0_list.append(warrant0)
            warrant1_list.append(warrant1)
            reason_list.append(reason)
            claim_list.append(claim)
            negclaim_list.append(negclaim)
            debate_meta_data_list.append(merge_parse_sent([title, info]))

        # int
        self.warrant0 = self.obtain_all(warrant0_list, max_sent_len)
        self.warrant1 = self.obtain_all(warrant1_list, max_sent_len)
        self.reason = self.obtain_all(reason_list, max_sent_len)
        self.claim = self.obtain_all(claim_list, max_sent_len)
        self.negclaim = self.obtain_all(negclaim_list, max_sent_len)
        self.debate_meta_data = self.obtain_all(debate_meta_data_list, max_sent_len)
        # text
        self.instance_id_list = np.array(instance_id_list)
        # float
        self.correct_label_w0_or_w1_list = np.array(correct_label_w0_or_w1_list, dtype=np.float32)

        self.do_id = word_vocab['do']

    def obtain_all(self, parse_sents, max_sent_len):
        results = {}
        # obtain the words
        words_list = []
        words_len_list = []
        for example in parse_sents:
            words = example.get_words(example, type=WORD_TYPE)
            words = data_utils.sent_to_index(words, self.word_vocab)
            words = data_utils.pad_1d_vector(words, max_sent_len)
            words_len = min(len(words), max_sent_len)

            words_list.append(words)
            words_len_list.append(words_len)
        results['word'] = np.array(words_list, dtype=np.int32)
        results['word_len'] = np.array(words_len_list, dtype=np.int32)

        # obtain the pos
        pos_list = []
        for example in parse_sents:
            pos = example.get_words(example, type='pos')
            pos = data_utils.sent_to_index(pos, self.pos_vocab)
            pos = data_utils.pad_1d_vector(pos, max_sent_len)
            pos_list.append(pos)
        results['pos'] = np.array(pos_list, dtype=np.int32)

        # obtain the char
        chars_list = []
        chars_len_list = []
        for example in parse_sents:
            words = example.get_words(example, type=WORD_TYPE)
            chars, chars_len = data_utils.char_padding(words, self.char_vocab, max_sent_len, config.max_word_len)
            chars_list.append(chars)
            chars_len_list.append(chars_len)
        results['char'] = np.array(chars_list, dtype=np.int32)
        results['char_len'] = np.array(chars_len_list, dtype=np.int32)

        return results

    def batch_iter(self, batch_size, shuffle=False):
        """
        Batch for yield
        1. support different batch_size in train and test
        2. To support different model with different data:
           Ex:  model_1 want data 1, 2, 3, 4;
                model_2 want data 1, 2, 3, 4, 5;
        Args:
            batch_size:
            shuffle:
        Returns:
            batch
        """
        n_data = len(self.correct_label_w0_or_w1_list)

        idx = np.arange(n_data)
        if shuffle:
            idx = np.random.permutation(n_data)

        for start_idx in range(0, n_data, batch_size):
            # end_idx = min(start_idx + batch_size, n_data)
            end_idx = start_idx + batch_size
            excerpt = idx[start_idx:end_idx]
            batch = data_utils.Batch()
            batch.add('instance_id', self.instance_id_list[excerpt])
            batch.add('label', self.correct_label_w0_or_w1_list[excerpt])

            for sent_type in ['warrant0', 'warrant1', 'reason', 'claim', 'debate']:
                self.__dict__[sent_type]['word']
                batch.add(sent_type, self.word_list[sent_type][excerpt])
            batch.add('warrant0', self.warrant0_list[excerpt])
            batch.add('warrant1', self.warrant1_list[excerpt])
            batch.add('reason', self.reason_list[excerpt])
            batch.add('claim', self.claim_list[excerpt])
            batch.add('debate', self.debate_meta_data_list[excerpt])

            batch.add('warrant0_len', self.warrant0_len[excerpt])
            batch.add('warrant1_len', self.warrant1_len[excerpt])
            batch.add('reason_len', self.reason_len[excerpt])
            batch.add('claim_len', self.claim_len[excerpt])
            batch.add('debate_len', self.debate_meta_data_len[excerpt])

            def batch_diff(batch_sent0, batch_sent1, do_id, max_diff_len):
                """
                Args:
                    do_id: word_vocab['do']
                """
                batch_diff_sent0 , batch_diff_sent1 = [], []
                batch_diff_sent0_len , batch_diff_sent1_len = [], []
                for sent0, sent1 in zip(batch_sent0, batch_sent1):
                    diff_sent0 = [word for word in sent0 if word not in sent1]
                    diff_sent1 = [word for word in sent1 if word not in sent0]
                    if not diff_sent0:
                        diff_sent0 = [do_id]
                    if not diff_sent1:
                        diff_sent1 = [do_id]
                    batch_diff_sent0.append(diff_sent0)
                    batch_diff_sent1.append(diff_sent1)
                    batch_diff_sent0_len.append(min(len(diff_sent0), max_diff_len))
                    batch_diff_sent1_len.append(min(len(diff_sent1), max_diff_len))

                batch_diff_sent0 = data_utils.pad_2d_matrix(batch_diff_sent0, max_diff_len)
                batch_diff_sent1 = data_utils.pad_2d_matrix(batch_diff_sent1, max_diff_len)
                return batch_diff_sent0, batch_diff_sent1, batch_diff_sent0_len, batch_diff_sent1_len

            def batch_diff_cont(batch_sent0, batch_sent1, do_id, max_diff_len):
                """
                Args:
                    do_id: word_vocab['do']
                """
                batch_diff_sent0, batch_diff_sent1 = [], []
                batch_diff_sent0_len, batch_diff_sent1_len = [], []
                for sent0, sent1 in zip(batch_sent0, batch_sent1):

                    la, ra, lb, rb = diffsents(sent0, sent1)
                    diff_sent0 = sent0[la: ra + 1]
                    diff_sent1 = sent1[lb: rb + 1]

                    batch_diff_sent0.append(diff_sent0)
                    batch_diff_sent1.append(diff_sent1)
                    batch_diff_sent0_len.append(min(len(diff_sent0), max_diff_len))
                    batch_diff_sent1_len.append(min(len(diff_sent1), max_diff_len))

                batch_diff_sent0 = data_utils.pad_2d_matrix(batch_diff_sent0, max_diff_len)
                batch_diff_sent1 = data_utils.pad_2d_matrix(batch_diff_sent1, max_diff_len)
                return batch_diff_sent0, batch_diff_sent1, batch_diff_sent0_len, batch_diff_sent1_len

            diff_warrant0, diff_warrant1, diff_warrant0_len, diff_warrant1_len = batch_diff_cont(self.warrant0_list[excerpt], self.warrant1_list[excerpt], self.do_id, config.max_diff_len)
            diff_warrant0 = np.array(diff_warrant0, dtype=np.int32)
            diff_warrant0 = np.array(diff_warrant0, dtype=np.int32)
            diff_warrant0_len = np.array(diff_warrant0_len, dtype=np.int32)
            diff_warrant1_len = np.array(diff_warrant1_len, dtype=np.int32)
            batch.add('diff_warrant0', diff_warrant0)
            batch.add('diff_warrant1', diff_warrant1)
            batch.add('diff_warrant0_len', diff_warrant0_len)
            batch.add('diff_warrant1_len', diff_warrant1_len)

            yield batch

    def write_out(self, predicts, output_file, acc):
        print("Write to file: {}".format(output_file))
        print("Examples: {}, Predicts: {}".format(len(self.examples), len(predicts)))
        utils.check_file_exist(output_file)
        with utils.create_write_file(output_file) as fw:
            print('Dev: %.4f' % acc, file=fw)
            for idx, example in enumerate(self.examples):
                example_str = example.get_instance_string()
                print("{}\t#\t{}".format(predicts[idx], example_str), file=fw)


def diffsents(sa, sb):
    """ tell the different part of a sentence pair"""
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


class Task(object):

    def __init__(self, task_name='word2vec-word', init=False, expand=True):
        tasks = {
            'word2vec-lemma': {'emb_file': config.word_embed_file, 'word_type': 'lemma'},
            'word2vec-word': {'emb_file': config.word_embed_file, 'word_type': 'word'},
            'fasttext-lemma': {'emb_file': config.fasttext_file, 'word_type': 'lemma'},
            'fasttext-word': {'emb_file': config.fasttext_file, 'word_type': 'word'},
            'paragram-lemma': {'emb_file': config.paragram_file, 'word_type': 'lemma'},
            'paragram-word': {'emb_file': config.paragram_file, 'word_type': 'word'},
        }
        task = tasks[task_name]
        global WORD_TYPE
        WORD_TYPE = task['word_type']

        self.train_file = config.train_exp_file if expand else config.train_file

        self.dev_file = config.dev_file
        self.test_file = config.test_file
        self.word_embed_file = task['emb_file']

        self.word_dim = config.word_dim
        self.max_len = config.max_sent_len
        self.num_class = config.num_class

        self.w2i_file, self.we_file = config.get_w2i_we_file(task_name)
        utils.check_file_exist(self.w2i_file)
        utils.check_file_exist(self.we_file)

        if init:
            word_vocab = self.build_vocab()
            self.word_vocab, self.embed = data_utils.load_word_embedding(word_vocab, self.word_embed_file, self.word_dim)

            data_utils.save_params(self.word_vocab, self.w2i_file)
            data_utils.save_params(self.embed, self.we_file)
        else:
            self.word_vocab = data_utils.load_params(self.w2i_file)
            self.embed = data_utils.load_params(self.we_file)

        print("vocab size: %d" % len(self.word_vocab), "we shape: ", self.embed.shape)
        self.train_data = Dataset(self.train_file, self.word_vocab, self.max_len, self.num_class)
        self.dev_data = Dataset(self.dev_file, self.word_vocab, self.max_len, self.num_class)
        if self.test_file:
            self.test_data = Dataset(self.test_file, self.word_vocab, self.max_len, self.num_class)

    def build_vocab(self):
        """
        build sents to  build vocab
        Return: vocab
        """
        if self.test_file is None:
            print('test_file is None')
            file_list = [self.train_file, self.dev_file]
        else:
            file_list = [self.train_file, self.dev_file, self.test_file]

        examples = []
        for file_name in file_list:
            examples += ParseExample.load_data(file_name)

        sents = []
        for example in examples:
            warrant0, warrant1, reason, claim, title, info = example.get_six(type=WORD_TYPE)
            debate_meta_data = title + info
            sents.append(warrant0)
            sents.append(warrant1)
            sents.append(reason)
            sents.append(claim)
            sents.append(debate_meta_data)

        vocab = data_utils.build_word_vocab(sents)

        return vocab


if __name__ == '__main__':
    # task = Task(init=True)
    # task = Task(task_name='word2vec-lemma', init=True)
    task = Task(task_name='paragram-lemma', init=True)
    task = Task(task_name='paragram-word', init=True)
