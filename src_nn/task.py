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

from data.example import Example, ParseExample


WORD_TYPE = 'lemma'

class Dataset(object):
    def __init__(self, file_name,
                 word_vocab,
                 max_sent_len,
                 num_class):
        """
        return the formatted matrix, which is used as the input to deep learning models
        Args: file_list:
              word_vocab:
        """
        self.examples = examples = ParseExample.load_data(file_name)

        instance_id_list = []
        warrant0_list = []
        warrant1_list = []
        correct_label_w0_or_w1_list = []
        reason_list = []
        claim_list = []
        debate_meta_data_list = []

        warrant0_len = []
        warrant1_len = []
        reason_len = []
        claim_len = []
        debate_meta_data_len = []

        for example in examples:
            warrant0, warrant1, reason, claim, title, info = example.get_six(type=WORD_TYPE)
            instance_id = example.get_id()
            debate_meta_data = title + info
            correct_label_w0_or_w1 = example.get_label()

            # convert to the ids
            warrant0 = data_utils.sent_to_index(warrant0, word_vocab)
            warrant1 = data_utils.sent_to_index(warrant1, word_vocab)
            reason = data_utils.sent_to_index(reason, word_vocab)
            claim = data_utils.sent_to_index(claim, word_vocab)
            debate_meta_data = data_utils.sent_to_index(debate_meta_data, word_vocab)
            correct_label_w0_or_w1 = data_utils.onehot_vectorize(correct_label_w0_or_w1, num_class)

            warrant0_len.append(min(len(warrant0), max_sent_len))
            warrant1_len.append(min(len(warrant1), max_sent_len))
            reason_len.append(min(len(reason), max_sent_len))
            claim_len.append(min(len(claim), max_sent_len))
            debate_meta_data_len.append(min(len(debate_meta_data), max_sent_len))

            # add to the result
            instance_id_list.append(instance_id)
            warrant0_list.append(warrant0)
            warrant1_list.append(warrant1)
            correct_label_w0_or_w1_list.append(correct_label_w0_or_w1)
            reason_list.append(reason)
            claim_list.append(claim)
            debate_meta_data_list.append(debate_meta_data)

        warrant0_list = data_utils.pad_2d_matrix(warrant0_list, max_sent_len)
        warrant1_list = data_utils.pad_2d_matrix(warrant1_list, max_sent_len)
        reason_list = data_utils.pad_2d_matrix(reason_list, max_sent_len)
        claim_list = data_utils.pad_2d_matrix(claim_list, max_sent_len)
        debate_meta_data_list = data_utils.pad_2d_matrix(debate_meta_data_list, max_sent_len)

        # text
        self.instance_id_list = np.array(instance_id_list)
        # flaot
        self.correct_label_w0_or_w1_list = np.array(correct_label_w0_or_w1_list, dtype=np.float32)
        # int
        self.warrant0_list = np.array(warrant0_list, dtype=np.int32)
        self.warrant1_list = np.array(warrant1_list, dtype=np.int32)
        self.reason_list = np.array(reason_list, dtype=np.int32)
        self.claim_list = np.array(claim_list, dtype=np.int32)
        self.debate_meta_data_list = np.array(debate_meta_data_list, dtype=np.int32)

        self.warrant0_len = np.array(warrant0_len, dtype=np.int32)
        self.warrant1_len = np.array(warrant1_len, dtype=np.int32)
        self.reason_len = np.array(reason_len, dtype=np.int32)
        self.claim_len = np.array(claim_len, dtype=np.int32)
        self.debate_meta_data_len = np.array(debate_meta_data_len, dtype=np.int32)

        self.do_id = word_vocab['do']

    def batch_iter(self, batch_size, shuffle=False):
        """
        UPDATE_0: add Batch for yield
        To support different model with different data:
        - model_1 want data 1, 2, 3, 4;
        - model_2 want data 1, 2, 3, 4, 5;
        ===
        during training: add some data to be enough batch_size
        during test: add some data to be enough batch_size
        :param batch_size:
        :param shuffle:
        :return:
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
            batch.add('warrant0', self.warrant0_list[excerpt])
            batch.add('warrant1', self.warrant1_list[excerpt])
            batch.add('label', self.correct_label_w0_or_w1_list[excerpt])
            batch.add('reason', self.reason_list[excerpt])
            batch.add('claim', self.reason_list[excerpt])
            batch.add('debate', self.debate_meta_data_list[excerpt])

            batch.add('warrant0_len', self.warrant0_len[excerpt])
            batch.add('warrant1_len', self.warrant1_len[excerpt])
            batch.add('reason_len', self.reason_len[excerpt])
            batch.add('claim_len', self.reason_len[excerpt])
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

            diff_warrant0, diff_warrant1, diff_warrant0_len, diff_warrant1_len = batch_diff(self.warrant0_list[excerpt], self.warrant1_list[excerpt], self.do_id, config.max_diff_len)
            diff_warrant0 = np.array(diff_warrant0, dtype=np.int32)
            diff_warrant0 = np.array(diff_warrant0, dtype=np.int32)
            diff_warrant0_len = np.array(diff_warrant0_len, dtype=np.int32)
            diff_warrant1_len = np.array(diff_warrant1_len, dtype=np.int32)
            batch.add('diff_warrant0', diff_warrant0)
            batch.add('diff_warrant1', diff_warrant1)
            batch.add('diff_warrant0_len', diff_warrant0_len)
            batch.add('diff_warrant1_len', diff_warrant1_len)

            yield batch


class Task(object):

    def __init__(self, task_name='word2vec-word', init=False):
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

        self.train_file = config.train_file
        self.dev_file = config.dev_file
        self.test_file = None
        self.word_embed_file = task['emb_file']

        self.word_dim = config.word_dim
        self.max_len = config.max_sent_len
        self.num_class = config.num_class

        self.w2i_file, self.we_file = config.get_w2i_we_file(task_name)
        utils.check_file_exist(self.w2i_file)
        utils.check_file_exist(self.we_file)

        self.train_predict_file = None
        self.dev_predict_file = None
        self.test_predict_file = None

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
