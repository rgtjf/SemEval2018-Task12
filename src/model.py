# coding: utf8

"""
@author: rgtjf
@file: model.py
@time: 2017/10/2 21:07
"""

from __future__ import print_function
import tensorflow as tf


class Model(object):

    def __init__(self, vocab, params):

        self.debate = tf.placeholder(tf.int32, [None, None], name='debate')
        self.claim = tf.placeholder(tf.int32, [None, None], name='claim')
        self.reason = tf.placeholder(tf.int32, [None, None], name='reason')
        self.W0 = tf.placeholder(tf.int32, [None, None], name='W0')
        self.W1 = tf.placeholder(tf.int32, [None, None], name='W1')

        self.debate_len = tf.placeholder(tf.int32, [None], name='debate')
        self.claim_len = tf.placeholder(tf.int32, [None], name='claim')
        self.reason_len = tf.placeholder(tf.int32, [None], name='reason')
        self.W0_len = tf.placeholder(tf.int32, [None], name='W0')
        self.W1_len = tf.placeholder(tf.int32, [None], name='W1')


        def BiLSTM(input_x, input_x_len, hidden_size, dropout_keep_rate=None):
            """
            """

            cell_fw = tf.contrib.rnn.GRUCell(hidden_size)
            cell_bw = tf.contrib.rnn.GRUCell(hidden_size)

            if dropout_keep_rate:
                cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=dropout_keep_rate)
                cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=dropout_keep_rate)

            b_outputs, b_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input_x, sequence_length=input_x_len, dtype=tf.float32)
            outputs = tf.concat(b_outputs, axis=2)
            return outputs

        def MaxPooling(input_x):
            """ Max Pooling """
            pool = tf.reduce_max(input_x, axis=1)
            return pool

        with tf.variable_scope("bi_lstm") as s:
            hidden_size = 64
            lstm_debate = BiLSTM(self.debate, self.debate_len, hidden_size // 2)
            s.reuse_variables()
            lstm_claim = BiLSTM(self.claim, self.claim_len, hidden_size // 2)
            lstm_reason = BiLSTM(self.reason, self.reason_len, hidden_size // 2)
            lstm_W0 = BiLSTM(self.W0, self.W0_len, hidden_size // 2)
            lstm_W1 = BiLSTM(self.W1, self.W1_len, hidden_size // 2)

        with tf.variable_scope("max-pooling") as s:
            pooling_debate = MaxPooling(lstm_debate)
            pooling_claim = MaxPooling(lstm_claim)
            pooling_reason = MaxPooling(lstm_reason)
            pooling_W0 = MaxPooling(lstm_W0)
            pooling_W1 = MaxPooling(lstm_W1)



    def train_model(self):
        pass


    def test_model(self):
        pass


    def _make_feed_dict(self, batch):
        """

        :param batch:
        :return:
        """

if __name__ == '__main__':
    pass
