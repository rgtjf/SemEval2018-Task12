# coding: utf8
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import config
import tf_utils
from attention_lstm import BasicAttLSTMCell


class IntraAttentionIIModel(object):

    def __init__(self, FLAGS=None):
        self.FLAGS = FLAGS
        self.config = config

        self.diff_len = config.max_diff_len
        self.seq_len = config.max_sent_len
        self.embed_size = config.word_dim
        self.num_class = config.num_class
        self.lstm_size = config.lstm_size

        # Add Word Embedding
        self.we = tf.Variable(FLAGS.we, name='emb')

        # Add PlaceHolder

        # define basic four input layers - for warrant0, warrant1, reason, claim
        self.input_warrant0 = tf.placeholder(tf.int32, (None, self.seq_len), name='warrant0')  # [batch_size, sent_len]
        self.input_warrant1 = tf.placeholder(tf.int32, (None, self.seq_len), name='warrant1')  # [batch_size, sent_len]
        self.input_reason = tf.placeholder(tf.int32, (None, self.seq_len), name='reason')  # [batch_size, sent_len]
        self.input_claim = tf.placeholder(tf.int32, (None, self.seq_len), name='claim')  # [batch_size, sent_len]
        self.input_debate = tf.placeholder(tf.int32, (None, self.seq_len), name='debate')  # [batch_size, sent_len]

        self.warrant0_len = tf.placeholder(tf.int32, (None, ), name='warrant0_len')  # [batch_size,]
        self.warrant1_len = tf.placeholder(tf.int32, (None, ), name='warrant1_len')  # [batch_size,]
        self.reason_len = tf.placeholder(tf.int32, (None, ), name='reason_len')  # [batch_size,]
        self.claim_len = tf.placeholder(tf.int32, (None, ), name='claim_len')  # [batch_size,]
        self.debate_len = tf.placeholder(tf.int32, (None, ), name='debate_len')  # [batch_size,]

        self.target_label = tf.placeholder(tf.int32, (None, self.num_class), name='label')  # [batch_size, num_class]

        self.drop_keep_rate = tf.placeholder(tf.float32)
        self.learning_rate = tf.placeholder(tf.float32)

        self.input_diff_warrant0 = tf.placeholder(tf.int32, (None, self.diff_len), name='diff_warrant0')  # [batch_size, sent_len]
        self.input_diff_warrant1 = tf.placeholder(tf.int32, (None, self.diff_len), name='diff_warrant1')  # [batch_size, sent_len]
        self.diff_warrant0_len = tf.placeholder(tf.int32, (None,), name='warrant0_len')  # [batch_size,]
        self.diff_warrant1_len = tf.placeholder(tf.int32, (None,), name='warrant1_len')  # [batch_size,]

        # now define embedded layers of the input
        embedded_warrant0 =  tf.nn.embedding_lookup(self.we, self.input_warrant0)
        embedded_warrant1 =  tf.nn.embedding_lookup(self.we, self.input_warrant1)
        embedded_reason = tf.nn.embedding_lookup(self.we, self.input_reason)
        embedded_claim = tf.nn.embedding_lookup(self.we, self.input_claim)
        embedded_debate = tf.nn.embedding_lookup(self.we, self.input_debate)

        embedded_diff_warrant0 = tf.nn.embedding_lookup(self.we, self.input_diff_warrant0)
        embedded_diff_warrant1 = tf.nn.embedding_lookup(self.we, self.input_diff_warrant1)

        ''' BiLSTM layer '''
        def BiLSTM(input_x, input_x_len, hidden_size, num_layers=1, dropout_rate=None, return_sequence=True):
            """
            TODO: return_sequence Bug
            """
            # cell = tf.contrib.rnn.GRUCell(hidden_size)
            cell_fw = tf.contrib.rnn.BasicLSTMCell(hidden_size)
            cell_bw = tf.contrib.rnn.BasicLSTMCell(hidden_size)

            if num_layers > 1:
                # Warning! Please consider that whether the cell to stack are the same
                cell_fw = tf.contrib.rnn.MultiRNNCell([cell_fw for _ in range(num_layers)])
                cell_bw = tf.contrib.rnn.MultiRNNCell([cell_bw for _ in range(num_layers)])

            if dropout_rate:
                cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=(1 - dropout_rate))
                cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=(1 - dropout_rate))

            b_outputs, b_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input_x,
                                                                  sequence_length=input_x_len,
                                                                  dtype=tf.float32)
            if return_sequence:
                outputs = tf.concat(b_outputs, axis=2)
            else:
                outputs = tf.concat([b_states[0][1], b_states[1][1]], axis=-1)
            return outputs

        def AttBiLSTM(attention_vector, input_x, input_x_len, hidden_size, return_sequence=True):
            cell_fw = BasicAttLSTMCell(attention_vector, num_units=hidden_size)
            cell_bw = BasicAttLSTMCell(attention_vector, num_units=hidden_size)

            b_outputs, b_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input_x,
                                                                  sequence_length=input_x_len, dtype=tf.float32)
            if return_sequence:
                outputs = tf.concat(b_outputs, axis=2)
            else:
                # states: [c, h]
                outputs = tf.concat([b_states[0][1], b_states[1][1]], axis=-1)
            return outputs

        avg_diff_warrant0 = tf_utils.AvgPooling(embedded_diff_warrant0, self.diff_warrant0_len, self.diff_len)
        avg_diff_warrant1 = tf_utils.AvgPooling(embedded_diff_warrant1, self.diff_warrant1_len, self.diff_len)

        with tf.variable_scope("att_warrant_lstm") as s:
            bilstm_warrant0 = AttBiLSTM(avg_diff_warrant0, embedded_warrant0, self.warrant0_len, self.lstm_size)
            s.reuse_variables()
            bilstm_warrant1 = AttBiLSTM(avg_diff_warrant1, embedded_warrant1, self.warrant1_len, self.lstm_size)

        with tf.variable_scope("bi_lstm") as s:
            bilstm_reason = BiLSTM(embedded_reason, self.reason_len, self.lstm_size)
            s.reuse_variables()
            bilstm_claim = BiLSTM(embedded_claim, self.claim_len, self.lstm_size)
            bilstm_debate = BiLSTM(embedded_debate, self.debate_len, self.lstm_size)

        ''' MaxPooling Layer '''
        pooling_warrant0 = tf_utils.MaxPooling(bilstm_warrant0, self.warrant0_len)
        pooling_warrant1 = tf_utils.MaxPooling(bilstm_warrant1, self.warrant1_len)
        pooling_reason = tf_utils.MaxPooling(bilstm_reason, self.reason_len)
        pooling_claim = tf_utils.MaxPooling(bilstm_claim, self.claim_len)
        pooling_debate = tf_utils.MaxPooling(bilstm_debate, self.debate_len)

        attention_vector_for_W0 = tf.concat([pooling_debate, pooling_reason, pooling_warrant0, pooling_claim, avg_diff_warrant0], axis=-1)
        attention_vector_for_W1 = tf.concat([pooling_debate, pooling_reason, pooling_warrant1, pooling_claim, avg_diff_warrant1], axis=-1)

        with tf.variable_scope("att_lstm") as s:
            attention_warrant0 = AttBiLSTM(attention_vector_for_W0, bilstm_warrant0, self.warrant0_len, self.lstm_size, return_sequence=False)
            s.reuse_variables()
            attention_warrant1 = AttBiLSTM(attention_vector_for_W1, bilstm_warrant1, self.warrant1_len, self.lstm_size, return_sequence=False)

        self.attention_warrant0 = attention_warrant0
        self.attention_warrant1 = attention_warrant1

        # concatenate them
        warrant_0minus1 = attention_warrant0 - attention_warrant1
        warrant_1minus0 = attention_warrant1 - attention_warrant0
        merge_warrant = tf.concat([warrant_1minus0, warrant_0minus1], axis=-1)
        merge_warrant = tf.concat([attention_warrant0, attention_warrant1, attention_warrant0 - attention_warrant1, attention_warrant0 * attention_warrant1], axis=-1)
        dropout_warrant = tf.nn.dropout(merge_warrant, self.drop_keep_rate)

        # and add one extra layer with ReLU
        with tf.variable_scope("linear") as s:
            dense1 = tf.nn.relu(tf_utils.linear(dropout_warrant, int(self.lstm_size / 2), bias=True, scope='dense'))
            logits = tf_utils.linear(dense1, self.num_class, bias=True, scope='logit')

        # Obtain the Predict, Loss, Train_op
        predict_prob = tf.nn.softmax(logits, name='predict_prob')
        predict_label = tf.cast(tf.argmax(logits, axis=1), tf.int32)

        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.target_label)
        loss = tf.reduce_mean(loss)

        # Build the loss
        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        # optimizer = tf.train.AdagradOptimizer(self.learning_rate)

        if FLAGS.clipper:
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), FLAGS.clipper)
            train_op = optimizer.apply_gradients(list(zip(grads, tvars)), global_step=global_step)
        else:
            train_op = optimizer.minimize(loss, global_step=global_step)

        self.predict_prob = predict_prob
        self.predict_label = predict_label
        self.loss = loss
        self.train_op = train_op
        self.global_step = global_step

    def train_model(self, sess, batch):
        feed_dict = {
            self.input_warrant0 : batch.warrant0,
            self.input_warrant1 : batch.warrant1,
            self.input_reason : batch.reason,
            self.input_claim : batch.claim,
            self.input_debate : batch.debate,

            self.warrant0_len : batch.warrant0_len,
            self.warrant1_len : batch.warrant1_len,
            self.reason_len : batch.reason_len,
            self.claim_len : batch.claim_len,
            self.debate_len : batch.debate_len,

            self.input_diff_warrant0 : batch.diff_warrant0,
            self.input_diff_warrant1 : batch.diff_warrant1,
            self.diff_warrant0_len: batch.diff_warrant0_len,
            self.diff_warrant1_len: batch.diff_warrant1_len,

            self.target_label : batch.label,
            self.drop_keep_rate : 0.9,
            self.learning_rate : 1e-3,
        }
        to_return = {
            'train_op': self.train_op,
            'loss': self.loss,
            'global_step': self.global_step,
        }
        return sess.run(to_return, feed_dict)

    def test_model(self, sess, batch):
        feed_dict = {
            self.input_warrant0: batch.warrant0,
            self.input_warrant1: batch.warrant1,
            self.input_reason: batch.reason,
            self.input_claim: batch.claim,
            self.input_debate: batch.debate,
            self.warrant0_len: batch.warrant0_len,
            self.warrant1_len: batch.warrant1_len,
            self.reason_len: batch.reason_len,
            self.claim_len: batch.claim_len,
            self.debate_len: batch.debate_len,

            self.input_diff_warrant0: batch.diff_warrant0,
            self.input_diff_warrant1: batch.diff_warrant1,
            self.diff_warrant0_len: batch.diff_warrant0_len,
            self.diff_warrant1_len: batch.diff_warrant1_len,
            self.drop_keep_rate: 1.0
        }
        to_return = {
            'predict_label': self.predict_label,
            'predict_prob': self.predict_prob,
            'attention_warrant0': self.attention_warrant0,

        }
        return sess.run(to_return, feed_dict)