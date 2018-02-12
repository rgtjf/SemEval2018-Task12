# coding: utf8
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import config
import tf_utils
from attention_lstm import AttBasicLSTMCell
from attention_lstm import AttGRUCell


class RelationModel(object):

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
        embedded_warrant0 = tf.nn.embedding_lookup(self.we, self.input_warrant0)
        embedded_warrant1 = tf.nn.embedding_lookup(self.we, self.input_warrant1)
        embedded_reason = tf.nn.embedding_lookup(self.we, self.input_reason)
        embedded_claim = tf.nn.embedding_lookup(self.we, self.input_claim)
        embedded_debate = tf.nn.embedding_lookup(self.we, self.input_debate)

        embedded_diff_warrant0 = tf.nn.embedding_lookup(self.we, self.input_diff_warrant0)
        embedded_diff_warrant1 = tf.nn.embedding_lookup(self.we, self.input_diff_warrant1)

        def conv_ngram(input_x, filter_sizes=(1, 2, 3), num_filters=32):
            """
            Conv ngram
            """
            sent_len = input_x.get_shape()[1]
            embed_size = input_x.get_shape()[2]
            input_x = tf.expand_dims(input_x, axis=-1)
            outputs = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.variable_scope("conv-maxpool-%s" % filter_size):
                    filter_shape = [filter_size, embed_size, 1, num_filters]
                    W = tf.get_variable("W", filter_shape, initializer=tf.random_normal_initializer())
                    b = tf.get_variable("b", [num_filters], initializer=tf.constant_initializer(0.0))
                    conv = tf.nn.conv2d(input_x, W, strides=[1, 1, embed_size, 1], padding='SAME', name="conv")
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    h = tf.squeeze(h, axis=2)
                    outputs.append(h)
            outputs = tf.concat(outputs, axis=2)
            return outputs

        with tf.variable_scope("conv") as s:
            ''' Conv Layer '''
            conv_warrant0 = conv_ngram(embedded_warrant0)
            s.reuse_variables()
            conv_warrant1 = conv_ngram(embedded_warrant1)
            conv_reason = conv_ngram(embedded_reason)
            conv_claim = conv_ngram(embedded_claim)
            conv_debate = conv_ngram(embedded_debate)
            conv_diff_warrant0 = conv_ngram(embedded_diff_warrant0)
            conv_diff_warrant1 = conv_ngram(embedded_diff_warrant1)

        with tf.variable_scope("pooling") as s:
            ''' Pooling Layer '''
            pooling_warrant0 = tf_utils.MaxPooling(conv_warrant0, self.warrant0_len)
            pooling_warrant1 = tf_utils.MaxPooling(conv_warrant1, self.warrant1_len)
            pooling_reason = tf_utils.MaxPooling(conv_reason, self.reason_len)
            pooling_claim = tf_utils.MaxPooling(conv_claim, self.claim_len)
            pooling_debate = tf_utils.MaxPooling(conv_debate, self.debate_len)

            pooling_diff_warrant0 = tf_utils.MaxPooling(conv_diff_warrant0, self.diff_warrant0_len)
            pooling_diff_warrant1 = tf_utils.MaxPooling(conv_diff_warrant1, self.diff_warrant1_len)

        attention_vector_for_W0 = tf.concat(
                [pooling_debate, pooling_reason, pooling_claim], axis=-1)
        attention_vector_for_W1 = tf.concat(
                [pooling_debate, pooling_reason, pooling_claim], axis=-1)

        with tf.variable_scope("relaton") as s:
            pooling_warrant0 = tf.concat([pooling_warrant0, pooling_diff_warrant0], axis=-1)
            attention_warrant0 = tf_utils.mlp_attention_vec(attention_vector_for_W0, pooling_warrant0)
            s.reuse_variables()
            pooling_warrant1 = tf.concat([pooling_warrant1, pooling_diff_warrant1], axis=-1)
            attention_warrant1 = tf_utils.mlp_attention_vec(attention_vector_for_W1, pooling_warrant1)

        self.attention_warrant0 = attention_warrant0
        self.attention_warrant1 = attention_warrant1

        # concatenate them
        merge_warrant = tf.concat([pooling_reason * pooling_claim,
                                   attention_warrant0, attention_warrant1,
                                   attention_warrant0 - attention_warrant1,
                                   attention_warrant0 * attention_warrant1], axis=-1)
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
        train_op = tf_utils.optimize(loss, 'adam', FLAGS.lambda_l2, self.learning_rate, global_step, FLAGS.clipper)

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

            self.target_label: batch.label,
            self.drop_keep_rate: 1.0
        }
        to_return = {
            'predict_label': self.predict_label,
            'predict_prob': self.predict_prob,
            'attention_warrant0': self.attention_warrant0,
            'loss': self.loss
        }
        return sess.run(to_return, feed_dict)