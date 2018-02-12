# coding: utf-8
from __future__ import print_function

import random
import numpy as np
import tensorflow as tf
from tensorboard_logger import Logger

import config
import evaluation
import utils
from task import Task
from models.intra_attention_cnn import IntraAttentionCNNModel
from models.intra_attention_cnn_margin import IntraAttentionCNNMarginModel
from models.intra_attention_cnn_wodiff import IntraAttentionCNNWoDiffModel
from models.intra_attention import IntraAttentionModel
from models.intra_attention_ii import IntraAttentionIIModel
from models.relation import RelationModel
from models.intra_attention_cnn_negclaim import IntraAttentionCNNNegClaimModel
FLAGS = tf.flags.FLAGS

# File Parameters
tf.flags.DEFINE_string('log_file', 'SemEval18.log', 'path of the log file')
# tf.flags.DEFINE_string('summary_dir', 'summary', 'path of summary_dir')
tf.flags.DEFINE_string('description', __file__, 'commit_message')
tf.flags.DEFINE_string('task', 'word2vec-lemma', 'task name')

# Task Parameters
tf.flags.DEFINE_string('model', 'intra_attention_ii', 'given the model name')
tf.flags.DEFINE_integer('max_epoch', 5, 'max epoches')
tf.flags.DEFINE_integer('display_step', 20, 'display each step')
tf.flags.DEFINE_boolean('expand', False, 'whether use the expand train data')
tf.flags.DEFINE_boolean('init', False, 'whether use the expand train data')

# Hyper Parameters
tf.flags.DEFINE_string('rnn_type', 'lstm', 'rnn_type: lstm / gru')
tf.flags.DEFINE_integer('batch_size', 32, 'batch size')
tf.flags.DEFINE_float('drop_keep_rate', 0.8, 'drop_keep_rate')
tf.flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
tf.flags.DEFINE_float('lambda_l2', 1e-3, 'lambda_l2')
tf.flags.DEFINE_float('clipper', 30, 'clipper')

FLAGS._parse_flags()

random.seed(1234)
np.random.seed(1234)
tf.set_random_seed(1234)

# Tensorboard Part
name = utils.get_time_name('run_{}'.format(FLAGS.model))
train_logger = Logger('runs/{}/train'.format(name))
dev_logger = Logger('runs/{}/dev'.format(name))
test_logger = Logger('runs/{}/test'.format(name))

# Predict Part
train_predict_file = 'runs/{}/train_predict.tsv'.format(name)
dev_predict_file = 'runs/{}/dev_predict.tsv'.format(name)
test_predict_file = 'runs/{}/test_predict.tsv'.format(name)

# Logger Part
# Change the log_file
FLAGS.log_file = 'runs/{}/SemEval18.log'.format(name)
logger = utils.get_logger(FLAGS.log_file)

logger.info(FLAGS.__flags)


def train_and_dev(model, task, run_name='run_0'):

    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True

    best_dev_result = 0.

    with tf.Session(config=gpu_config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        total_batch = 0

        def do_eval(dataset):
            batch_size = 100  # for Simple
            num_data = len(dataset.correct_label_w0_or_w1_list)
            preds, golds = [], []
            for batch in dataset.batch_iter(batch_size):
                results = model.test_model(sess, batch)
                preds.append(results['predict_label'])
                golds.append(np.argmax(batch.label, 1))

            preds = np.concatenate(preds, 0)
            golds = np.concatenate(golds, 0)
            predict_labels = [config.id2category[predict] for predict in preds]
            gold_labels = [config.id2category[gold] for gold in golds]

            predict_labels = predict_labels[:num_data]
            gold_labels = gold_labels[:num_data]

            acc = evaluation.Evalation_list(gold_labels, predict_labels)
            return acc, predict_labels

        dev_acc, dev_predicts = do_eval(task.dev_data)
        # test_acc = do_eval(task.test_data)
        logger.info('dev = {:.5f}'.format(dev_acc))

        for epoch in range(FLAGS.max_epoch):
            for batch in task.train_data.batch_iter(FLAGS.batch_size, shuffle=True):
                total_batch += 1
                results = model.train_model(sess, batch)
                step = int(results['global_step'])
                loss = results['loss']

                train_logger.log_value(run_name, loss, step)

                if total_batch % FLAGS.display_step == 0:
                    print('batch_{} steps_{} cost_val: {:.5f}'.format(total_batch, step, loss))
                    # logger.info('==>  Epoch {:02d}/{:02d}: '.format(epoch, total_batch))

                    cum_loss = 0
                    for dev_batch in task.dev_data.batch_iter(100, shuffle=False):
                        results = model.test_model(sess, dev_batch)
                        length = len(dev_batch.label)
                        cum_loss += results['loss'] * length
                    cum_loss /= len(task.dev_data.instance_id_list)
                    dev_logger.log_value(run_name, cum_loss, step)

            train_acc, _ = do_eval(task.train_data)
            dev_acc, dev_predicts = do_eval(task.dev_data)
            test_acc, test_predicts = do_eval(task.test_data)
            logger.info('epoch = {:d}, train = {:5f}, dev = {:.5f}'.format(epoch, train_acc, dev_acc))

            if dev_acc > best_dev_result:
                best_dev_result = dev_acc
                logger.info('dev = {:.5f} best!!!!'.format(best_dev_result))
                # write predict to file for further analysis
                task.dev_data.write_out(dev_predicts, dev_predict_file.replace('predict', run_name), dev_acc)
                task.test_data.write_out(test_predicts, test_predict_file.replace('predict', run_name), dev_acc)

    return best_dev_result


def main():
    task = Task(task_name=FLAGS.task, init=FLAGS.init, expand=FLAGS.expand)
    FLAGS.we = task.embed

    if FLAGS.model == 'intra_attention_i':
        model = IntraAttentionModel(FLAGS)
    elif FLAGS.model == 'intra_attention_ii':
        model = IntraAttentionIIModel(FLAGS)
    elif FLAGS.model == 'intra_attention_cnn':
        model = IntraAttentionCNNModel(FLAGS)
    elif FLAGS.model == 'relation':
        model = RelationModel(FLAGS)
    elif FLAGS.model == 'intra_attention_cnn_wo':
        model = IntraAttentionCNNWoDiffModel(FLAGS)
    elif FLAGS.model == 'intra_attention_cnn_margin':
        model = IntraAttentionCNNMarginModel(FLAGS)
    elif FLAGS.model == 'intra_attention_cnn_negclaim':
        model = IntraAttentionCNNNegClaimModel(FLAGS)
    else:
        raise NotImplementedError

    accs = []
    for run in range(3):
        print('\n\n run: ', run)
        dev_acc = train_and_dev(model, task, 'run_{}'.format(run))
        accs.append(dev_acc)
    logger.info(accs)
    logger.info('mean = {:.5f}, std = {:.5f}'.format(np.mean(accs), np.std(accs)))

if __name__ == '__main__':
   main()