# coding: utf8
from __future__ import print_function

import os
import argparse
import numpy as np
from collections import Counter
import sys
sys.path.append('../src')
sys.path.append('../src_nn')
import evaluation

class Classify(object):

    def __init__(self, strategy):
        self.strategy = strategy

    def __call__(self, inputs):
        return self.strategy(inputs)


class Strategy(object):

    def __init__(self):
        pass

    def __call__(self, inputs):
        pass


class Vote(Strategy):

    def __init__(self):
        super(Vote, self).__init__()

    def __call__(self, inputs):
        predicts, golds = [], []
        # (a1, b1, c1), (a2, b2, c2), (a3, b3, c3)
        for input in inputs:
            predicts.append(input[1])
            golds.append(input[2])
        gold = golds[0]
        # predicts: [3-fold, n_data]
        predicts = np.array(predicts, dtype=np.int32).transpose()
        vote_predict = []
        for predict in predicts:
            counter = Counter(predict)
            vote_predict.append(counter.most_common(1)[0][0])
        acc = evaluation.Evalation_list(gold, vote_predict, print_all=True)
        return acc


class VoteEnsemble(Strategy):

    def __init__(self):
        self.trainer = "Vote Ensemble"
        print("Using %s Classifier" % (self.trainer))

    def train_model(self, train_file_path, model_path):
        pass

    def test_model(self, test_file_path, model_path, result_file_path):
        print("==> Load the data ...")
        # X_test, Y_test = self.load_file(test_file_path)
        # print(test_file_path, shape(X_test))
        # X_test = X_test.toarray()
        # for x in X_test[:10]:
        #     print(x)
        #
        # print("==> Test the model ...")
        # y_pred = []
        # for x in X_test:
        #     x = sum(x) / len(x)
        #     y_pred.append(x)
        #
        # print("==> Save the result ...")
        # with utils.create_write_file(result_file_path) as f:
        #     for y in y_pred:
        #         print(y, file=f)
        # return y_pred

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    args, _ = parser.parse_known_args()

    devs, tests = [], []
    for file in os.listdir(args.output_dir):
        if 'dev' in file and 'tsv' in file:
            file = os.path.join(args.output_dir, file)
            with open(file) as fr:
                dev_acc = float(fr.readline().strip().split()[1])
                predicts, golds = [], []
                for line in fr:
                    items = line.strip().split('\t#\t')
                    predict = int(items[0])
                    gold = int(items[1].strip('\t')[0])
                    predicts.append(predict)
                    golds.append(gold)
            devs.append((dev_acc, predicts, golds))
        elif 'test' in file and 'tsv' in file:
            file = os.path.join(args.output_dir, file)
            with open(file) as fr:
                dev_acc = float(fr.readline().strip().split()[1])
                predicts, golds = [], []
                for line in fr:
                    items = line.strip().split('\t#\t')
                    predict = int(items[0])
                    gold = int(items[1].strip('\t')[0])
                    predicts.append(predict)
                    golds.append(gold)
            tests.append((dev_acc, predicts, golds))

    classify =  Classify(Vote())
    acc = classify(devs)
    # acc = classify(tests)
    print(acc)
