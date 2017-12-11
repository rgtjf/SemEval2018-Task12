# coding: utf8

"""
@author: rgtjf
@file: evaluation.py
@time: 2017/9/30 14:33
"""

from __future__ import print_function

import argparse

from metric.confusion_matrix import Alphabet, ConfusionMatrix

DICT_LABEL_TO_INDEX = {'0': 0, '1'  : 1}
DICT_INDEX_TO_LABEL = {index:label for label, index in DICT_LABEL_TO_INDEX.items()}


def Evaluation(gold_file_path, predict_file_path):
    with open(gold_file_path) as gold_file, open(predict_file_path) as predict_file:
        gold_list = [line.strip().split('\t')[3] for line in gold_file][1:]
        predicted_list = [ line.strip().split('\t')[0] for line in predict_file][0:]

        binary_alphabet = Alphabet()
        for i in range(2):
            binary_alphabet.add(DICT_INDEX_TO_LABEL[i])

        cm = ConfusionMatrix(binary_alphabet)
        cm.add_list(predicted_list, gold_list)
        acc = cm.print_out()
        return acc

if __name__ == '__main__':

    parser = argparse.ArgumentParser(usage='python evaluation.py --predict_file results.tsv '
                                           '--gold_file ../data/dev-only-labels.txt')
    parser.add_argument('--predict_file', type=str, required=True)
    parser.add_argument('--gold_file', type=str, required=True)

    args,_ = parser.parse_known_args()
    Evaluation(args.predict_file, args.gold_file)