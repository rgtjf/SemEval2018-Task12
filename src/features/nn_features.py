# coding: utf8
from __future__ import print_function

import os
from stst import Feature
from stst import utils
import config


class NNFeature(Feature):

    def __init__(self, nn_name, run_dir, **kwargs):
        super(NNFeature, self).__init__(**kwargs)
        self.nn_name = nn_name
        self.run_dir = run_dir
        self.feature_name += '-%s' % (nn_name)

    def extract_instances(self, train_instances):

        features = [None] * len(train_instances)
        infos = [None] * len(train_instances)

        if 'train' in self.train_file:
            phrase = 'train'
        elif 'dev' in self.train_file:
            phrase = 'dev'
        elif 'test' in self.train_file:
            phrase = 'test'
        else:
            raise NotImplementedError

        best_dev = 0.
        for file in os.listdir(self.run_dir):
            if phrase in file and 'tsv' in file:
                print(file)
                file_path = os.path.join(self.run_dir, file)
                with open(file_path) as fr:
                    dev_acc = float(fr.readline().strip().split()[1])
                    if best_dev < dev_acc:
                        best_dev = dev_acc
                        for idx, line in enumerate(fr):
                            items = line.strip().split('\t#\t')
                            predict = int(items[0])
                            features[idx] = [predict]
                            # if features[idx] is None:
                            #     features[idx] = [predict]
                            # else:
                            #     features[idx].append(predict)


        print(len(features), features[0])

        return features, infos


class NNAVGFeature(Feature):

    def __init__(self, nn_name, run_dir, **kwargs):
        super(NNAVGFeature, self).__init__(**kwargs)
        self.nn_name = nn_name
        self.run_dir = run_dir
        self.feature_name += '-%s' % (nn_name)

    def extract_instances(self, train_instances):

        features = [None] * len(train_instances)
        infos = [None] * len(train_instances)

        if 'train' in self.train_file:
            phrase = 'train'
        elif 'dev' in self.train_file:
            phrase = 'dev'
        elif 'test' in self.train_file:
            phrase = 'test'
        else:
            raise NotImplementedError

        for file in os.listdir(self.run_dir):
            if phrase in file and 'tsv' in file:
                print(file)
                file_path = os.path.join(self.run_dir, file)
                with open(file_path) as fr:
                    dev_acc = float(fr.readline().strip().split()[1])
                    for idx, line in enumerate(fr):
                        items = line.strip().split('\t#\t')
                        predict = int(items[0])
                        if features[idx] is None:
                            features[idx] = [predict]
                        else:
                            features[idx].append(predict)

        print(len(features), features[0])

        return features, infos