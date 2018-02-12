# coding: utf8

from input import data
import stst
from features.pmi_feature import *
from features.warrant_feature import *
from metric import evaluation

classifier = stst.Classifier(stst.LIB_LINEAR_LR())
model = stst.Model('NLP', classifier)

model.add(Warrant_Feature(load=False))
# model.add(BowFeature(load=False))
# model.add(BI_feature(load=False))

train_file = config.train_file
train_instances = data.load_parse_data(train_file)

dev_file = config.dev_file
dev_instances = data.load_parse_data(dev_file)

test_file = config.test_file
test_instances = data.load_parse_data(test_file)


model.train(train_instances, train_file)
acc = evaluation.Evaluation(train_file, model.output_file)
print(acc)


model.test(dev_instances, dev_file)
acc = evaluation.Evaluation(dev_file, model.output_file)
print(acc)

model.test(test_instances, test_file)
acc = evaluation.Evaluation(test_file, model.output_file)
print(acc)

#
# with open(config.dev_file) as f:
#     f.readline()
#     # #id	warrant0	warrant1	correctLabelW0orW1	reason	claim	debateTitle	debateInfo
#     # #id	correctLabelW0orW1
#     print('#id\tcorrectLabelW0orW1')
#     for line in f:
#         id = line.strip().split('\t')[0]
#         print('%s\t%d' % (id, random.randint(0, 1)))
#
# print(evaluation.Evaluation('./results.tsv', config.dev_label_file) )
