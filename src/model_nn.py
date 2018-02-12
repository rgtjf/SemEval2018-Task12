# coding: utf8

from input import data
import stst
from features.pmi_feature import *
from features.warrant_feature import *
from features.nn_features import *
from metric import evaluation

train_file = config.train_file
train_instances = data.load_parse_data(train_file)

dev_file = config.dev_file
dev_instances = data.load_parse_data(dev_file)

test_file = config.test_file
test_instances = data.load_parse_data(test_file)

lr = stst.Classifier(stst.LIB_LINEAR_LR())
nlp_model = stst.Model('lr', lr)
nlp_model.add(Warrant_Feature())

nlp_model.train(train_instances, train_file)
nlp_model.test(dev_instances, dev_file)
nlp_model.test(test_instances, test_file)

vote = stst.Classifier(stst.VoteEnsemble())
model1 = stst.Model('vote1', vote)

# model1.add(NNAVGFeature('intra_attention_cnn_margin', config.NN_RUN_DIR + '/run_intra_attention_cnn_margin_0121_19_04', load=False))  # 0.5
# model1.add(NNAVGFeature('intra_attention_cnn', config.NN_RUN_DIR + '/run_intra_attention_cnn_0121_19_03', load=False)) # 0.46
# model1.add(NNFeature('intra_attention_i', config.NN_RUN_DIR + '/run_intra_attention_i_0121_19_03', load=False)) # 0.5
# model1.add(NNAVGFeature('run_intra_attention_cnn_negclaim', config.NN_RUN_DIR + '/run_intra_attention_cnn_negclaim_0121_20_37', load=False))

model1.add(NNFeature('intra_attention_cnn_margin', config.NN_RUN_DIR + '/run_intra_attention_cnn_margin_0121_19_04', load=False))  # 0.60
model1.add(NNFeature('intra_attention_cnn', config.NN_RUN_DIR + '/run_intra_attention_cnn_0121_19_03', load=False)) # 0.46
model1.add(NNFeature('intra_attention_i', config.NN_RUN_DIR + '/run_intra_attention_i_0121_19_03', load=False)) # 0.5


model2 = stst.Model('vote2', vote)
model2.add(NNFeature('run_intra_attention_cnn_negclaim2', config.NN_RUN_DIR + '/run_intra_attention_cnn_negclaim_0123_22_45', load=False)) # 0.60
model2.add(NNFeature('run_intra_attention_cnn_negclaim3', config.NN_RUN_DIR + '/run_intra_attention_cnn_negclaim_0123_23_25', load=False)) # 0.60
model2.add(NNFeature('run_intra_attention_cnn_negclaim4', config.NN_RUN_DIR + '/run_intra_attention_cnn_negclaim_0124_00_05', load=False)) # 0.60


model3 = stst.Model('vote3', vote)
# model3.add(NNFeature('run_intra_attention_cnn_negclaim2', config.NN_RUN_DIR + '/run_intra_attention_cnn_negclaim_0123_22_45', load=False)) # 0.60
# model3.add(NNFeature('run_intra_attention_cnn_negclaim3', config.NN_RUN_DIR + '/run_intra_attention_cnn_negclaim_0123_23_25', load=False)) # 0.60
# model3.add(NNFeature('run_intra_attention_cnn_negclaim4', config.NN_RUN_DIR + '/run_intra_attention_cnn_negclaim_0124_00_05', load=False)) # 0.60
# model3.add(NNAVGFeature('intra_attention_cnn_margin', config.NN_RUN_DIR + '/run_intra_attention_cnn_margin_0121_19_04', load=False))  # 0.5
# model3.add(NNAVGFeature('intra_attention_cnn', config.NN_RUN_DIR + '/run_intra_attention_cnn_0121_19_03', load=False)) # 0.46
# model3.add(NNAVGFeature('intra_attention_i', config.NN_RUN_DIR + '/run_intra_attention_i_0121_19_03', load=False)) # 0.5
# model3.add(NNAVGFeature('intra_attention_ii', config.NN_RUN_DIR + '/run_intra_attention_ii_0121_19_03', load=False)) # 0.36
model3.add(NNAVGFeature('run_intra_attention_cnn_negclaim2', config.NN_RUN_DIR + '/run_intra_attention_cnn_negclaim_0123_22_45', load=False)) # 0.60
model3.add(NNAVGFeature('run_intra_attention_cnn_negclaim3', config.NN_RUN_DIR + '/run_intra_attention_cnn_negclaim_0123_23_25', load=False)) # 0.60
model3.add(NNAVGFeature('run_intra_attention_cnn_negclaim4', config.NN_RUN_DIR + '/run_intra_attention_cnn_negclaim_0124_00_05', load=False)) # 0.60

vote = stst.Classifier(stst.VoteEnsemble())
model = stst.Model('vote', vote)


# model.add(Warrant_Feature(load=False))
# model.add(BowFeature(load=False))
# model.add(BI_feature(load=False))
# model.add(NNFeature('intra_attention_cnn_margin', config.NN_RUN_DIR + '/run_intra_attention_cnn_margin_0121_19_04', load=False))  # 0.5
# model.add(NNFeature('intra_attention_cnn', config.NN_RUN_DIR + '/run_intra_attention_cnn_0121_19_03', load=False)) # 0.46
# model.add(NNFeature('intra_attention_cnn_wo', config.NN_RUN_DIR + '/run_intra_attention_cnn_wo_0121_19_04', load=False))  # 0.40
# model.add(NNFeature('intra_attention_i', config.NN_RUN_DIR + '/run_intra_attention_i_0121_19_03', load=False)) # 0.5
# model.add(NNFeature('intra_attention_ii', config.NN_RUN_DIR + '/run_intra_attention_ii_0121_19_03', load=False)) # 0.36

# model.add(NNAVGFeature('run_intra_attention_cnn_negclaim', config.NN_RUN_DIR + '/run_intra_attention_cnn_negclaim_0121_20_37', load=False)) # 0.53

# model.add(NNFeature('run_intra_attention_cnn_negclaim', config.NN_RUN_DIR + '/run_intra_attention_cnn_negclaim_0121_20_37', load=False)) # 0.60
# model.add(NNFeature('run_intra_attention_cnn_negclaim2', config.NN_RUN_DIR + '/run_intra_attention_cnn_negclaim_0123_22_45', load=False)) # 0.60
# model.add(NNFeature('run_intra_attention_cnn_negclaim3', config.NN_RUN_DIR + '/run_intra_attention_cnn_negclaim_0123_23_25', load=False)) # 0.60
# model.add(NNFeature('run_intra_attention_cnn_negclaim4', config.NN_RUN_DIR + '/run_intra_attention_cnn_negclaim_0124_00_05', load=False)) # 0.60

model.add(model1)
model.add(model2)
model.add(model3)


# model.add(model1)

# model.add(nlp_model) # 0.33

# model.train(train_instances, train_file)
# acc = evaluation.Evaluation(train_file, model.output_file)
# print(acc)

model.test(dev_instances, dev_file)
acc = evaluation.Evaluation(dev_file, model.output_file)
print(acc)

model.test(test_instances, test_file)
acc = evaluation.EvaluationTopK(test_file, model.output_file, 30)
print(acc)

evaluation.CaseStudyTopK(test_file, model.output_file, 30)
