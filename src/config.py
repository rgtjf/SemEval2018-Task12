# coding: utf8
import codecs

ROOT_DIR = '/home/junfeng/workspace/SemEval18'
# [Data]
DATA_DIR = ROOT_DIR + '/data'
train_file = DATA_DIR + '/train-w-swap-full.txt'
train_exp_file = DATA_DIR + '/train-w-swap-full-exp.txt'
dev_file = DATA_DIR + '/dev-full.txt'
dev_label_file = DATA_DIR + '/dev-only-labels.txt'
test_file = DATA_DIR + '/test-full-tmp.txt'
claim_file = DATA_DIR + '/claim.txt'

# [Output]
DATA_DIR = ROOT_DIR + '/outputs'
train_predict_file = DATA_DIR + '/train-predict.txt'
dev_predict_file = DATA_DIR + '/dev-predict.txt'

# [Resource]
RESOURCE_DIR = ROOT_DIR + '/src/resources'
negation_term_file = RESOURCE_DIR + '/dict_negation_terms.txt'
negative_word_file = RESOURCE_DIR + '/dict_negative_words.txt'


## [NN]
NN_RUN_DIR = ROOT_DIR + '/src_nn/runs'

# [Claim]
def load_claim(filepath):
    claim = {}
    with codecs.open(filepath, encoding='utf8') as f:
        for line in f:
            items = line.strip().split('\t')
            if len(items) != 2:
                print(line)
            claim[items[0]] = items[1]
            claim[items[1]] = items[0]
    return claim

claim_dict = load_claim(claim_file)