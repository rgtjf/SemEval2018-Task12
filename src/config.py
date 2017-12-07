ROOT_DIR = '/home/junfeng/SemEval18/task12'
# [Data]
DATA_DIR = ROOT_DIR + '/data'
train_file = DATA_DIR + '/train-w-swap-full.txt'
dev_file = DATA_DIR + '/dev-full.txt'
dev_label_file = DATA_DIR + '/dev-only-labels.txt'

# [Output]
DATA_DIR = ROOT_DIR + '/outputs'
train_predict_file = DATA_DIR + '/train-predict.txt'
dev_predict_file = DATA_DIR + '/dev-predict.txt'

# [Resource]
RESOURCE_DIR = ROOT_DIR + '/src/resources'
negation_term_file = RESOURCE_DIR + '/dict_negation_terms.txt'
negative_word_file = RESOURCE_DIR + '/dict_negative_words.txt'
