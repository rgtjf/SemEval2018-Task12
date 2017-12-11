
category2id = {'0': 0, '1': 1}
id2category = {index: label for label, index in category2id.items()}


max_sent_len = 100
num_class = 2
lstm_size = 64
max_diff_len = 10


ROOT_DIR = '/home/junfeng/SemEval18/task12'
# [Data]
DATA_DIR = ROOT_DIR + '/data'
train_file = DATA_DIR + '/train-w-swap-full.txt'
dev_file = DATA_DIR + '/dev-full.txt'
dev_label_file = DATA_DIR + '/dev-only-labels.txt'

word_embed_file = '/home/junfeng/word2vec/word2vec.300d.txt'
glove_file = '/home/junfeng/GloVe/glove/glove.840B.300d.txt'
paragram_file = '/home/junfeng/paragram-embedding/paragram_300_sl999.txt'
fasttext_file = '/home/junfeng/FastText/wiki.en.vec'

word_dim = 300


# [Output]

def get_w2i_we_file(task_name):
    DIR = DATA_DIR + '/' + task_name
    w2i_file = DIR + '/w2i.p'
    we_file = DIR + 'we.p'
    return w2i_file, we_file

OUTPUT_DIR = ROOT_DIR + '/nn_outputs'
w2i_file = OUTPUT_DIR + '/w2i.p'
we_file = OUTPUT_DIR + '/we.p'

train_predict_file = OUTPUT_DIR + '/train-predict.txt'
dev_predict_file = OUTPUT_DIR + '/dev-predict.txt'

SAVE_DIR = ROOT_DIR + '/nn_save'

# [Resource]
RESOURCE_DIR = ROOT_DIR + '/src/resources'
negation_term_file = RESOURCE_DIR + '/dict_negation_terms.txt'
negative_word_file = RESOURCE_DIR + '/dict_negative_words.txt'


