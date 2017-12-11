import random
import numpy as np
import re, os
import pickle

from tqdm import tqdm
import pyprind

pad_word = '__PAD__'
unk_word = '__UNK__'


def save_params(params, fname):
    """
    Pickle uses different protocols to convert your data to a binary stream.
    - In python 2 there are 3 different protocols (0, 1, 2) and the default is 0.
    - In python 3 there are 5 different protocols (0, 1, 2, 3, 4) and the default is 3.
    You must specify in python 3 a protocol lower than 3 in order to be able to load
    the data in python 2. You can specify the protocol parameter when invoking pickle.dump.
    """
    if os.path.exists(fname):
        os.remove(fname)
    with open(fname, 'wb') as fw:
        pickle.dump(params, fw, protocol=2)


def load_params(fname):
    if not os.path.exists(fname):
        raise RuntimeError('no file: %s' % fname)
    with open(fname, 'rb') as fr:
        params = pickle.load(fr)
    return params


def make_batches(size, batch_size):
    """
    make batch index according to batch_size and size
    :param size: the size of dataset
    :param batch_size: the size of batch
    :return: list: [(0, batch_size), (batch_size, 2*batch_size), ..., (. , min(., .))]
    """
    nb_batch = int(np.ceil(size/float(batch_size)))
    return [(i*batch_size, min(size, (i+1)*batch_size)) for i in range(0, nb_batch)]


def vectorize(score, num_class):
    """
    NOT suitable for classification
    during classification, the index usually starts from zero, however (score=1, num_classer=3) -> [1, 0, 0]
    :param score: 1.2 (0, 2)
    :param num_class: 3
    :return: one-hot represent: [0.8, 0.2, 0.0] * [1, 2, 0]
    """
    one_hot = np.zeros(num_class, dtype=float)
    score = float(score)
    ceil, floor = int(np.ceil(score)), int(np.floor(score))
    if ceil == floor:
        one_hot[floor - 1] = 1
    else:
        one_hot[floor - 1] = ceil - score
        one_hot[ceil - 1] = score - floor
    one_hot = one_hot + 0.00001
    return one_hot


def onehot_vectorize(label, num_class):
    """
    For classification
    during classification, the index usually starts from zero, however (score=1, num_classer=3) -> [1, 0, 0]
    :param score: 1.2 (0, 2)
    :param num_class: 3
    :return: one-hot represent: [0.8, 0.2, 0.0] * [1, 2, 0]
    """
    one_hot = np.zeros(num_class, dtype=float)
    one_hot[label] = 1.0
    return one_hot


def sent_to_index(sent, word_vocab):
    """

    :param sent:
    :param word_vocab:
    :return:
    """
    sent_index = []
    for word in sent:
        if word not in word_vocab:
            sent_index.append(word_vocab[unk_word])
        else:
            sent_index.append(word_vocab[word])
    return sent_index


def pad_1d_vector(words, max_sent_len, dtype=np.int32):
    padding_words = np.zeros((max_sent_len, ), dtype=dtype)
    kept_length = len(words)
    if kept_length > max_sent_len:
        kept_length = max_sent_len
    padding_words[:kept_length] = words[:kept_length]
    return padding_words


def pad_2d_matrix(batch_words, max_sent_len=None, dtype=np.int32):
    """

    :param batch_words: [batch_size, sent_length]
    :param max_sent_len: if None, max(sent_length)
    :param dtype:
    :return: padding_words: [batch_size, max_sent_length], 0
    """

    if max_sent_len is None:
        max_sent_len = np.max([len(words) for words in batch_words])

    batch_size = len(batch_words)
    padding_words = np.zeros((batch_size, max_sent_len), dtype=dtype)

    for i in range(batch_size):
        words = batch_words[i]
        kept_length = len(words)
        if kept_length > max_sent_len:
            kept_length = max_sent_len
        padding_words[i, :kept_length] = words[:kept_length]
    return padding_words


def pad_3d_tensor(batch_chars, max_sent_length=None, max_word_length=None, dtype=np.int32):
    """

    :param batch_chars: [batch_size, sent_length, word_length]
    :param max_sent_length:
    :param max_word_length:
    :param dtype:
    :return:
    """
    if max_sent_length is None:
        max_sent_length = np.max([len(words) for words in batch_chars])

    if max_word_length is None:
        max_word_length = np.max([np.max([len(chars) for chars in words]) for words in batch_chars])

    batch_size = len(batch_chars)
    padding_chars = np.zeros((batch_size, max_sent_length, max_word_length), dtype=dtype)

    for i in range(batch_size):
        sent_length = max_sent_length

        if len(batch_chars[i]) < max_sent_length:
            sent_length = len(batch_chars[i])

        for j in range(sent_length):
            chars = batch_chars[i][j]
            kept_length = len(chars)
            if kept_length > max_word_length:
                kept_length = max_word_length
            padding_chars[i, j, :kept_length] = chars[:kept_length]
    return padding_chars


def build_word_vocab(sents):
    """

    :param sents:
    :return: word2index
    """
    words = set()
    for sent in sents:
        words.update(sent)
    words_vocab = {word:index+2 for index, word in enumerate(words)}
    words_vocab[pad_word] = 0
    words_vocab[unk_word] = 1
    return words_vocab


def build_char_vocab(sents):
    """

    :param sents:
    :return: char2index
    """
    chars = set()
    for sent in sents:
        for word in sent:
            word = list(word)
            chars.update(word)
    chars_vocab = {char:index+2 for index, char in enumerate(chars)}
    chars_vocab[pad_word] = 0
    chars_vocab[unk_word] = 1
    return chars


def load_fasttext_unk_words(oov_word_list, word2index, word_embedding):
    pass


def load_fasttext(word2index, emb_file, n_dim=100):
    """
    UPDATE_0: save the oov words in oov.p (pickle)
    Pros: to analysis why the this happen !!!
    ===
    :param word2index: dict, word2index['__UNK__'] = 0
    :param emb_file: str, file_path
    :param n_dim:
    :return: np.array(n_words, n_dim)
    """
    pass


def get_embedding_file_len(file_path):
    context = os.popen('wc -l %s' % (file_path)).read()
    line = int(context.split()[0])
    return line

def load_word_embedding_std(in_vocab, emb_file, n_dim=300):
    """
    load word embedding in a standard way:
    Args:
        in_vocab:
        emb_file:
        n_dim:

    Returns:
        word2index, embeddings
    """
    word2index = {}
    embeddings = []
    word2index[pad_word] = 0
    embeddings.append(np.zeros(n_dim))
    word2index[unk_word] = 1
    embeddings.append(np.random.uniform(-0.25, 0.25, (n_dim, )))
    word_index = 2
    print('Load word embedding: %s' % emb_file)

    total_line = get_embedding_file_len(emb_file)
    process_bar = pyprind.ProgPercent(total_line)

    with open(emb_file, 'r', errors='ignore') as f:
    # with open(emb_file, 'r') as f:
        for idx, line in enumerate(f):
            process_bar.update()
            # if the first line is (vocab, ndim)
            if idx == 0 and len(line.split()) == 2:
                continue
            # split the line
            sp = line.rstrip().split()
            if len(sp) != n_dim + 1:
                print(sp[0:len(sp) - n_dim])

            w = ''.join(sp[0:len(sp) - n_dim])
            emb = [float(x) for x in sp[len(sp) - n_dim:]]
            assert len(emb) == n_dim
            if w in in_vocab and w not in word2index:
                word2index[w] = word_index
                embeddings.append(emb)
                word_index += 1

    pre_trained_len = len(word2index)
    n_words = len(in_vocab)

    print('Pre-trained: {}/{} {:.2f}'.format(pre_trained_len, n_words, pre_trained_len * 100.0 / n_words))

    oov_word_list = [w for w in in_vocab if w not in word2index]
    print('oov word list example (30): ', oov_word_list[:30])
    pickle.dump(oov_word_list, open('./oov.p', 'wb'))

    embeddings = np.array(embeddings, dtype=np.float32)
    return word2index, embeddings


def load_word_embedding_my(word2index, emb_file, n_dim=300):
    """
    load word embedding in a tricky way:
    obtain the embedding equal to the len(word2index)
    Args:
        word2index:
        emb_file:
        n_dim:

    Returns:
        word2index, embeddings
    """

    print('Load word embedding: %s' % emb_file)

    pre_trained = {}
    n_words = len(word2index)

    embeddings = np.random.uniform(-0.25, 0.25, (n_words, n_dim))
    embeddings[0, ] = np.zeros(n_dim)

    total_line = get_embedding_file_len(emb_file)
    process_bar = pyprind.ProgPercent(total_line)

    # with open(emb_file, 'r', errors='ignore') as f:
    with open(emb_file, 'r') as f:
        for idx, line in enumerate(f):
            process_bar.update()
            # if the first line is (vocab, ndim)
            if idx == 0 and len(line.split()) == 2:
                continue
            # split the line
            sp = line.rstrip().split()
            if len(sp) != n_dim + 1:
                print(sp[0:len(sp) - n_dim])

            w = ''.join(sp[0:len(sp) - n_dim])
            emb = [float(x) for x in sp[len(sp) - n_dim:]]

            if w in word2index and w not in pre_trained:
                embeddings[word2index[w]] = emb
                pre_trained[w] = 1

    pre_trained_len = len(pre_trained)

    print('Pre-trained: {}/{} {:.2f}'.format(pre_trained_len, n_words, pre_trained_len * 100.0 / n_words))

    oov_word_list = [w for w in word2index if w not in pre_trained]
    print('oov word list example (30): ', oov_word_list[:30])
    pickle.dump(oov_word_list, open('./oov.p', 'wb'))

    embeddings = np.array(embeddings, dtype=np.float32)
    return word2index, embeddings


def load_word_embedding(word2index, emb_file, n_dim=300, method='std'):
    """
    Pros:
        1. save the oov words in oov.p, to further analysis.

    Args:
        word2index: dict, word2index / vocab_dict / idf_vocab
        emb_file: file, word2vec / glove / fasttext
        n_dim: int, 300
        method: 'std' / 'my'

    Returns:
        word2index: word2index['__PAD__'] = 1, word2index['__UNK__'] = 1
        embeddings: according to word2index
    """
    if method == 'std':
        word2index, embeddings = load_word_embedding_std(word2index, emb_file, n_dim)
    elif method == 'my':
        assert word2index[pad_word] == 0
        assert word2index[unk_word] == 1
        word2index, embeddings = load_word_embedding_my(word2index, emb_file, n_dim)
    else:
        raise NotImplementedError
    return word2index, embeddings


def load_embed_from_text(emb_file, n_dim):
    """
    load_embedding from raw text
    Args:
        emb_file:
        n_dim:

    Returns:
        word2index: dict
        embeddings: numpy
    """
    print('Load word embedding: %s' % emb_file)

    word2index = {}
    embeddings = []

    word2index[pad_word] = 0
    embeddings.append(np.zeros(n_dim))
    word2index[unk_word] = 1
    embeddings.append(np.random.uniform(-0.25, 0.25, (n_dim, )))
    word_index = 2

    total_line = get_embedding_file_len(emb_file)
    process_bar = pyprind.ProgPercent(total_line)

    with open(emb_file, 'r') as f:
        for idx, line in enumerate(f):
            process_bar.update()
            # if the first line is (vocab, ndim)
            if idx == 0 and len(line.split()) == 2:
                print('embedding info: ', line)
                continue

            # split the line
            sp = line.rstrip().split()
            if len(sp) != n_dim + 1:
                print(sp[0:len(sp) - n_dim])

            w = ''.join(sp[0:len(sp) - n_dim])
            emb = [float(x) for x in sp[len(sp) - n_dim:]]

            word2index[w] = word_index
            embeddings.append(emb)
            word_index += 1

    embeddings = np.array(embeddings, dtype=np.float32)
    print('finished load input embed!!')

    return word2index, embeddings


class Batch(object):
    """
    Tricks:
    1. setattr and getattr
    2. __dict__ and vars
    """
    def __init__(self):

        pass

    def add(self, name, value):
        setattr(self, name, value)

    def get(self, name):
        if name == 'self':
            value = self.__dict__  # or value = vars(self)
        else:
            value = getattr(self, name)
        return value
