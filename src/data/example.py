# coding: utf8
from __future__ import print_function

import json
import codecs
import unicodedata
from nltk.tokenize.casual import TweetTokenizer
from stst.dict_utils import DictLoader

def nltk_tokenize(s):
    """
    Tokenization of the given text using TweetTokenizer delivered along with NLTK
    :param s: text
    :return: list of tokens
    """
    sentence_splitter = TweetTokenizer()
    tokens = sentence_splitter.tokenize(s)
    result = []
    for word in tokens:
        # the last "decode" function is because of Python3
        # http://stackoverflow.com/questions/2592764/what-does-a-b-prefix-before-a-python-string-mean
        w = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8').strip()
        # and add only if not empty (it happened in some data that there were empty tokens...)
        if w:
            result.append(w)

    return result


def tokenize(s, method='split'):
    result = []
    if method == 'split':
        result = s.split()
    elif result == 'nltk':
        result = nltk_tokenize(s)
    else:
        raise NotImplementedError
    return result


class Example(object):
    """
    Example:
        an argument: (claim [str/List/json], reason)
        select: warrant (explain the reason of the argument)
        debate_title, debate_info
    """
    def __init__(self, example_dict):
        self.id = example_dict['id']
        self.claim = tokenize(example_dict['claim'])
        self.reason = tokenize(example_dict['reason'])
        self.warrant0 = tokenize(example_dict['warrant0'])
        self.warrant1 = tokenize(example_dict['warrant1'])
        self.title = tokenize(example_dict['title'])
        self.info  = tokenize(example_dict['info'])
        self.debate = tokenize(example_dict['debate'])
        self.label = int(example_dict['label'])

    def get_label(self):
        return self.label

    def get_claim(self):
        return self.claim

    def get_reason(self):
        return self.reason

    def get_warrant0(self):
        return self.warrant0

    def get_warrant1(self):
        return self.warrant1

    def get_title(self):
        return self.title

    def get_info(self):
        return self.info

    def get_id(self):
        return self.id

    def get_instance_string(self):
        instance_string = "{}\t{}\t{}\t{}\t{}".format(self.label,
                                                      ' '.join(self.warrant0), ' '.join(self.warrant1),
                                                      ' '.join(self.claim), ' '.join(self.reason))
        return instance_string

    def get_six(self, return_str=False):
        if return_str:
            return ' '.join(self.warrant0), ' '.join(self.warrant1), ' '.join(self.reason), \
                    ' '.join(self.claim), ' '.join(self.title), ' '.join(self.info)
        else:
            return self.warrant0, self.warrant1, self.reason, self.claim, self.title, self.info

    def get_all(self):
        return self.id, self.warrant0, self.warrant1, self.label, self.reason, self.claim, self.debate

    @staticmethod
    def load_data(file_path):
        """ return list of examples """
        examples = []
        with codecs.open(file_path, encoding='utf8') as f:
            #id	warrant0	warrant1	correctLabelW0orW1	reason	claim	debateTitle	debateInfo
            headline = f.readline()
            for line in f:
                example_dict = {}
                items = line.strip().split('\t')
                example_dict['id'] = items[0]
                example_dict['warrant0'] = items[1]
                example_dict['warrant1'] = items[2]
                example_dict['label'] = items[3]
                example_dict['reason'] = items[4]
                example_dict['claim'] = items[5]
                example_dict['title'] = items[6]
                example_dict['info'] = items[7]
                example_dict['debate'] = items[6] + ' ' + items[7]
                example = Example(example_dict)
                examples.append(example)
        return examples


class ParseExample():
    """ Parse Example """

    def __init__(self, example_json):
        """
        list -> class
        Args:
            example_json: list, [id, label, ...]
        """
        self.id, self.label, self._warrant0,  self.warrant0, self._warrant1, self.warrant1, \
            self._reason, self.reason,  self._claim, self.claim, self._title, self.title, \
            self._info, self.info = example_json

        self.label = int(self.label)

    def get_words(self, parse_sent, **kwargs):
        """
        Given parse_sent, return the object
        Args:
            kwargs: type=word/lemma/pos/ner, stopwords=True/False, lower=True/False
        Returns:
            sent: List / Str
        """
        sent = []

        if 'stopwords' in kwargs and kwargs['stopwords'] is True:
            stopwords_file = 'resources/dict_stopwords.txt'
            tokens = parse_sent["sentences"][0]["tokens"]
            sent = [token[kwargs["type"]] for token in tokens if
                    token['word'].lower() not in DictLoader().load_dict('stopwords', stopwords_file)]

            if len(sent) == 0:
                sent = [token[kwargs["type"]] for token in tokens]

        else:
            tokens = parse_sent["sentences"][0]["tokens"]
            sent = [token[kwargs["type"]] for token in tokens]

        if 'lower' in kwargs and kwargs['lower'] is True:
            sent = [w.lower() for w in sent]

        if 'return_str' in kwargs and kwargs['return_str'] is True:
            sent = ' '.join(sent)
        return sent

    def get_label(self):
        return self.label

    def get_claim(self, **kwargs):
        return self.get_words(self.claim, **kwargs)

    def get_reason(self, **kwargs):
        return self.get_words(self.reason, **kwargs)

    def get_warrant0(self, **kwargs):
        return self.get_words(self.warrant0, **kwargs)

    def get_warrant1(self, **kwargs):
        return self.get_words(self.warrant1, **kwargs)

    def get_title(self, **kwargs):
        return self.get_words(self.title, **kwargs)

    def get_info(self, **kwargs):
        return self.get_words(self.info, **kwargs)

    def get_id(self):
        return self.id

    def get_sent(self, name, **kwargs):
        if name == 'warrant0':
            sent = self.get_warrant0(**kwargs)
        elif name == 'warrant1':
            sent = self.get_warrant1(**kwargs)
        elif name == 'reason':
            sent = self.get_reason(**kwargs)
        elif name == 'claim':
            sent = self.get_claim(**kwargs)
        elif name == 'title':
            sent = self.get_title(**kwargs)
        elif name == 'info':
            sent = self.get_info(**kwargs)
        else:
            raise NotImplementedError
        return sent

    def get_instance_string(self):
        instance_string = "{}\t{}\t{}\t{}\t{}".format(self.label,
                                                      ' '.join(self.warrant0), ' '.join(self.warrant1),
                                                      ' '.join(self.claim), ' '.join(self.reason))
        return instance_string

    def get_six(self, return_str=False, **kwargs):
        """
        warrant0, warrant1, reason, claim, title, info = train_instance.get_six(type='word', return_str=True)
        Args:
            return_str: False, return list; True, return str
        """
        warrant0 = self.get_warrant0(**kwargs)
        warrant1 = self.get_warrant1(**kwargs)
        reason = self.get_reason(**kwargs)
        claim = self.get_claim(**kwargs)
        title = self.get_title(**kwargs)
        info = self.get_info(**kwargs)

        if return_str:
            return ' '.join(warrant0), ' '.join(warrant1), ' '.join(reason), \
                    ' '.join(claim), ' '.join(title), ' '.join(info)
        else:
            return warrant0, warrant1, reason, claim, title, info

    @staticmethod
    def load_data(file_path):
        """
        Load Data from File
        Returns:
            list of parse examples
        """
        parse_file_path = file_path.replace('data', 'generate/parse')
        print('*' * 50)
        parse_data = []
        with codecs.open(parse_file_path, 'r', encoding='utf8') as f:
            for line in f:
                # obtain the json object
                parse_sent = json.loads(line)
                # obtain the class
                parse_example = ParseExample(parse_sent)
                parse_data.append(parse_example)
        print("Load Data, file_path=%s  n_line=%d\n" % (parse_file_path, len(parse_data)))
        return parse_data