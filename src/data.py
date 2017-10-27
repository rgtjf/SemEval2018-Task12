# coding: utf8

"""
@author: rgtjf
@file: data.py
@time: 2017/10/23 13:38
"""

from __future__ import print_function

import stst
import codecs


class Example(stst.Example):
    """
    Example:
        an argument: (claim [str/List/json], reason)
        select: warrant (explain the reason of the argument)
        debate_title, debate_info
    """
    def __init__(self, example_dict):
        self.id = example_dict['id']
        self.claim = example_dict['claim'].split()
        self.reason = example_dict['reason'].split()
        self.warrant0 = example_dict['warrant0'].split()
        self.warrant1 = example_dict['warrant1'].split()
        self.title = example_dict['title'].split()
        self.info  = example_dict['info'].split()
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

    def get_instance_string(self):
        instance_string = "{}\t{}\t{}\t{}\t{}".format(self.label,
                                                      ' '.join(self.warrant0), ' '.join(self.warrant1),
                                                      ' '.join(self.claim), ' '.join(self.reason))
        return instance_string

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
                example = Example(example_dict)
                examples.append(example)
        return examples


def calc_avg_tokens(train_instances):
    warrant0 = []
    warrant1 = []
    reason = []
    claim = []
    title = []
    info = []
    for train_instance in train_instances:
        warrant0.append(len(train_instance.get_warrant0()))
        warrant1.append(len(train_instance.get_warrant1()))
        reason.append(len(train_instance.get_reason()))
        claim.append(len(train_instance.get_claim()))
        title.append(len(train_instance.get_title()))
        info.append(len(train_instance.get_info()))

    print(sum(warrant0) / len(warrant0))
    print(sum(warrant1) / len(warrant1))
    print(sum(reason) / len(reason))
    print(sum(claim) / len(claim))
    print(sum(title) / len(title))
    print(sum(info) / len(info))


def load_parse_data(file_path, init=False):
    """
    Load data after Parse, like POS, NER, etc.
    Value: List of Example:class
    Parameter:
    """

    ''' Pre-Define Write File '''
    parse_train_file = file_path.replace('data', 'parse')
    parse_word_file = file_path.replace('data', 'word')
    parse_lemma_file = file_path.replace('data', 'lemma')
    parse_pos_file = file_path.replace('data', 'pos')
    parse_ner_file = file_path.replace('data', 'ner')
    parse_stopwords_lemma_file = file_path.replace('data', 'stopwords/lemma')

    if flag:

        print(file_path)
        print(gs_file)

        ''' Parse Data '''
        if 'sts' in file_path:
            data = load_STS(file_path)
        elif 'sick' in file_path or sick:
            data = load_SICK(file_path)
        else:
            data = load_data(file_path, gs_file)

        print('*' * 50)
        print("Parse Data, train_file=%s, gs_file=%s, n_train=%d\n" % (file_path, gs_file, len(data)))

        # idx = 0
        parse_data = []
        process_bar = pyprind.ProgPercent(len(data))
        for (sa, sb, score) in data:
            process_bar.update()
            # idx += 1
            # if idx > 20:
            #     break
            try:
                parse_sa = nlp.parse(sa)
                parse_sb = nlp.parse(sb)
            except Exception:
                print(sa, sb)
                traceback.print_exc()
                #parse_sa = sa
                #parse_sb = sb
            parse_data.append((parse_sa, parse_sb, score))

        ''' Write Data to File '''

        f_parse = utils.create_write_file(parse_train_file)
        f_word = utils.create_write_file(parse_word_file)
        f_lemma = utils.create_write_file(parse_lemma_file)
        f_pos = utils.create_write_file(parse_pos_file)
        f_ner = utils.create_write_file(parse_ner_file)
        f_stopwords_lemma = utils.create_write_file(parse_stopwords_lemma_file)

        for parse_instance in parse_data:
            line = json.dumps(parse_instance)

            sentpair_instance = SentPair(parse_instance)

            score = str(sentpair_instance.get_score())
            sa_word, sb_word = sentpair_instance.get_word(type='word')
            sa_lemma, sb_lemma = sentpair_instance.get_word(type='lemma')
            sa_pos, sb_pos = sentpair_instance.get_word(type='pos')
            sa_ner, sb_ner = sentpair_instance.get_word(type='ner')

            sa_stopwords_lemma, sb_stopwords_lemma = sentpair_instance.get_word(type='lemma', stopwords=True)

            s_word = score \
                     + '\t' \
                     + ' '.join([w + '/' + p for w, p in zip(sa_word, sa_word)])  \
                     + '\t'                                                      \
                     + ' '.join([w + '/' + p for w, p in zip(sb_word, sb_word)])

            s_lemma = score \
                     + '\t' \
                     + ' '.join([w + '/' + p for w, p in zip(sa_word, sa_lemma)]) \
                     + '\t' \
                     + ' '.join([w + '/' + p for w, p in zip(sb_word, sb_lemma)])

            s_pos = score \
                     + '\t' \
                     + ' '.join([w + '/' + p for w, p in zip(sa_word, sa_pos)]) \
                     + '\t' \
                     + ' '.join([w + '/' + p for w, p in zip(sb_word, sb_pos)])

            s_ner = score \
                     + '\t' \
                     + ' '.join([w + '/' + p for w, p in zip(sa_word, sa_ner)]) \
                     + '\t' \
                     + ' '.join([w + '/' + p for w, p in zip(sb_word, sb_ner)])

            s_stopwords_lemma = score \
                     + '\t' \
                     + ' '.join([w for w in sa_stopwords_lemma]) \
                     + '\t' \
                     + ' '.join([w for w in sb_stopwords_lemma]) \

            print(line, file=f_parse)
            print(s_word, file=f_word)
            print(s_lemma, file=f_lemma)
            print(s_pos, file=f_pos)
            print(s_ner, file=f_ner)
            print(s_stopwords_lemma, file=f_stopwords_lemma)

        f_parse.close()
        f_word.close()
        f_lemma.close()
        f_pos.close()
        f_ner.close()
        f_stopwords_lemma.close()

    ''' Load Data from File '''

    print('*' * 50)


    parse_data = []
    with codecs.open(parse_train_file, 'r', encoding='utf8') as f:
        for line in f:
            parse_json = json.loads(line)
            sentpair_instance = SentPair(parse_json)
            parse_data.append(sentpair_instance)

    print("Load Data, train_file=%s, gs_file=%s, n_train=%d\n" % (file_path, gs_file, len(parse_data)))
    return parse_data


if __name__ == '__main__':
    print('Train')
    train_file = '../data/train-full.txt'
    train_instances = Example.load_data(train_file)
    calc_avg_tokens(train_instances)

    print('Dev')
    dev_file = '../data/dev-full.txt'
    dev_instances = Example.load_data(dev_file)
    calc_avg_tokens(dev_instances)

    train_swap_file = '../data/train-w-swap-full.txt'
    train_instances = Example.load_data(train_swap_file)

    print(len(train_instances))
    print(train_instances[0].get_instance_string())

    for train_instance in train_instances[:10]:
        print(train_instance.get_instance_string())

    # for idx in range(len(train_instances)):
    #     print(idx, end=',')
    #     if idx % 2 == 1:
    #         a = train_instances[idx-1]
    #         b = train_instances[idx]
    #
    #         if a.get_warrant0() == b.get_warrant1() and a.get_warrant1() == b.get_warrant0():
    #             if a.get_label() + b.get_label() == 1:
    #                 print('sc', end=',')
    #             else:
    #                 raise ValueError
    #         else:
    #             print(b.get_instance_string())
