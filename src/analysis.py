# coding: utf8

import config
import codecs


def print_cases(start=0, end=30):
    with codecs.open('../data/test-only-data.txt', encoding='utf8') as f:
        lines = f.readlines()
        lines = lines[start:end]
        idx = start
        for line in lines:
            # #id	warrant0	warrant1	correctLabelW0orW1	reason	claim	debateTitle	debateInfo
            items = line.strip().split('\t')
            print('w0:' + items[1])
            print('w1:' + items[2])
            print(' ')
            print('r:' + items[3])
            print('c:' + items[4])
            print('t:' + items[5])
            print('i:' + items[6])
            idx += 1
            print('-' * 50 + str(idx-1))

# print_cases(30, 60)


def print_topic(topic='Have Comment Sections Failed?'):
    with codecs.open('../data/test-only-data.txt', encoding='utf8') as f:
        lines = f.readlines()
        lines = lines[1:]
        idx = 1
        for line in lines:
            # #id	warrant0	warrant1	correctLabelW0orW1	reason	claim	debateTitle	debateInfo
            items = line.strip().split('\t')
            idx += 1
            if items[5] == topic:
                print('w0:' + items[1])
                print('w1:' + items[2])
                print(' ')
                print('r:' + items[3])
                print('c:' + items[4])
                print('t:' + items[5])
                print('i:' + items[6])
                print('-' * 50 + str(idx-1))

# print_cases(30, 60)

print_topic()