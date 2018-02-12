# coding: utf8
import config
from input import data

def diffsents(sa, sb):
    """ tell the different part of a sentence pair"""
    m = len(sa)
    n = len(sb)
    la = lb = 0
    ra = m - 1
    rb = n - 1
    while la < m and lb < n:
        if sa[la] == sb[lb]:
            la += 1
            lb += 1
        else:
            break
    while ra >= 0 and rb >= 0:
        if sa[ra] == sb[rb]:
            ra -= 1
            rb -= 1
        else:
            break
    while la > ra or lb > rb:
        # la -= 1
        ra += 1
        # lb -= 1
        rb += 1
    if la == ra == m or lb == rb == n:
        la -= 1
        ra -= 1
        lb -= 1
        rb -= 1
    assert 0 <= la <= ra < m, "{}\t{}\t{}\t{}\t{}".format(m, la, ra, sa, sb)
    assert 0 <= lb <= rb < n, "{}\t{}\t{}\t{}\t{}".format(n, lb, rb, sb, sa)
    return sa[la: ra+1], sb[lb: rb+1]
    # return la, ra, lb, rb


train_file = config.train_file
train_instances = data.load_parse_data(train_file)

for train_instance in train_instances:
    word_type, stopwords, lower = 'lemma', False, True
    warrant0, warrant1, reason, claim, debate, negclaim = train_instance.get_six(type=word_type, stopwords=stopwords,
                                                                                 lower=lower)
    sa, sb = diffsents(warrant0, warrant1)
    if 'not' not in sa and 'not' not in sb:
        print('%s\t%s' % (' '.join(sa), ' '.join(sb)))
