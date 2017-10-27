# coding: utf8

import random
import config
import evaluation

with open(config.dev_file) as f:
    f.readline()
    # #id	warrant0	warrant1	correctLabelW0orW1	reason	claim	debateTitle	debateInfo
    # #id	correctLabelW0orW1
    print('#id\tcorrectLabelW0orW1')
    for line in f:
        id = line.strip().split('\t')[0]
        print('%s\t%d' % (id, random.randint(0, 1)))

print( evaluation.Evaluation('./results.tsv', config.dev_label_file) )
