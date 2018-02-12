## The Argument Reasoning Comprehension Task

### 2017.11.07
dl methods
从nlp方法中知道关键点还是比较warrant0和warrant1（nlp光靠否定词到0.60）

- finish intra_attention model
```
1. merge[warrant0-warrant1, warrant1-warrant0] 0.68 不稳定
2. merge[warrant0, warrant1] 0.6867 稳定
3. merge[warrant0, warrant1] states返回h 0.68
4. merge[warrant0, warrant1, warrant0-warrant1, warrant0 * warrant1] 0.68 相差不大，考虑用最多的
```
- 对于warrant0和warrant1
```
1. 将warrant0和warrant1中不同的部分拿出来，用来计算attention，突出两个的不同
  - 没有的时候补['do'] 0.69
  - AttLSTM (avg_diff_warrant0, emb_warrant0)
  - attention vector ( ... avg_diff_warrant0)
```
- 问题：
```
1. dev 数据集特别少，怎么做评估（如何说明模型有效？）
   [x] 多次mean+std
   [.] loss图形显示
   [ ] 将train的一部分拿出来（0.9/0.1做交叉验证）
```
- Next
```
[ ] 考虑减少模型参数, e.g., LSTM -> GRU
[ ] 考虑从外部得到数据， e.g., word2vec / KG / Encoder
```

nlp methods
- finish negative features
```
1. 观察到warrant0和warrant1大部分将其添加了否定词(如直接+not)
   判断warrant0和warrant1是否含有否定词作为特征，train/dev上结果0.62/0.60
2. 添加判断warrant0 == reason!=claim，认为warrant0为否定时，应该和reason以及claim的一致性有关， train/dev的结果为0.62/0.61
3. 考虑到unigram特征，将[warrant0 + reason + claim]和[warrant1 + reason + claim]分别做unigram表示，然后拼接，train/dev上为0.58/0.58
```
这部分可以看出着重需要考虑warrant0和warrant1之间不同的地方，dl做表示时可以很好的得到不同的表示，为nlp则比较麻烦。
- Next
```
[ ] 考虑warrant0和(reason, claim)做bi-gram
```


### 2017.11.06
nlp methods
- BUG:
> FileNotFoundError: [Errno 2] No such file or directory: '../generate/outputs/NLP/train-w-swap-full.txt'

需要手工建立文件夹

### 2017.10.27
- 三段论
- 修改代码：
    - 原来的结果
        ```
        Acc dev
        0.633   0.589   0.608
        Acc test
        0.633   0.589   0.608
        ```
    ``get_attention_lstm_intra_warrant_A``
    - 将第一层LSTM编码进行共享, LSTM都是对句子编码，得到句子表示，这样可以减少参数。性能从0.63到0.66
       ```
       Acc dev
       0.661   0.658   0.636
       Acc test
       0.661   0.658   0.636
       ```
    - 更改dropout(0.9-0.8)，性能从0.66到0.65
       ```
        Acc dev
        0.658   0.658   0.646
        Acc test
        0.658   0.658   0.646
       ```
- 数据统计
    - Train
        ```
        warrant0 / warrant1 / reason / claim / title / info
        9.818181818181818
        9.85702479338843
        12.988429752066116
        5.737190082644628
        5.756198347107438
        17.58595041322314
        ```
    - Dev
        ```
        9.515822784810126
        9.810126582278482
        12.708860759493671
        6.734177215189874
        6.205696202531645
        16.93354430379747
        ```




### 2017.10.22 - 2017.10.24
- Task:
    Given: an argument (claim, reason)
    Select: warrant (explain the reason of the argument)
- Example:
> Argument: Miss America gives honors and education scholarships. And since ..., Miss America is good for women.
  - ✔ scholarships would give women a chance to study
  - ✗ scholarships would take women from the home

> Government is already struggling to pay for basic needs. And since
....., Sport leagues should not enjoy nonprofit
   - ✔ government isn’t required to pay for all the country’s needs
   - ✗ government is required to pay for the country’s needs
  - Title: Tax Break for Sports
  - Info: Should pro sports leagues enjoy nonprofit status?



### 2017.09.30 - 2017.10.07

- background https://arxiv.org/abs/1708.01425
- dataset https://github.com/UKPLab/argument-reasoning-comprehension-task/tree/master/mturk/annotation-task/data/exported-SemEval2018-train-dev-test
- results

Human average                  0.798+0.162
Human w/ training in reasoning 0.909+0.114
---
Random baseline 0.479 0.018 0.508+0.015
Language model  0.623             0.559
---
Intra-warrant attention w/ context 0.616+0.012  0.58+0.033

#### Download the data and run a baseline
1. /data/download.sh
2. run random and obrain 0.478

Train a Deep Leaning Model
- look at the reference

Train a NLP Model
- stst
- corenlp
- features


#### 2017.10.02
- how to build a model
```
 R + W  -> C
 R + ^W -> ^C
```
- diff W0 and W1
  取W0和W1的公共部分，和差异部分，分别取出来（后面做一个统计，分析出是不是差异都是否定词）
