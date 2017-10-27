q## The Argument Reasoning Comprehension Task


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
