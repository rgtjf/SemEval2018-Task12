# coding: utf8
from __future__ import print_function

import json

from pycorenlp.corenlp import StanfordCoreNLP


class StanfordNLP:
    def __init__(self, server_url='http://localhost:9000'):
        self.server = StanfordCoreNLP(server_url)

    def parse(self, text):
        output = self.server.annotate(text, properties={
            'timeout': '50000',
            'ssplit.isOneSentence': 'true',
            'depparse.DependencyParseAnnotator': 'basic',
            'annotators': 'tokenize,lemma,ssplit,pos,depparse,parse,ner',
            # 'annotators': 'tokenize,lemma,ssplit,pos,ner',
            'outputFormat': 'json'
        })

        return output

if __name__ == '__main__':

    nlp = StanfordNLP('http://precision:9000')
    text = '♡'
    print(text)
    parsetext = nlp.parse(u"Happy December 1st ♡ | #christmas #christmastree #pretty @ The Distillery Historic District")
    print(json.dumps(parsetext, indent=2))
    print((parsetext['sentences'][0]['tokens'][3]['word']))
