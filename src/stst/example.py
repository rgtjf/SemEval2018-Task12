# coding: utf8
from __future__ import print_function


class Example(object):

    def __init__(self, example_dict):
        """
        suppose to do a text classifier task, and example is a list of words.
        score is the topic class of the example
        """
        self.example = example_dict['example']
        self.score = example_dict['score']

    def get_words(self):
        """ Return a list of words """
        return self.example
    
    def get_score(self):
        """ Return the gold score """
        return self.score
    
    def get_instance_string(self):
        """Abstract Method 
        Must Used in model.py
        """
        example = self.example
        instance_string = str(self.score) + '\t' + example
        return instance_string
