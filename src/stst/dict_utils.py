# coding: utf8
from __future__ import print_function

import ast

from stst import utils


@utils.singleton
class DictLoader(object):
    def __init__(self):
        self.dict_manager = {}

    def load_dict(self, dict_name, dict_file_path, sep='\t'):
        """
        manage the dict from list or dict
        list index start from 1
        """
        if dict_name not in self.dict_manager:

            dict_object = {}

            ''' load dict from file '''
            print('load dict from file {}'.format(dict_file_path))

            f_dict = utils.create_read_file(dict_file_path)

            for idx, line in enumerate(f_dict):
                if line[0] == '[' and line[-1] == ']':
                    continue
                line = line.strip().split(sep)
                if len(line) == 1:
                    dict_object[line[0]] = idx + 1
                elif len(line) == 2:
                    # NOT eval, for the value may be str
                    # dict_object[line[0]] = line[1]
                    dict_object[line[0]] = ast.literal_eval(line[1])
                else:
                    raise NotImplementedError

            self.dict_manager[dict_name] = dict_object

        return self.dict_manager[dict_name]


class DictCreater(object):

    def __init__(self):
        pass

    def _create_dict(func):
        """ Python decorator """

        def __create_dict(*args, **kwargs):
            print("====> create dict for function [{}]".format(func.__name__))

            ret = func(*args, **kwargs)

            ''' remove item whose frequency is less than threshold '''
            if 'threshold' in kwargs:
                threshold = kwargs['threshold']
                for key in ret.keys():
                    if ret[key] < threshold:
                        ret.pop(key)

            ''' write dict to file '''
            file_name = 'dict_{}.txt'.format(func.__name__)
            f_dict = utils.create_write_file(file_name)

            if type(ret) == list:
                # ensure it is set
                ret = list(set(ret))
                ret = sorted(ret)
                for idx, item in enumerate(ret):
                    print(str(item), file=f_dict)

            elif type(ret) == dict:
                # order the dict
                for item in sorted(ret.keys()):
                    print(item, ret[item])
                    print('%s\t%s' % (item, ret[item]), file=f_dict)
            else:
                raise NotImplementedError

            f_dict.close()

            print("====> write file {}, {:d}    instances".format(file_name, len(ret)))
            return ret

        return __create_dict

    @_create_dict
    def stopwords(self):
        """ Take Stopwords for Example """
        fp = open('stanford.stopwords.txt', 'r')
        stanford_stopwords = [line.strip('\r\n ') for line in fp.readlines()]
        stopwords = stanford_stopwords
        return stopwords

if __name__ == '__main__':
    pass