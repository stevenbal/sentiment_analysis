from copy import copy
import collections
from functools import reduce
from collections import defaultdict

class NestedDict(dict):
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value

    # def set_by_path(self, path, value):
    #     length = len(path)
    #     res = self
    #     for i, key in enumerate(path):
    #         if i == length - 1:
    #             res[key] = value
    #         else:
    #             res = res[key]
    #     #reduce(lambda x, y: x.setdefault(y, value), path, self)

    def get_by_path(self, path):
        value = reduce(lambda x, y: x.get(y, {}), path, self)
        return value if value else 0

    def set_by_path(self, path, value):
        for key in path[:-1]:
            self = self.setdefault(key, {})
        self[path[-1]] = value

    def add_by_path(self, path, value):
        for key in path[:-1]:
            self = self.setdefault(key, {})
        self[path[-1]] = self[path[-1]] + value if path[-1] in self else value


def instantiate_models(N):
    """
    Description:    function that creates empty nested dictionaries of order
                    1 until N, used to store the n-gram models of that order

    Input:
    -N:             int, the order of the complete n-gram model

    Output:
    -models:        list, contains empty nested dictionaries of order 1 until N
    """
    models = []
    current = int
    for _ in range(N):
        current = lambda : collections.defaultdict(current)
        model = copy(current())
        models.append(model)
        print(model)
    return models

def add_to_model(dic, keys, value):
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = dic[keys[-1]] + value if keys[-1] in dic else value

def get_value(dic, path):
    try:
        value = reduce(dict.get, path, dic)
        return value if value else 0
    except:
        return 0

def get_relative_freq(models, words):
    length = len(words)
    voc_size = len(models[0].values())
    if length == 1:
        return (get_value(models[0], words) + 1) / float(sum(models[0].values()) + voc_size)
    else:
        ngram_freq = get_value(models[length-1], words)
        n_min_one_freq = get_value(models[length-2], words[:-1])
        return (ngram_freq + 1) / float(n_min_one_freq + voc_size)

a,b,c = instantiate_models(3)

from time import time

# path = ['a','b','c']
# X = 1000000
# N = NestedDict()
# t1 = time()
# for i in range(X):
#     N.add_by_path(path, i)
# t2 = time()
# print((t2-t1)/X)
# print(N)
#
# path = ['a','b','c']
# X = 1000000
# N = NestedDict()
# t1 = time()
# for i in range(X):
#     N.add_by_path2(path, i)
# t2 = time()
# print((t2-t1)/X)
# print(N)
# #

m = [NestedDict() for _ in range(3)]
m[0].add_by_path(['a', 'b', 'c'], 10)
print(m)
