import collections
import sys
import operator
import os
import cPickle as pickle
import re
import math
from nltk.stem import *
from copy import copy

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

class LanguageModel:
    def __init__(self, source="", N=2, words=True, stemming=False, model_file=""):
        if source:
            self.words = words
            self.stemming = stemming
            self.models = self.make_models(source, N)
        elif model_file:     
            self.words, self.stemming, self.models = pickle.load(open(model_file, "rb"))
    
    def instantiate_models(self, n):
        models = []
        current = int
        for N in range(n):
            current = lambda : collections.defaultdict(current)
            model = copy(current())
            models.append(model)
        return models
         
    def make_models(self, directory, N):
        if self.stemming:
            stemmer = PorterStemmer()
        models = self.instantiate_models(N)
        files = os.listdir(directory)
        for filename in files:
            with open(directory + '/' + filename) as text:
                for line in text:
                    line = re.sub(r'[^A-z\s]', '', line)
                    line = "<s> " + line + " </s>"
                    line = line.lower().split() if self.words else line.lower()
                    if self.stemming:
                        line = [stemmer.stem(word) for word in line]
                    for j in xrange(1, N+1):
                        for i in xrange(j, len(line)+1):
                            #words = line[0:i] if i-j < 0 else line[i-j:i]
                            words = line[i-j:i]
                            add_to_model(models[j-1], words, 1)
                text.close()
        for i in range(len(models)):
            models[i] = dict(models[i])
        return models
    
    def compute_prob(self, sentence, N=None):
        sentence = re.sub(r'[^A-z\s]', '', sentence)
        sentence = "<s> " + sentence + " </s>"
        sentence = sentence.lower().split() if self.words else sentence.lower()
        if self.stemming:
            stemmer = PorterStemmer()
            sentence = [stemmer.stem(word) for word in sentence]
        sentence_prob = 1
        if not N:
            N = len(self.models)
        for i in xrange(1, len(sentence)+1):
            words = sentence[0:i] if i-N < 0 else sentence[i-N:i]
            sentence_prob += math.log(get_relative_freq(self.models[:N], words))
        return sentence_prob

    def get_models(self):
        return self.models

    def set_models(self, models):
        self.models = models

    def save_models(self, filename):
        pickle.dump([self.words, self.stemming, self.models], open(filename, "wb"))

    def load_models(self, filename):
        self.models = pickle.load(open(filename, "rb"))

class Classifier:
    def __init__(self, *models):
        self.models = {model[0] : model[1] for model in models}

    def classify(self, sentence, mixture=0):
        if mixture:
            results = []
            for i in range(1, mixture+1):
                probs = {category:model.compute_prob(sentence, N=i) for category, model in self.models.items()}
                results.append(max(probs, key=probs.get))
            return collections.Counter(results).most_common(1)[0][0]
        else:
            probs = {category:model.compute_prob(sentence) for category, model in self.models.items()}
            return max(probs, key=probs.get)

    '''def test_accuracy(self, filename, sentiment, mixture=0):
        total = 0
        correct = 0
        with open(filename) as text:
            for line in text.readlines():
                if self.classify(line, mixture=mixture) == sentiment:
                    correct += 1
                total += 1
            text.close()
        return correct / float(total)'''
    
    def test_accuracy(self, filename, sentiment, mixture=0):
        with open(filename) as text:
            txt = text.readlines()
            results = [self.classify(line, mixture=mixture) for line in txt]
            total = len(txt)
            text.close()
        return results.count(sentiment) / float(total)
