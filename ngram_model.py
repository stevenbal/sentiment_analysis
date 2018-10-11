import collections
#import operator
#import sys
import os
import csv
import pickle
import re
import math
from functools import reduce
from nltk.stem import PorterStemmer
from copy import copy

#TODO toy example with small text to check if ngram computation is correct!!

#separate class for dicts
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
            #self.model_order = N
            self.models = self.make_models(source, N)
        elif model_file:
            self.words, self.stemming, self.models = pickle.load(open(model_file, "rb"))
            
    #TODO Separate class for nested dicts
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
                    line = re.sub(r'[^A-z0-9\s]', '', line)
                    line = "<s> " + line + " </s>"
                    line = line.lower().split() if self.words else line.lower()
                    if self.stemming:
                        line = [stemmer.stem(word) for word in line]
                    for j in range(1, N+1):
                        for i in range(j, len(line)+1):
                            #words = line[0:i] if i-j < 0 else line[i-j:i]
                            words = line[i-j:i]
                            add_to_model(models[j-1], words, 1)
                text.close()
        for i in range(len(models)):
            models[i] = dict(models[i])
        return models

    def compute_prob(self, sentence, N=None):
        sentence = re.sub(r'[^A-z0-9\s]', '', sentence)
        sentence = "<s> " + sentence + " </s>"
        sentence = sentence.lower().split() if self.words else sentence.lower()
        if self.stemming:
            stemmer = PorterStemmer()
            sentence = [stemmer.stem(word) for word in sentence]
        sentence_prob = 1
        if not N:
            N = len(self.models)
        for i in range(1, len(sentence)+1):
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
        self.models = {model[0]: model[1] for model in models}

    def classify(self, sentence, mixture=[]):
        if mixture:
            if max(mixture) > len(self.models['positive'].get_models()):
                raise ModelOrderError(max(mixture))
            results = []
            for order in mixture:
                probs = {category: model.compute_prob(sentence, N=order) for category, model in self.models.items()}
                results.append(max(probs, key=probs.get))
            #print(results)
            return collections.Counter(results).most_common(1)[0][0]
        else:
            probs = {category:model.compute_prob(sentence) for category, model in self.models.items()}
            #print(probs)
            return max(probs, key=probs.get)

    def evaluate(self, filename, mixture=0):
        results = {'positive': {'true': 0, 'false': 0}, 'negative': {'true': 0, 'false': 0}}
        i = 0
        with open(filename, errors='replace') as csvfile:
            datareader = csv.reader(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for line, sentiment in datareader:
                #i += 1
                #if i > 10:
                #    break
                prediction = self.classify(line, mixture=mixture)
                if prediction == sentiment:
                    results[sentiment]['true'] += 1
                else:
                    results[prediction]['false'] += 1
            csvfile.close()
        print('precision {}, recall {}'.format(self.compute_precision(results), self.compute_recall(results)))
        return results

    def compute_precision(self, results):
        if (results['positive']['true'] + results['positive']['false']):
            precision = results['positive']['true'] / (results['positive']['true'] + results['positive']['false'])
        else:
            precision = 0
        return precision

    def compute_recall(self, results):
        if (results['positive']['true'] + results['positive']['false']):
            recall = results['positive']['true'] / (results['positive']['true'] + results['negative']['false'])
        else:
            recall = 0
        return recall

class ModelOrderError(ValueError):
    pass
