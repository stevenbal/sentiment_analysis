import collections
#import operator
#import sys
import os
import csv
import _pickle as pickle
import re
import math
from functools import reduce
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from copy import copy

#TODO toy example with small text to check if ngram computation is correct!!

# separate class for dicts
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
    def __init__(self, source="", N=2, words=True, stemming=False, stopword_removal=True, model_file=""):
        """
        Description:        constructor for an n-gram LanguageModel object with given parameters

        Input:
        -source:            str, the directory containing the text files of the
                            corpus used to construct the n-gram model
        -N:                 int, the order of the model (default: 2)
        -words:             bool, indicates whether the n-gram model is created
                            from words or characters (default: True)
        -stopword_removal:  bool, indicates whether stopwords are not stored
                            in the language model (default: True)
        -model_file:        str, can be used instead of source, indicates the
                            filename of a previously constructed language model
                            which will be loaded if specified
        """
        if source:
            self.words = words
            self.stemming = stemming
            self.stopword_removal = stopword_removal
            self.models = self.make_models(source, N)
        elif model_file:
            self.words, self.stemming, self.stopword_removal, self.models = pickle.load(open(model_file, "rb"))

    def __repr__(self):
        """
        Description:    function that shows the representation of the class instance

        Output:
        -object_string: string that shows the model parameters of an instance
        """
        parameters = [len(self.models), self.words, self.stemming, self.stopword_removal]
        object_string = "LanguageModel(N={}, words={}, stemming={}, stopword_removal={})".format(*parameters)
        return object_string

    # TODO Separate class for nested dicts
    def instantiate_models(self, N):
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
        return models

    def make_models(self, directory, N):
        """
        Description:    function that constructs the n-gram model for a given order
                        and a given corpus directory

        Input:
        -directory:     str, the directory containing the text files of the
                        corpus used to construct the n-gram model
        -N:             int, the order of the model

        Output:
        -models:        list, contains n-gram models from order 1 until N, constructed
                        from text files from a corpus directory
        """
        if self.stemming:
            stemmer = PorterStemmer()
        if self.stopword_removal:
            stopwords_english = list(set(stopwords.words('english')))
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
                    if self.stopword_removal:
                        line = [x for x in line if x not in stopwords_english]
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
        """
        Description:        function that computes the log probability of a sentence
                            for a language model

        Input:
        -sentence:          str, the sentence to be classified
        -N:                 int, if specified, uses this order of n-gram model to
                            compute the probability, else uses the max order of model (default: None)

        Output:
        -sentence_prob:     float, the log probability of the sentence
        """
        sentence = re.sub(r'[^A-z0-9\s]', '', sentence)
        sentence = "<s> " + sentence + " </s>"
        sentence = sentence.lower().split()
        if self.stopword_removal:
            stopwords_english = list(set(stopwords.words('english')))
            sentence = [x for x in sentence if x not in stopwords_english]
        if self.stemming:
            stemmer = PorterStemmer()
            sentence = [stemmer.stem(word) for word in sentence]
        if not self.words:
            sentence = ' '.join(sentence)
        sentence_prob = 1
        if not N:
            N = len(self.models)
        for i in range(1, len(sentence)+1):
            words = sentence[0:i] if i-N < 0 else sentence[i-N:i]
            sentence_prob += math.log(get_relative_freq(self.models[:N], words))
        return sentence_prob

    def get_models(self):
        """
        Description:    returns the list of n-gram models

        Output:
        -self.models:   list, contains the n-gram models of this language model
        """
        return self.models

    def save_models(self, filename):
        """
        Description:    stores the n-gram models and parameters in a pickled file
                        that can be loaded later into a LanguageModel instance

        Input:
        -filename:      str, the name of the file in which the data will be stored
        """
        pickle.dump([self.words, self.stemming, self.stopword_removal, self.models], open(filename, "wb"))

    def load_models(self, filename):
        """
        Description:    loads a previously saved model and parameters into an
                        LanguageModel instance

        Input:
        -filename:      str, the name of the file containing the model and parameters
        """
        self.models = pickle.load(open(filename, "rb"))

class Classifier:
    def __init__(self, *models):
        """
        Description:    constructor for a Classifier object, given one or more
                        LanguageModel objects

        Input:
        -*models:       tuple, any number of tuples containing the class the
                        LanguageModel object represents, and the LanguageModel object
        """
        self.models = {model[0]: model[1] for model in models}

    def classify(self, sentence, mixture=[]):
        """
        Description:        function that returns the most probable class for a given
                            sentence

        Input:
        -sentence:          str, the sentence to be classified
        -mixture:           list, contains integers specifying the orders of models
                            to be used as a mixture model, the class is selected by
                            majority vote in this case (default: [])

        Output:
        -predicted_class:   str, the class as predicted by the classifier
        """
        if mixture:
            if max(mixture) > len(self.models['positive'].get_models()):
                raise ModelOrderError(max(mixture))
            results = []
            for order in mixture:
                probs = {category: model.compute_prob(sentence, N=order) for category, model in self.models.items()}
                results.append(max(probs, key=probs.get))
            predicted_class = collections.Counter(results).most_common(1)[0][0]
            return predicted_class
        else:
            probs = {category:model.compute_prob(sentence) for category, model in self.models.items()}
            predicted_class = max(probs, key=probs.get)
            return predicted_class

    def evaluate(self, filename, mixture=[]):
        """
        Description:    function that evaluates the performance of the classifier
                        for a given test corpus

        Input:
        -filename:      str, the name of the csv file containing the test corpus
        -mixture:       list, specifies the mixture model (default: [])

        Output:
        -results:       dict, contains the counts of true and false positives
                        and true and false negatives
        """
        results = {'positive': {'true': 0, 'false': 0}, 'negative': {'true': 0, 'false': 0}}
        i = 0
        with open(filename, errors='replace') as csvfile:
            datareader = csv.reader(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for line, sentiment in datareader:
                prediction = self.classify(line, mixture=mixture)
                if prediction == sentiment:
                    results[sentiment]['true'] += 1
                else:
                    results[prediction]['false'] += 1
            csvfile.close()
        print('precision {}, recall {}'.format(self.compute_precision(results), self.compute_recall(results)))
        return results

    def compute_precision(self, results):
        """
        Description:    function that computes precision given the classification results

        Input:
        -results:       dict, contains the counts of true and false positives
                        and true and false negatives

        Output:
        -precision:     float, precision as produced by classifier
        """
        if (results['positive']['true'] + results['positive']['false']):
            precision = results['positive']['true'] / (results['positive']['true'] + results['positive']['false'])
        else:
            precision = 0
        return precision

    def compute_recall(self, results):
        """
        Description:    function that computes recall given the classification results

        Input:
        -results:       dict, contains the counts of true and false positives
                        and true and false negatives

        Output:
        -recall:        float, recall as produced by classifier
        """
        if (results['positive']['true'] + results['positive']['false']):
            recall = results['positive']['true'] / (results['positive']['true'] + results['negative']['false'])
        else:
            recall = 0
        return recall

class ModelOrderError(ValueError):
    """Indicates that an invaled order for a mixture model was given"""
    pass
