import os
import _pickle as pickle
import re
import math
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from copy import copy

from NestedDict import NestedDict

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
        -stemming:          bool, indicates whether the words are to be stemmed (default false)
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

    def get_relative_freq(self, models, words):
        length = len(words)
        voc_size = len(models[0].values())
        if length == 1:
            return (models[0].get_by_path(words) + 1) / float(sum(models[0].values()) + voc_size)
        else:
            ngram_freq = models[length-1].get_by_path(words)
            n_min_one_freq = models[length-2].get_by_path(words[:-1])
            return (ngram_freq + 1) / float(n_min_one_freq + voc_size)

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
        models = [NestedDict() for _ in range(N)]
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
                            #add_to_model(models[j-1], words, 1)
                            models[j-1].add_by_path(words, 1)
                text.close()
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
            sentence_prob += math.log(self.get_relative_freq(self.models[:N], words))
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
