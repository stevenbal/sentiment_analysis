import collections
import sys
import operator
import os
import re

class LanguageModel:
    def __init__(self, source, words=True):
        self.words = words
        self.unigrams, self.bigrams = self.make_model(source)

    def make_model(self, directory):
        bigrams = collections.Counter()
        unigrams = collections.Counter()
        files = os.listdir(directory)
        for filename in files:
            with open(directory + '/' + filename) as text:
                for line in text:
                    line = "<s> " + line + " </s>"
                    #line = line.lower().split()
                    line = re.sub(r'[^A-z\s]', '', line)
                    line = line.lower().split() if self.words else line.lower()
                    for i in xrange(0, len(line)-1):
                        [word1, word2] = line[i:i+2]
                        unigrams[word1] += 1
                        if i is len(line)-1:
                            unigrams[word2] += 1
                        if word1 not in bigrams:
                            bigrams[word1] = {}
                        if word2 in bigrams[word1]:
                            bigrams[word1][word2] += 1
                        else:
                            bigrams[word1][word2] = 1
                text.close()
        return (unigrams, bigrams)

    def compute_prob(self, sentence):
        sentence = "<s> " + sentence + " </s>"
        sentence = re.sub(r'[^A-z\s]', '', sentence)
        sentence = sentence.lower().split() if self.words else sentence.lower()
        sentence_prob = 1
        for i in xrange(0, len(sentence)-1):
            w1 = sentence[i]
            w2 = sentence[i+1]
            if w1 in self.bigrams:
                if w2 in self.bigrams[w1]:
                    sentence_prob *= self.bigrams[w1][w2] / float(self.unigrams[w1])
                else:
                    sentence_prob *= 1 / float(self.unigrams[w1])
            else:
                sentence_prob *= 1 / float(sum([sum(self.bigrams[word1].itervalues()) for word1 in self.bigrams]))
        return sentence_prob

    def get_models(self):
        return (self.unigrams, self.bigrams)



class Classifier:
    def __init__(self, *models):
        self.models = {model[0] : model[1] for model in models}

    def classify(self, sentence):
        probs = {category:model.compute_prob(sentence) for category, model in self.models.items()}
        return max(probs, key=probs.get)

    def test_accuracy(self, filename, sentiment):
        total = 0
        correct = 0
        with open(filename) as text:
            for line in text.readlines():
                if self.classify(line) == sentiment:
                    correct += 1
                total += 1
            text.close()
        return correct / float(total)
