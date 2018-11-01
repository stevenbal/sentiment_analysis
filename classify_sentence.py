import sys
from time import time

import resources.LanguageModel as ngram
import resources.NaiveBayesClassifier as NBclassifier

if len(sys.argv) > 1:
    sentence = sys.argv[1]
else:
    sentence = input('Please enter a sentence: ')

# Load the ngram models for negative and positive sentiments, n=3 and stopwords
# are omitted

LM_pos = ngram.LanguageModel(model_file='models/positive_n3_nostopwords.p')
LM_neg = ngram.LanguageModel(model_file='models/negative_n3_nostopwords.p')

# Construct classifier from the two models
classifier = NBclassifier.NaiveBayesClassifier(('positive', LM_pos), ('negative', LM_neg))

# Determine the sentiment of the given sentence, only use bigram model
print(classifier.classify(sentence, mixture=[1, 2]))
