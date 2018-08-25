import collections
import sys
import operator
import bigram_model as bigram
import os
import nltk
from nltk.corpus import wordnet as wn

if len(sys.argv) > 1:
    sentence = sys.argv[1]
else:
    sentence = input('Please enter a sentence: ')

LM_pos = bigram.LanguageModel('corpora/review_polarity/txt_sentoken/pos', N=3)
LM_neg = bigram.LanguageModel('corpora/review_polarity/txt_sentoken/neg', N=3)

classifier = bigram.Classifier(('positive', LM_pos), ('negative', LM_neg))

#print classifier.test_accuracy('corpora/rt-polaritydata/rt-polaritydata/rt-polarity.pos', 'positive')
#print classifier.test_accuracy('corpora/rt-polaritydata/rt-polaritydata/rt-polarity.neg', 'negative')

print(classifier.classify(sentence))

#testa

