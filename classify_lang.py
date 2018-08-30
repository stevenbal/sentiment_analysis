#import collections
import sys
#import operator
import bigram_model as bigram
#import os
#import nltk
#from nltk.corpus import wordnet as wn
from time import time

if len(sys.argv) > 1:
    sentence = sys.argv[1]
else:
    sentence = input('Please enter a sentence: ')

#LM_pos = bigram.LanguageModel('corpora/review_polarity/txt_sentoken/pos', N=3)
#LM_neg = bigram.LanguageModel('corpora/review_polarity/txt_sentoken/neg', N=3)

LM_pos = bigram.LanguageModel(model_file='models/positive_n5.p')
LM_neg = bigram.LanguageModel(model_file='models/negative_n5.p')
#LM_pos_char = bigram.LanguageModel(model_file='models/positive_char_n5.p')
#LM_neg_char = bigram.LanguageModel(model_file='models/negative_char_n5.p')

classifier = bigram.Classifier(('positive', LM_pos), ('negative', LM_neg))
#classifier_char = bigram.Classifier(('positive', LM_pos_char), ('negative', LM_neg_char))


#print(classifier.classify(sentence))
print(classifier.evaluate('corpora/merged_sentiment_data.csv', mixture=[2]))
#print(classifier.evaluate('corpora/merged_sentiment_data.csv', mixture=[5]))
