import sys
from time import time
import json
import argparse

import ngram_model as ngram
import NaiveBayesClassifier as NBclassifier

parser = argparse.ArgumentParser()

parser.add_argument('--N', help='Order of n-gram model')
parser.add_argument('--words', help='Treat words as smallest units')
parser.add_argument('--stemming', help='Apply stemming')
parser.add_argument('--stopword_removal', help='Remove stopwords')
parser.add_argument('--mixture', nargs = '*', help='Specify orders of mixture model')
parser.add_argument('--method', help='Specify how to obtain models')

args = parser.parse_args()

if len(args.__dict__) > 4:
    N = int(args.N)
    words = args.words == 'True'
    stemming = args.stemming  == 'True'
    stopword_removal = args.stopword_removal  == 'True'
    method = args.method
    if args.mixture:
        mixture = [int(i) for i in args.mixture]
    else:
        mixture = []
else:
    N = 3
    words = True
    stemming = True
    stopword_removal = True
    method = 'load'
    mixture = [1, 2]

def create_path_string(sentiment):
    path = f'models/{sentiment}'
    def create_string(N, words, stemming, nostopwords):
        modified_path = path + f'_n{N}'
        if not words:
            modified_path += '_char'
        if nostopwords:
            modified_path += '_nostopwords'
        if stemming:
            modified_path += '_stemmed'
        return modified_path + '.p'
    return create_string

positive_path = create_path_string('positive')
negative_path = create_path_string('negative')

if method == 'create':
    LM_pos = ngram.LanguageModel('corpora/review_polarity/txt_sentoken/pos', N=N, words=words, stemming=stemming, stopword_removal=stopword_removal)
    LM_neg = ngram.LanguageModel('corpora/review_polarity/txt_sentoken/neg', N=N, words=words, stemming=stemming, stopword_removal=stopword_removal)
    LM_pos.save_models(positive_path(N, words, stemming, stopword_removal))
    LM_neg.save_models(negative_path(N, words, stemming, stopword_removal))

# Load the ngram models for negative and positive sentiments, n=3 and stopwords
# are omitted
if method == 'load':
    LM_pos = ngram.LanguageModel(model_file=positive_path(N, words, stemming, stopword_removal))
    LM_neg = ngram.LanguageModel(model_file=negative_path(N, words, stemming, stopword_removal))

# Construct classifier from the two models
classifier = NBclassifier.NaiveBayesClassifier(('positive', LM_pos), ('negative', LM_neg))

# Evaluate the classifier on the test set and report precision and recall,
# use mixture model of bigram and unigram model
res = classifier.evaluate('corpora/merged_sentiment_data.csv', mixture=mixture)

with open('results/NaiveBayesClassifier_results.txt', 'a') as result_file:
    result_file.write(f'NaiveBayesClassifier with N={N} words={words} stemming={stemming} stopword_removal={stopword_removal}\n')
    result_file.write(f'Mixture model of orders: {mixture}\n')
    result_file.write(f'Precision: {res[0]}\n')
    result_file.write(f'Recall: {res[1]}\n')
    result_file.write(f'Counts: {res[2]}\n\n')
