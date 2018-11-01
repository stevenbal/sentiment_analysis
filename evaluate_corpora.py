import sys
from time import time
import json
import argumentparser

import resources.LanguageModel as ngram
import resources.NaiveBayesClassifier as NBclassifier

# Load the settings to be used for the model instantiation
N = argumentparser.N
words = argumentparser.words
stemming = argumentparser.stemming
stopword_removal = argumentparser.stopword_removal
method = argumentparser.method
mixture = argumentparser.mixture

# Load the paths of the model files that will be created/loaded
positive_path = argumentparser.positive_path
negative_path = argumentparser.negative_path

# Create the n-gram models with the given settings and save them to the correct paths
if method == 'create':
    LM_pos = ngram.LanguageModel('corpora/review_polarity/txt_sentoken/pos', N=N, words=words, stemming=stemming, stopword_removal=stopword_removal)
    LM_neg = ngram.LanguageModel('corpora/review_polarity/txt_sentoken/neg', N=N, words=words, stemming=stemming, stopword_removal=stopword_removal)
    LM_pos.save_models(positive_path(N, words, stemming, stopword_removal))
    LM_neg.save_models(negative_path(N, words, stemming, stopword_removal))

# Load the n-gram models with the given settings
if method == 'load':
    try:
        LM_pos = ngram.LanguageModel(model_file=positive_path(N, words, stemming, stopword_removal))
        LM_neg = ngram.LanguageModel(model_file=negative_path(N, words, stemming, stopword_removal))
    except:
        print('Models with the given settings cannot be loaded, please choose --method=create')
        exit()

# Construct classifier from the two n-gram models
classifier = NBclassifier.NaiveBayesClassifier(('positive', LM_pos), ('negative', LM_neg))

# Evaluate the classifier on the test set and report precision and recall,
# use a mixture model if specified
res = classifier.evaluate('corpora/merged_sentiment_data.csv', mixture=mixture)

with open('results/NaiveBayesClassifier_results.txt', 'a') as result_file:
    result_file.write(f'NaiveBayesClassifier with N={N} words={words} stemming={stemming} stopword_removal={stopword_removal}\n')
    result_file.write(f'Mixture model of orders: {mixture}\n')
    result_file.write(f'Precision: {res[0]}\n')
    result_file.write(f'Recall: {res[1]}\n')
    result_file.write(f'Counts: {res[2]}\n\n')
