import sys
from time import time

import os

from ngram import LanguageModel, NaiveBayesClassifier

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

# Load the ngram models for negative and positive sentiments, n=2 with stemming
model_file_pos = os.path.join(BASE_DIR, "models/positive_n2_stemmed_rottentomatoes.p")
model_file_neg = os.path.join(BASE_DIR, "models/negative_n2_stemmed_rottentomatoes.p")

LM_pos = LanguageModel("positive", model_file=model_file_pos)
LM_neg = LanguageModel("negative", model_file=model_file_neg)

# Construct classifier from the two models
classifier = NaiveBayesClassifier(LM_pos, LM_neg)

# Determine the sentiment of the given sentence, only use bigram model

if __name__ == "__main__":
    if len(sys.argv) > 1:
        sentence = sys.argv[1]
    else:
        sentence = input("Please enter a sentence: ")

    print(classifier.classify(sentence, prediction_thres=0.02))
