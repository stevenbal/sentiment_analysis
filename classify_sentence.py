import sys
from time import time

import resources.LanguageModel as ngram
import resources.NaiveBayesClassifier as NBclassifier

# Load the ngram models for negative and positive sentiments, n=3 and stopwords
# are omitted

LM_pos = ngram.LanguageModel(model_file='models/positive_n2_stemmed_rottentomatoes.p')
LM_neg = ngram.LanguageModel(model_file='models/negative_n2_stemmed_rottentomatoes.p')

# LM_pos = ngram.LanguageModel(model_file='models/test_pos1.p')
# LM_neg = ngram.LanguageModel(model_file='models/test_neg1.p')

# LM_pos = ngram.LanguageModel('corpora/merged_training_data_pos.csv', N=1, words=True, stemming=True, stopword_removal=True)
# LM_neg = ngram.LanguageModel('corpora/merged_training_data_neg.csv', N=1, words=True, stemming=True, stopword_removal=True)
# LM_pos.save_models('models/test_pos1.p')
# LM_neg.save_models('models/test_neg1.p')

# Construct classifier from the two models
classifier = NBclassifier.NaiveBayesClassifier(('positive', LM_pos), ('negative', LM_neg))

# Determine the sentiment of the given sentence, only use bigram model

if __name__ == '__main__':
    if len(sys.argv) > 1:
        sentence = sys.argv[1]
    else:
        sentence = input('Please enter a sentence: ')

    print(classifier.classify(sentence, mixture=[], prediction_thres=0.02))
