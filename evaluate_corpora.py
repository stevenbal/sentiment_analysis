import sys
from time import time

import ngram_model as ngram
import NaiveBayesClassifier as NBclassifier

#TODO word vector embeddings

LM_pos = ngram.LanguageModel('corpora/review_polarity/txt_sentoken/pos', N=3, words=True)
LM_neg = ngram.LanguageModel('corpora/review_polarity/txt_sentoken/neg', N=3, words=True)

# LM_pos.save_models('models/positive_char_n6.p')
# LM_neg.save_models('models/negative_char_n6.p')

# Load the ngram models for negative and positive sentiments, n=3 and stopwords
# are omitted
LM_pos = ngram.LanguageModel(model_file='models/positive_n3.p')
LM_neg = ngram.LanguageModel(model_file='models/negative_n3.p')

#LM_pos_char = ngram.LanguageModel(model_file='models/positive_char_n6_nostopwords.p')
#LM_neg_char = ngram.LanguageModel(model_file='models/negative_char_n6_nostopwords.p')

# Construct classifier from the two models
classifier = NBclassifier.NaiveBayesClassifier(('positive', LM_pos), ('negative', LM_neg))
#classifier_char = ngram.Classifier(('positive', LM_pos_char), ('negative', LM_neg_char))



# Evaluate the classifier on the test set and report precision and recall,
# use mixture model of bigram and unigram model
print(classifier.evaluate('corpora/merged_sentiment_data.csv', mixture=[1,2]))
