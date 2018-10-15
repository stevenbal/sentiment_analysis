import sys
import ngram_model_old as ngram
from time import time

t1 = time()
LM_pos = ngram.LanguageModel('corpora/review_polarity/txt_sentoken/pos', N=3, words=True)
LM_neg = ngram.LanguageModel('corpora/review_polarity/txt_sentoken/neg', N=3, words=True)
print(time()-t1)
#
# LM_pos.save_models('models/positive_char_n6.p')
# LM_neg.save_models('models/negative_char_n6.p')

# Load the ngram models for negative and positive sentiments, n=3 and stopwords
# are omitted
# LM_pos = ngram.LanguageModel(model_file='models/positive_n3_nostopwords.p')
# LM_neg = ngram.LanguageModel(model_file='models/negative_n3_nostopwords.p')

#LM_pos_char = ngram.LanguageModel(model_file='models/positive_char_n6_nostopwords.p')
#LM_neg_char = ngram.LanguageModel(model_file='models/negative_char_n6_nostopwords.p')

# Construct classifier from the two models
classifier = ngram.Classifier(('positive', LM_pos), ('negative', LM_neg))
#classifier_char = ngram.Classifier(('positive', LM_pos_char), ('negative', LM_neg_char))



# Evaluate the classifier on the test set and report precision and recall,
# use mixture model of bigram and unigram model
t1 = time()
print(classifier.evaluate('corpora/merged_sentiment_data.csv', mixture=[1,2]))
print(time()-t1)
