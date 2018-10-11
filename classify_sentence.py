import sys
import ngram_model as ngram
from time import time

if len(sys.argv) > 1:
    sentence = sys.argv[1]
else:
    sentence = input('Please enter a sentence: ')

# LM_pos = ngram.LanguageModel('corpora/review_polarity/txt_sentoken/pos', N=6, words=False)
# LM_neg = ngram.LanguageModel('corpora/review_polarity/txt_sentoken/neg', N=6, words=False)
#
# LM_pos.save_models('models/positive_char_n6.p')
# LM_neg.save_models('models/negative_char_n6.p')

'''
Load the ngram models for negative and positive sentiments, n=3 and stopwords
are omitted
'''
LM_pos = ngram.LanguageModel(model_file='models/positive_n3_nostopwords.p')
LM_neg = ngram.LanguageModel(model_file='models/negative_n3_nostopwords.p')

#LM_pos_char = ngram.LanguageModel(model_file='models/positive_char_n6_nostopwords.p')
#LM_neg_char = ngram.LanguageModel(model_file='models/negative_char_n6_nostopwords.p')

'''
Construct classifier from the two models
'''
classifier = ngram.Classifier(('positive', LM_pos), ('negative', LM_neg))
#classifier_char = ngram.Classifier(('positive', LM_pos_char), ('negative', LM_neg_char))

'''
Determine the sentiment of the given sentence, only use bigram model
'''
print(classifier.classify(sentence, mixture=[2]))
