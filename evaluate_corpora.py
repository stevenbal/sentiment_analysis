from time import time
import argumentparser
import pandas as pd
import os

import resources.LanguageModel as ngram
import resources.NaiveBayesClassifier as NBclassifier

corpora_dir = 'corpora/processed/'

# Load the settings to be used for the model instantiation
N = argumentparser.N
words = argumentparser.words
stemming = argumentparser.stemming
stopword_removal = argumentparser.stopword_removal
method = argumentparser.method
mixture = argumentparser.mixture
training_corpus = argumentparser.training_corpus
testing_corpus = argumentparser.testing_corpus

# Load the paths of the model files that will be created/loaded
positive_path = argumentparser.positive_path
negative_path = argumentparser.negative_path

# Create the n-gram models with the given settings and save them to the correct paths
if method == 'create':
    t1 = time()
    LM_pos = ngram.LanguageModel(corpora_dir+training_corpus+'/positive_training.csv', N=N, 
                                 words=words, stemming=stemming, 
                                 stopword_removal=stopword_removal)
    t2 = time()
    LM_neg = ngram.LanguageModel(corpora_dir+training_corpus+'/negative_training.csv', N=N, 
                                 words=words, stemming=stemming, 
                                 stopword_removal=stopword_removal)
    t3 = time()
    LM_pos.save_models(positive_path(N, words, stemming, stopword_removal, training_corpus))
    LM_neg.save_models(negative_path(N, words, stemming, stopword_removal, training_corpus))
    print('Model construction took', t3-t1, 'seconds')

# Load the n-gram models with the given settings
if method == 'load':
    try:
        LM_pos = ngram.LanguageModel(model_file=positive_path(N, words, stemming, stopword_removal, training_corpus))
        LM_neg = ngram.LanguageModel(model_file=negative_path(N, words, stemming, stopword_removal, training_corpus))
    except:
        print('Models with the given settings cannot be loaded, please choose --method=create')
        exit()

# Construct classifier from the two n-gram models
classifier = NBclassifier.NaiveBayesClassifier(('positive', LM_pos), ('negative', LM_neg))

# Evaluate the classifier on the test set and report precision and recall,
# use a mixture model if specified
t4 = time()
precision, recall, counts = classifier.evaluate(corpora_dir+testing_corpus+'/development_data.csv', mixture=mixture, prediction_thres=0)
t5 = time()

print('Evaluation took', t5-t4, 'seconds')

def save_results(result_filename, training_corpus, **kwargs):
    if result_filename in os.listdir('results/'+training_corpus):
        results = pd.read_csv('results/'+training_corpus+'/'+result_filename)
        mask = True
        for argument, value in kwargs.items():
            mask &= (result['argument'] == value)
        if not results[mask].empty:
            results.loc[mask] = [list(kwargs.values())]
        else:
            results = results.append(pd.DataFrame([list(kwargs.values())]))
        results.to_csv('results/'+training_corpus+result_filename, index=False)
    else:
        results = pd.DataFrame([list(kwargs.values())])
        results.columns = list(kwargs.keys())
        results.to_csv('results/'+training_corpus+'/'+result_filename, index=False)

save_results('NaiveBayesClassifier_results.csv', training_corpus, N=N, stemming=stemming,
              stopword_removal=stopword_removal, mixture=mixture, testing_corpus=testing_corpus,
              precision=precision, recall=recall, counts=counts)
