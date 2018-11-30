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
dev_corpus = argumentparser.dev_corpus

# Load the paths of the model files that will be created/loaded
positive_path = argumentparser.positive_path
negative_path = argumentparser.negative_path

# Create the n-gram models with the given settings and save them to the correct paths
if method == 'create':
    t1 = time()
    LM_pos = ngram.LanguageModel(f'{corpora_dir}{training_corpus}/positive_training.csv', N=N, 
                                 words=words, stemming=stemming, 
                                 stopword_removal=stopword_removal)
    t2 = time()
    LM_neg = ngram.LanguageModel(f'{corpora_dir}{training_corpus}/negative_training.csv', N=N, 
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

# Evaluate the classifier on the test set and report precision, recall and counts,
# use a mixture model if specified
t4 = time()
precision, recall, counts = classifier.evaluate(f'{corpora_dir}{dev_corpus}/development_data.csv', mixture=mixture, prediction_thres=0)
t5 = time()

# precision, recall, counts = 5.0,3.0,"{'positive': {'true': 3781, 'false': 1608}, 'negative': {'true': 3723, 'false': 1550}}"

print('Evaluation took', t5-t4, 'seconds')

def save_results(result_filename, training_corpus, **kwargs):
    """
    Description:        function that writes the results of the classifier to
                        a properly named .csv file

    Input:
    -result_filename:   str, name of the file to which the classification results
                        will be written
    -training_corpus:   str, name of the corpus used to train the models with
    -**kwargs:          keyword args, specifies the other settings that are
                        relevant, such as order of N-gram model, whether stemming
                        is used, etc.
    """
    res_path = f'results/{training_corpus}/{result_filename}'
    kwargs['mixture'] = str(kwargs['mixture'])
    colnames = list(kwargs.keys())
    arg_values = arg_values = list(kwargs.values())
    if result_filename in os.listdir(f'results/{training_corpus}'):
        results = pd.read_csv(res_path)
        masked_res = results
        for argument, value in kwargs.items():
            if argument in ['precision', 'recall', 'counts']:
                continue
            masked_res = masked_res[masked_res[argument] == value]
        if not masked_res.empty:
            masked_res = [arg_values]
        else:
            results = results.append(pd.DataFrame([arg_values], columns=colnames))
        results.to_csv(res_path, index=False)
    else:
        results = pd.DataFrame([arg_values], columns=colnames)
        results.to_csv(res_path, index=False)

save_results('NaiveBayesClassifier_results.csv', training_corpus, N=N, stemming=stemming,
              stopword_removal=stopword_removal, mixture=mixture, dev_corpus=dev_corpus,
              precision=precision, recall=recall, counts=counts)
