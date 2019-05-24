from time import time
import argumentparser
import pandas as pd
import os
from settings import BASE_DIR

import resources.LanguageModel as ngram
import resources.NaiveBayesClassifier as NBclassifier
import logging

logging.basicConfig(filename='process_times.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

corpora_dir = os.path.join(BASE_DIR, 'corpora/processed/')

# Load the settings to be used for the model instantiation
N = argumentparser.N
words = argumentparser.words
stemming = argumentparser.stemming
stopword_removal = argumentparser.stopword_removal
method = argumentparser.method
mixture = argumentparser.mixture
train_corpus = argumentparser.train_corpus
dev_corpus = argumentparser.dev_corpus

# Load the paths of the model files that will be created/loaded
positive_path = argumentparser.positive_path(N, words, stemming,
                                             stopword_removal, train_corpus)
negative_path = argumentparser.negative_path(N, words, stemming,
                                             stopword_removal, train_corpus)

# Create the n-gram models with the given settings and save them to the 
# correct paths
if method == 'create':
    start = time()
    train_corpus_pos = f'{corpora_dir}{train_corpus}/positive_training.csv'
    train_corpus_neg = f'{corpora_dir}{train_corpus}/negative_training.csv'
    LM_pos = ngram.LanguageModel('positive', train_corpus_pos, N=N,
                                 words=words, stemming=stemming,
                                 stopword_removal=stopword_removal)
    LM_neg = ngram.LanguageModel('negative', train_corpus_neg, N=N,
                                 words=words, stemming=stemming,
                                 stopword_removal=stopword_removal)
    end = time()
    LM_pos.save_models(positive_path)
    LM_neg.save_models(negative_path)
    logging.info(f'Model construction for {train_corpus} took {end-start} \
                   seconds')

# Load the n-gram models with the given settings
if method == 'load':
    try:
        LM_pos = ngram.LanguageModel('positive', model_file=positive_path)
        LM_neg = ngram.LanguageModel('negative', model_file=negative_path)
    except BaseException:
        print('Models with the given settings cannot be loaded, please choose \
               --method=create')
        exit()

# Construct classifier from the two n-gram models
classifier = NBclassifier.NaiveBayesClassifier(LM_pos, LM_neg)

# Evaluate the classifier on the test set and report precision, recall and
# counts, use a mixture model if specified
start = time()
dev_data = f'{corpora_dir}{dev_corpus}/development_data.csv'
precision, recall, counts = classifier.evaluate(dev_data, mixture=mixture,
                                                prediction_thres=0)
end = time()

logging.info(f'Evaluation for {dev_corpus} took {end-start} seconds')


def save_results(result_filename, train_corpus, **kwargs):
    """
    Description:        function that writes the results of the
                        classifier to a properly named .csv file

    Input:
    -result_filename:   str, name of the file to which the
                        classification results will be written
    -train_corpus:   str, name of the corpus used to train the models
                        with
    -**kwargs:          keyword args, specifies the other settings that
                        are relevant, such as order of N-gram model,
                        whether stemming is used, etc.
    """
    res_path = f'results/{train_corpus}/{result_filename}'
    kwargs['mixture'] = str(kwargs['mixture'])
    colnames = list(kwargs.keys())
    arg_values = arg_values = list(kwargs.values())
    if result_filename in os.listdir(f'results/{train_corpus}'):
        results = pd.read_csv(res_path)
        masked_res = results
        for argument, value in kwargs.items():
            if argument in ['precision', 'recall', 'counts']:
                continue
            masked_res = masked_res[masked_res[argument] == value]
        if not masked_res.empty:
            masked_res = [arg_values]
        else:
            results = results.append(
                pd.DataFrame(
                    [arg_values],
                    columns=colnames))
        results.to_csv(res_path, index=False)
    else:
        results = pd.DataFrame([arg_values], columns=colnames)
        results.to_csv(res_path, index=False)


save_results('NaiveBayesClassifier_results.csv', train_corpus, N=N,
             words=words, stemming=stemming, stopword_removal=stopword_removal,
             mixture=mixture, dev_corpus=dev_corpus,
             precision=precision, recall=recall, counts=counts)
