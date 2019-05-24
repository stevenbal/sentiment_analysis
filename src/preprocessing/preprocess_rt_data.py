import csv
import pandas as pd
import os
from settings import BASE_DIR
import numpy as np
from resources.utilities import preprocess_sentence


def merge_dev_data(result_filename, file_pos, file_neg):
    """
    Description:        function that merges dev data from both
                        sentiments into a single data structure
    Input:
    -result_filename:   str, name of the file to write the result to
    -file_pos:          str, name of file containing positive dev data
    -file_neg:          str, name of file containing negative dev data
    """
    merged_data = []
    with open(file_pos, errors='replace') as text:
        txt = text.readlines()
        merged_data += [(preprocess_sentence(line), 'positive')
                        for line in txt]
        text.close()
    with open(file_neg, errors='replace') as text:
        txt = text.readlines()
        merged_data += [(preprocess_sentence(line), 'negative')
                        for line in txt]
        text.close()
    df = pd.DataFrame(merged_data, columns=['text', 'sentiment'])
    df['text'] = df['text'].apply(lambda x: x.strip())
    df = df.replace('', np.nan)
    df = df[df['text'].notnull()]
    df.to_csv(result_filename, index=False)


def merge_training_data(result_filename, original_dir, sentiment):
    """
    Description:            function that merges the training text files
                            for the positive and negative directories

    Input:
    -result_filename_pos:   str, name of the file that will contain
                            training data for the given sentiment
    -original_dir:          str, the directory containing the text files
    -sentiment:             str, the sentiment of the given text files
    """
    df = pd.DataFrame()
    for filename in os.listdir(original_dir):
        with open(f'{original_dir}/{filename}', errors='replace') as text:
            txt = text.readlines()
            preprocessed_lines = [preprocess_sentence(line) for line in txt]
            data = pd.DataFrame(list(zip(preprocessed_lines,
                                         [sentiment] * len(txt))))
            df = df.append(data)
            text.close()
    df.columns = ['text', 'sentiment']
    df['text'] = df['text'].apply(lambda x: x.strip())
    df = df.replace('', np.nan)
    df = df.dropna()
    df.to_csv(result_filename, index=False)


# Specify result directory
result_dir = os.path.join(BASE_DIR, 'corpora/processed/rottentomatoes/')

# Specify positive and negative training paths
training_positive_path = os.path.join(BASE_DIR, 'corpora/raw/review_polarity/txt_sentoken/pos')
training_negative_path = os.path.join(BASE_DIR, 'corpora/raw/review_polarity/txt_sentoken/neg')

# Merge the training text files for positive and negative
merge_training_data(f'{result_dir}positive_training.csv',
                    training_positive_path, 'positive')
merge_training_data(f'{result_dir}negative_training.csv',
                    training_negative_path, 'negative')

# Merge the development data and save it to a .csv file
dev_positive_path = os.path.join(BASE_DIR, 'corpora/raw/rt-polaritydata/rt-polaritydata/rt-polarity.pos')
dev_negative_path = os.path.join(BASE_DIR, 'corpora/raw/rt-polaritydata/rt-polaritydata/rt-polarity.neg')
merge_dev_data(f'{result_dir}development_data.csv', dev_positive_path,
               dev_negative_path)
