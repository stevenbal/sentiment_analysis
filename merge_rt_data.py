import csv
import pandas as pd
import os

def merge_dev_data(result_filename, file_pos, file_neg):
    """
    Description:    function that merges test data from both sentiments into
                    a single data structure
    Input:
    -file_pos:      str, name of file containing positive test data
    -file_neg:      str, name of file containing negative test data

    Output:
    -merged_data:   list, contains tuples each of which contains a sentence and
                    its sentiment
    """
    merged_data = []
    with open(file_pos, errors='replace') as text:
        txt = text.readlines()
        merged_data += [(line, 'positive') for line in txt]
        text.close()
    with open(file_neg, errors='replace') as text:
        txt = text.readlines()
        merged_data += [(line, 'negative') for line in txt]
        text.close()
    df = pd.DataFrame(merged_data, columns=['text', 'sentiment'])
    df['text'] = df['text'].apply(lambda x: x.strip())
    df.to_csv(result_filename, index=False)

def merge_training_data(result_filename_pos, result_filename_neg, dir_pos, dir_neg):
    df_pos = pd.DataFrame()
    for filename in os.listdir(dir_pos):
        with open(dir_pos+'/'+filename, errors='replace') as text:
            txt = text.readlines()
            data = pd.DataFrame(list(zip(txt, ['positive']*len(txt))))
            df_pos = df_pos.append(data)
            text.close()
    df_neg = pd.DataFrame()
    for filename in os.listdir(dir_neg):
        with open(dir_neg+'/'+filename, errors='replace') as text:
            txt = text.readlines()
            data = pd.DataFrame(list(zip(txt, ['negative']*len(txt))))
            df_neg = df_neg.append(data)
            text.close()
    df_pos.columns = ['text', 'sentiment']
    df_neg.columns = ['text', 'sentiment']
    df_pos['text'] = df_pos['text'].apply(lambda x: x.strip())
    df_neg['text'] = df_neg['text'].apply(lambda x: x.strip())
    df_pos.to_csv(result_filename_pos, index=False)
    df_neg.to_csv(result_filename_neg, index=False)

def store_data(filename, data):
    """
    Description:    function that stores merged data into a csv file

    Input:
    -filename:      str, name of the resulting csv file
    -data:          list, the data structure to be stored
    """
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        [csvwriter.writerow(line) for line in data]

result_dir = 'corpora/processed/rottentomatoes/'

training_positive_path = 'corpora/raw/review_polarity/txt_sentoken/pos'
training_negative_path = 'corpora/raw/review_polarity/txt_sentoken/neg'
merge_training_data(result_dir+'positive_training.csv', result_dir+'negative_training.csv', training_positive_path, training_negative_path)

testing_positive_path = 'corpora/raw/rt-polaritydata/rt-polaritydata/rt-polarity.pos'
testing_negative_path = 'corpora/raw/rt-polaritydata/rt-polaritydata/rt-polarity.neg'
merge_dev_data(result_dir+'development_data.csv', testing_positive_path, testing_negative_path)