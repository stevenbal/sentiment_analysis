import pandas as pd
from resources.utilities import preprocess_sentence

def apply_preprocessing(data):
    """
    Description:    function that applies preprocessing to the text in the data

    Input:
    -data:          pandas DataFrame, the data to be preprocessed
    Output:
    -data:          pandas DataFrame, the preprocessed data
    """
    for i, row in data.iterrows():
        row['text'] = preprocess_sentence(row['text'])
    return data

# Load the cleaned data from a .csv file
data = pd.read_csv('corpora/raw/twitter/cleaned_twitter.csv', encoding="ISO-8859-1")

# Set the data split percentages
training_perc = 0.6 
dev_perc = 0.3
testing_perc = 0.1

# Assign the data for the positive and negative sentiments to proper dataframes
pos_data = data[data['target'] == 'positive'][['text', 'target']]
neg_data = data[data['target'] == 'negative'][['text', 'target']]

# Modify column names to conform to standard format
pos_data.columns = ['text', 'sentiment']
neg_data.columns = ['text', 'sentiment']

# Shuffle the positive data and split according to percentages
pos_data_shuffled = pos_data.sample(frac=1).reset_index(drop=True)
len_pos = len(pos_data_shuffled)

pos_training = pos_data_shuffled.loc[:int(len_pos * training_perc)]
pos_dev = pos_data_shuffled.loc[int(len_pos * training_perc):int(len_pos * (training_perc + dev_perc))]
pos_testing = pos_data_shuffled.loc[int(len_pos * (training_perc + dev_perc)):]

# Shuffle the negative data and split according to percentages
neg_data_shuffled = neg_data.sample(frac=1).reset_index(drop=True)
len_neg = len(neg_data_shuffled)

neg_training = neg_data_shuffled.loc[:int(len_neg * training_perc)]
neg_dev = neg_data_shuffled.loc[int(len_neg * training_perc):int(len_neg * (training_perc + dev_perc))]
neg_testing = neg_data_shuffled.loc[int(len_neg * (training_perc + dev_perc)):]

# Construct the development and testing data from their positive and negative
# components
dev_data = pd.concat([pos_dev, neg_dev])
testing_data = pd.concat([pos_testing, neg_testing])

# Apply preprocessing to each DataFrame
pos_training = apply_preprocessing(pos_training)
neg_training = apply_preprocessing(neg_training)
dev_data = apply_preprocessing(dev_data)
testing_data = apply_preprocessing(testing_data)

# Specify the directory of the resulting .csv files
result_dir = 'corpora/processed/twitter/'

# Write each DataFrame to the appropriate file
pos_training.to_csv(f'{result_dir}positive_training.csv', index=False)
neg_training.to_csv(f'{result_dir}negative_training.csv', index=False)
dev_data.to_csv(f'{result_dir}development_data.csv', index=False)
testing_data.to_csv(f'{result_dir}testing_data.csv', index=False)