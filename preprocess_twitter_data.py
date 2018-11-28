import pandas as pd

data = pd.read_csv('corpora/raw/twitter/preprocessed_twitter.csv', encoding="ISO-8859-1")

training_perc = 0.6 
dev_perc = 0.3
testing_perc = 0.1

pos_data = data[data['target'] == 'positive'][['text', 'target']]
neg_data = data[data['target'] == 'negative'][['text', 'target']]

pos_data.columns = ['text', 'sentiment']
neg_data.columns = ['text', 'sentiment']

pos_data_shuffled = pos_data.sample(frac=1).reset_index(drop=True)
len_pos = len(pos_data_shuffled)

pos_training = pos_data_shuffled.loc[:int(len_pos * training_perc)]
pos_dev = pos_data_shuffled.loc[int(len_pos * training_perc):int(len_pos * (training_perc + dev_perc))]
pos_testing = pos_data_shuffled.loc[int(len_pos * (training_perc + dev_perc)):]

neg_data_shuffled = neg_data.sample(frac=1).reset_index(drop=True)
len_neg = len(neg_data_shuffled)

neg_training = neg_data_shuffled.loc[:int(len_neg * training_perc)]
neg_dev = neg_data_shuffled.loc[int(len_neg * training_perc):int(len_neg * (training_perc + dev_perc))]
neg_testing = neg_data_shuffled.loc[int(len_neg * (training_perc + dev_perc)):]

dev_data = pd.concat([pos_dev, neg_dev])
testing_data = pd.concat([pos_testing, neg_testing])

result_dir = 'corpora/processed/twitter/'

pos_training.to_csv(f'{result_dir}positive_training.csv', index=False)
neg_training.to_csv(f'{result_dir}negative_training.csv', index=False)
dev_data.to_csv(f'{result_dir}development_data.csv', index=False)
testing_data.to_csv(f'{result_dir}testing_data.csv', index=False)