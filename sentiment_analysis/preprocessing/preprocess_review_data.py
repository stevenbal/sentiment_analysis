import pandas as pd
import os
from ..settings import BASE_DIR

path = os.path.join(BASE_DIR, 'corpora/raw/reviews/')
res_path = os.path.join(BASE_DIR, 'corpora/processed/')
filenames = [
    'amazon_cells_labelled.txt',
    'imdb_labelled.txt',
    'yelp_labelled.txt'
]
res_dirs = ['amazon/', 'imdb/', 'yelp/']

for filename, res_dir in zip(filenames, res_dirs):
    data = pd.read_csv(path + filename, encoding='ISO-8859-1', delimiter='\t')

    data.columns = ['text', 'sentiment']
    data.loc[data['sentiment'] == 1, 'sentiment'] = 'positive'
    data.loc[data['sentiment'] == 0, 'sentiment'] = 'negative'
    print(data.head())
    # data.to_csv()
