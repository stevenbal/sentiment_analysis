import csv

def merge_sentiment_data(file_pos, file_neg):
    merged_data = []
    with open(file_pos, errors='replace') as text:
        txt = text.readlines()
        merged_data += [(line, 'positive') for line in txt]
        text.close()
    with open(file_neg, errors='replace') as text:
        txt = text.readlines()
        merged_data += [(line, 'negative') for line in txt]
        text.close()
    return merged_data

def store_data(filename, data):
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        [csvwriter.writerow(line) for line in data]

data = merge_sentiment_data('corpora/rt-polaritydata/rt-polaritydata/rt-polarity.pos', 'corpora/rt-polaritydata/rt-polaritydata/rt-polarity.neg')
store_data('corpora/merged_sentiment_data.csv', data)
