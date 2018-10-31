import csv

def merge_sentiment_data(file_pos, file_neg):
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
    return merged_data

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

corpus_positive_path = 'corpora/rt-polaritydata/rt-polaritydata/rt-polarity.pos'
corpus_negative_path = 'corpora/rt-polaritydata/rt-polaritydata/rt-polarity.neg'
data = merge_sentiment_data(corpus_positive_path, corpus_negative_path)

store_data('corpora/merged_sentiment_data.csv', data)
