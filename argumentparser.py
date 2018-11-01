import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--N', help='Order of n-gram model')
parser.add_argument('--words', help='Treat words as smallest units')
parser.add_argument('--stemming', help='Apply stemming')
parser.add_argument('--stopword_removal', help='Remove stopwords')
parser.add_argument('--mixture', nargs = '*', help='Specify orders of mixture model')
parser.add_argument('--method', help='Specify how to obtain models')

args = parser.parse_args()

if len(args.__dict__) > 4:
    N = int(args.N)
    words = args.words == 'True'
    stemming = args.stemming  == 'True'
    stopword_removal = args.stopword_removal  == 'True'
    method = args.method
    if args.mixture:
        mixture = [int(i) for i in args.mixture]
    else:
        mixture = []
else:
    N = 3
    words = True
    stemming = True
    stopword_removal = True
    method = 'load'
    mixture = [1, 2]

def create_path_string(sentiment):
    path = f'models/{sentiment}'
    def create_string(N, words, stemming, nostopwords):
        modified_path = path + f'_n{N}'
        if not words:
            modified_path += '_char'
        if nostopwords:
            modified_path += '_nostopwords'
        if stemming:
            modified_path += '_stemmed'
        return modified_path + '.p'
    return create_string

positive_path = create_path_string('positive')
negative_path = create_path_string('negative')