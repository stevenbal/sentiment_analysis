import argparse
import os

models_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models/")

parser = argparse.ArgumentParser()

parser.add_argument("--N", help="Order of n-gram model")
parser.add_argument("--words", help="Treat words as smallest units")
parser.add_argument("--stemming", help="Apply stemming")
parser.add_argument("--stopword_removal", help="Remove stopwords")
parser.add_argument(
    "--mixture",
    nargs="*",
    help="Specify orders of mixture model",
)
parser.add_argument("--method", help="Specify how to obtain models")
parser.add_argument(
    "--train_corpus",
    help="Specify corpus to train the model with",
)
parser.add_argument(
    "--dev_corpus",
    help="Specify corpus to evaluate the model with (development data)",
)

arguments = parser.parse_args()

N = int(arguments.N) if arguments.N else 2
words = arguments.words == "True" or True
stemming = arguments.stemming == "True" or True
stopword_removal = arguments.stopword_removal == "True" or False
method = arguments.method or "load"
train_corpus = arguments.train_corpus or "rottentomatoes"
dev_corpus = arguments.dev_corpus or "rottentomatoes"
if arguments.mixture:
    mixture = [int(i) for i in arguments.mixture]
else:
    mixture = []


def create_path_string(sentiment):
    path = f"{models_dir}/{sentiment}"

    def create_string(N, words, stemming, nostopwords, train_corpus):
        modified_path = path + f"_n{N}"
        if not words:
            modified_path += "_char"
        if nostopwords:
            modified_path += "_nostopwords"
        if stemming:
            modified_path += "_stemmed"
        modified_path += "_" + train_corpus
        return modified_path + ".p"

    return create_string


positive_path = create_path_string("positive")
negative_path = create_path_string("negative")
