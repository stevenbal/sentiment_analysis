# sentiment-analysis

Simple Naive Bayes classifier that uses n-gram models to try and predict whether a given sentence is of positive or negative sentiment.

Written in Python version 3.6.2

## Directory descriptions
### corpora/
Contains corpora with documents labeled with sentiment 'positive' or 'negative'

### deprecated/
Contains older versions of certain .py files

### models/
Contains n-gram models that were created with specific settings and saved, used to speed up the classification task

### resources/
Contains the class files that are used in the scripts

### results/
Contains preliminary results for the Naive Bayes classifier are shown in results/NaiveBayesClassifier_results.txt


## Basic functionality
To classify a given sentence, use the following terminal command:
```console
user@user:~$ python3 classify_sentence.py "..."
```
Where ... denotes the sentence to be classified
