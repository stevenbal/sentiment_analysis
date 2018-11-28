# sentiment-analysis

Simple Naive Bayes classifier that uses n-gram models to try and predict whether a given sentence is of positive or negative sentiment.

Written in Python version 3.6.2, dependencies are listed in 'environment.yml'.
Using an environment manager such as conda, the correct packages can be easily installed
by using the following bash command:
```console
user@user:~$ conda env create -f environment.yml
```
The environment can be activated using:
```console
user@user:~$ source activate sentiment-analysis
```

The environment can be deactivated using:
```console
user@user:~$ source deactivate
```

## Directory descriptions
### corpora/processed/
Contains different corpora of which the format has been standardized as follows:
- Two .csv files for training data: one for positive and one for negative examples
- One .csv file for development data, used to evaluate the classifier
- Optional if the corpus is large enough: one .csv file for testing data

### corpora/raw/
Contains the raw corpus data

### models/
Contains n-gram models that were created with specific settings and saved, used to
save time when classifying by loading instead of recreating certain models

### resources/
Contains the class files that are used in the scripts

### results/
Contains preliminary results for each different training corpus

## Basic command line functionality
To classify a given sentence, the following bash command can be used:
```console
user@user:~$ python classify_sentence.py "..."
```
Where ... denotes the sentence to be classified

## Basic classifier instantiation and sentence classification in Python
To classify a sentence in a Python script:
```python
import resources.LanguageModel as ngram
import resources.NaiveBayesClassifier as NBclassifier

# Modify the model_file argument to select another model from models/
LM_pos = ngram.LanguageModel(model_file='models/positive_n2_stemmed_rottentomatoes.p')
LM_neg = ngram.LanguageModel(model_file='models/negative_n2_stemmed_rottentomatoes.p')

# Construct classifier from the two models
classifier = NBclassifier.NaiveBayesClassifier(('positive', LM_pos), ('negative', LM_neg))

sentence = '...'
classifier.classify(sentence)
```
