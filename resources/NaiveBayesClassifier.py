import collections
import csv

import resources.visualize as visualize
from resources.LanguageModel import LanguageModel

class NaiveBayesClassifier:
    def __init__(self, *models):
        """
        Description:    constructor for a Classifier object, given one or more
                        LanguageModel objects

        Input:
        -*models:       tuple, any number of tuples containing the class the
                        LanguageModel object represents, and the LanguageModel object
        """
        self.models = {model[0]: model[1] for model in models}

    def classify(self, sentence, mixture=[]):
        """
        Description:        function that returns the most probable class for a given
                            sentence

        Input:
        -sentence:          str, the sentence to be classified
        -mixture:           list, contains integers specifying the orders of models
                            to be used as a mixture model, the class is selected by
                            majority vote in this case (default: [])

        Output:
        -predicted_class:   str, the class as predicted by the classifier
        """
        if mixture:
            if max(mixture) > len(self.models['positive'].get_models()):
                raise ModelOrderError(max(mixture))
            results = []
            for order in mixture:
                probs = {category: model.compute_prob(sentence, N=order) for category, model in self.models.items()}
                results.append(max(probs, key=probs.get))
            predicted_class = collections.Counter(results).most_common(1)[0][0]
            return predicted_class
        else:
            probs = {category: model.compute_prob(sentence) for category, model in self.models.items()}
            predicted_class = max(probs, key=probs.get)
            return predicted_class

    def evaluate(self, filename, mixture=[]):
        """
        Description:    function that evaluates the performance of the classifier
                        for a given test corpus

        Input:
        -filename:      str, the name of the csv file containing the test corpus
        -mixture:       list, specifies the mixture model (default: [])

        Output:
        -results:       dict, contains the counts of true and false positives
                        and true and false negatives
        """
        results = {'positive': {'true': 0, 'false': 0}, 'negative': {'true': 0, 'false': 0}}
        i = 0
        correct_labels = []
        predicted_labels = []
        with open(filename, errors='replace') as csvfile:
            datareader = csv.reader(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for line, sentiment in datareader:
                correct_labels.append(sentiment)
                prediction = self.classify(line, mixture=mixture)
                predicted_labels.append(prediction)
                if prediction == sentiment:
                    results[sentiment]['true'] += 1
                else:
                    results[prediction]['false'] += 1
            csvfile.close()
        print('precision {}, recall {}'.format(self.compute_precision(results), self.compute_recall(results)))
        visualize.plot_confusion_matrix(correct_labels, predicted_labels, ['Negative', 'Positive'])
        return self.compute_precision(results), self.compute_recall(results), results

    def compute_precision(self, results):
        """
        Description:    function that computes precision given the classification results

        Input:
        -results:       dict, contains the counts of true and false positives
                        and true and false negatives

        Output:
        -precision:     float, precision as produced by classifier
        """
        if (results['positive']['true'] + results['positive']['false']):
            precision = results['positive']['true'] / (results['positive']['true'] + results['positive']['false'])
        else:
            precision = 0
        return precision

    def compute_recall(self, results):
        """
        Description:    function that computes recall given the classification results

        Input:
        -results:       dict, contains the counts of true and false positives
                        and true and false negatives

        Output:
        -recall:        float, recall as produced by classifier
        """
        if (results['positive']['true'] + results['positive']['false']):
            recall = results['positive']['true'] / (results['positive']['true'] + results['negative']['false'])
        else:
            recall = 0
        return recall

class ModelOrderError(ValueError):
    """Indicates that an invaled order for a mixture model was given"""
    pass
