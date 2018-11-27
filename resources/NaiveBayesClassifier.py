import collections
import csv

import resources.visualize as visualize
from resources.LanguageModel import LanguageModel
import numpy as np
import pandas as pd

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

    def classify(self, sentence, mixture=[], prediction_thres=0):
        """
        Description:        function that returns the most probable class for a given
                            sentence

        Input:
        -sentence:          str, the sentence to be classified
        -mixture:           list, contains integers specifying the orders of models
                            to be used as a mixture model, the class is selected by
                            majority vote in this case (default: [])
        -prediction_thres:  float, specifies the minimum required difference divided
                            by the average log probability as produced by the n-gram
                            models to produce a prediction, else the classification
                            returns 'undefined'

        Output:
        -predicted_class:   str, the class as predicted by the classifier
        """
        if mixture:
            if max(mixture) > len(self.models['positive'].get_models()):
                raise ModelOrderError(max(mixture))
            results = []
            for order in mixture:
                probs = {category: model.compute_prob(sentence, N=order) for category, model in self.models.items()}
                sorted_probs = sorted(probs, key=probs.get, reverse=True)
                most_prob, second_most_prob = sorted_probs[0:2]
                difference = abs(probs[most_prob] - probs[second_most_prob])
                # print(probs)
                # print(sorted_probs, difference / abs(np.mean(list(probs.values()))))
                if difference / abs(np.mean(list(probs.values()))) < prediction_thres:
                    predicted_class = 'undefined'
                else:
                    predicted_class = most_prob
                results.append(predicted_class)
            predicted_class = collections.Counter(results).most_common(1)[0][0]
            return predicted_class
        else:
            probs = {category: model.compute_prob(sentence) for category, model in self.models.items()}
            sorted_probs = sorted(probs, key=probs.get, reverse=True)
            most_prob, second_most_prob = sorted_probs[0:2]
            # print(most_prob, second_most_prob)
            difference = abs(probs[most_prob] - probs[second_most_prob])

            # most_prob_class = max(probs, key=probs.get)
            # perc = probs[predicted_class] / probs[min(probs, key=probs.get)]
            # print('diff', diff)
            # print('perc', perc)
            # print('diff/max', diff / abs(np.mean(list(probs.values()))))

            if difference / abs(np.mean(list(probs.values()))) < prediction_thres:
                predicted_class = 'undefined'
            else:
                predicted_class = most_prob
            return predicted_class

    def evaluate(self, filename, mixture=[], prediction_thres=0):
        """
        Description:        function that evaluates the performance of the classifier
                            for a given test corpus

        Input:
        -filename:          str, the name of the csv file containing the test corpus
        -mixture:           list, specifies the mixture model (default: [])
        -prediction_thres:  float, specifies the minimum required difference divided
                            by the average log probability as produced by the n-gram
                            models to produce a prediction, else the classification
                            returns 'undefined'

        Output:
        -results:           dict, contains the counts of true and false positives
                            and true and false negatives
        """
        results = {'positive': {'true': 0, 'false': 0}, 'negative': {'true': 0, 'false': 0}}
        correct_labels = []
        predicted_labels = []
        data = pd.read_csv(filename, encoding="ISO-8859-1")
        for index, row in data.iterrows():
            sentiment = row['sentiment']
            correct_labels.append(sentiment)
            prediction = self.classify(row['text'], mixture=mixture, prediction_thres=prediction_thres)
            predicted_labels.append(prediction)
            if prediction == 'undefined':
                continue
            if prediction == sentiment:
                results[sentiment]['true'] += 1
            else:
                results[prediction]['false'] += 1
        precision = self.compute_precision(results)
        recall = self.compute_recall(results)
        print(f'precision {precision}, recall {recall}')
        visualize.plot_confusion_matrix(correct_labels, predicted_labels, ['Negative', 'Positive', 'Undefined'])
        return precision, recall, results

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
