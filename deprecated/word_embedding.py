import numpy as np
import os
import re
from scipy.sparse import dok_matrix, csr_matrix
from time import time
import _pickle as pickle

def create_docs(directory, sentiment):
    res = []
    files = os.listdir(directory)
    all_words = []
    for filename in files:
        with open(directory + '/' + filename) as text:
            for line in text:
                line = re.sub(r'[^A-z0-9\s]', '', line)
                all_words += line.split()
                res.append((line, sentiment))
            text.close()
    docs, sentiments = list(zip(*res))
    all_words = sorted(list(set(all_words)))
    return docs, sentiments, all_words

x_pos, y_pos, all_words_pos = create_docs('corpora/review_polarity/txt_sentoken/pos', 'positive')
x_neg, y_neg, all_words_neg = create_docs('corpora/review_polarity/txt_sentoken/neg', 'negative')

all_words = sorted(list(set(all_words_pos + all_words_neg)))

# print(len(a))
#
# result = dok_matrix((len(a), len(a)))
#
# result[0, 1] = 1
# result[0, 3] = 1
# result[1, :] = 1
#
# print(result[0, :] * result[1, :].T)

docs = x_pos + x_neg

print(len(docs))

def vector_embed(documents, context_range, all_words):
    all_mapping = {word: index for index, word in enumerate(all_words)}
    result = csr_matrix((len(all_words), len(all_words)))
    t1 = time()
    for document in documents:
        document = document.split()
        for i, word in enumerate(document):
            # if word not in all_words:
            #     continue
            minimum, maximum = [max(0, i-context_range[0]), min(len(document), i+context_range[1])]
            context = document[minimum:maximum]
            word_index = all_mapping[word]
            context_indices = [all_mapping[context_word] for context_word in context if context_word != word]
            #context_indices.remove(word_index)
            for j in context_indices:
                result[word_index, j] += 1
                #np.add.at(result[word_index, :], context_indices, np.ones(len(context_indices)))
    print(time()-t1)
    return result

def save_models(filename, model):
    pickle.dump(model, open(filename, 'wb'))

def load_models(filename):
    return pickle.load(open(filename, 'rb'))

def compute_dist(point1, point2):
    return (point1 * point2.T).multiply((1 / (np.linalg.norm(point1) * np.linalg.norm(point2))))

def cluster_assign(data, means):
    num_points = np.shape(data)[0]
    B = np.zeros(num_points)
    for i in range(num_points):
        dist = [(j, compute_dist(data[i,:], means[j])) for j in range(len(means))]
        B[i] = sorted(dist, key=lambda x: x[1])[0][0]
    return B

def k_means_clustering(data, k=15):
    init_indices = np.random.choice(data.shape[0], k, replace=False)
    init_means = [data[index, :] for index in init_indices]
    print(compute_dist(init_means[0], init_means[1]))
    #print(init_means)

#model = vector_embed(docs[:100], (4, 4), all_words)

#save_models('vector_model2.p', model)

t2 = time()
model = load_models('vector_model2.p')
print(time()-t2)

k_means_clustering(model)

#new_sentence = vector_embed(['what the fuck is this garbage'], (3, 3), all_words)

#print(new_sentence)
#print(r[2594, :])
