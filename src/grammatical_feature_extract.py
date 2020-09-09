import spacy
import math
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def template(dataset: list, refer_dataset=None, refer_term=None, statistic_key='', result_key=' ', step=2, threshold=0.):
    if refer_term is None:
        refer_term = {}
        for sample in refer_dataset:
            for i in range(len(sample[statistic_key]) - (step-1-1)):
                n_gram = " ".join(sample[statistic_key][i:i + step])
                if n_gram in refer_term.keys():
                    refer_term[n_gram] += 1
                else:
                    refer_term[n_gram] = 1

    n_gram_list = []
    for sample in dataset:
        essay_n_gram = {}
        for i in range(len(sample[statistic_key]) - (step-1-1)):
            n_gram = " ".join(sample[statistic_key][i:i + step])
            if n_gram in essay_n_gram.keys():
                essay_n_gram[n_gram] += 1
            else:
                essay_n_gram[n_gram] = 1

        essay_n_gram_rate = {}
        for n_gram in essay_n_gram.keys():
            if n_gram not in refer_term.keys():
                essay_n_gram_rate[n_gram] = 1
            else:
                essay_n_gram_rate[n_gram] = essay_n_gram[n_gram] / refer_term[n_gram]

        sample[result_key] = sum([math.pow(essay_n_gram_rate[n_gram], 1) if essay_n_gram_rate[n_gram] > threshold else 0 for n_gram in essay_n_gram_rate.keys()]) / len(essay_n_gram_rate.keys())
        n_gram_list.append(sample[result_key])

    return refer_term, {result_key: {'mean': np.mean(n_gram_list), 'std': np.std(n_gram_list)}}


def word_bigram(dataset: list, refer_dataset=None, refer_term=None):
    return template(dataset, refer_dataset, refer_term,
                    statistic_key='essay_token', result_key='word_bigram', step=2,
                    threshold=0.1)


def word_trigram(dataset: list, refer_dataset=None, refer_term=None):
    return template(dataset, refer_dataset, refer_term,
                    statistic_key='essay_token', result_key='word_trigram', step=3,
                    threshold=0.2)


def pos_bigram(dataset: list, refer_dataset=None, refer_term=None):
    return template(dataset, refer_dataset, refer_term,
                    statistic_key='essay_pos', result_key='pos_bigram', step=2,
                    threshold=0.001)


def pos_trigram(dataset: list, refer_dataset=None, refer_term=None):
    return template(dataset, refer_dataset, refer_term,
                    statistic_key='essay_pos', result_key='pos_trigram', step=3,
                    threshold=0.002)


def gram_vector_template(dataset, refer_dataset=None, refer_vectorizer=None, refer_vector=None, statistic_key=None, result_key=None, step=-1):
    if refer_vectorizer is None:
        n_gram_corpus = []
        for sample in refer_dataset:
            n_gram_s = []
            for i in range(len(sample[statistic_key]) - (step-1-1)):
                n_gram = "_".join(sample[statistic_key][i:i + step])
                n_gram_s.append(n_gram)
            n_gram_corpus.append(" ".join(n_gram_s))

        refer_vectorizer = CountVectorizer(max_features=1000)
        refer_vectorizer.fit(n_gram_corpus)
        refer_vector = refer_vectorizer.transform(n_gram_corpus)
        refer_vector = np.sum(refer_vector, axis=0)

    n_gram_corpus = []
    for sample in dataset:
        n_gram_s = []
        for i in range(len(sample[statistic_key]) - (step - 1 - 1)):
            n_gram = "_".join(sample[statistic_key][i:i + step])
            n_gram_s.append(n_gram)
        n_gram_corpus.append(" ".join(n_gram_s))

    vector = refer_vectorizer.transform(n_gram_corpus)
    term = vector / refer_vector
    # print(term.shape)
    # print(term[1, 1])

    for (i, sample) in enumerate(dataset):
        for term_num in range(0, term.shape[1]):
            sample[result_key + str(term_num)] = term[i, term_num]

    return refer_vectorizer, refer_vector


def pos_bigram_vector(dataset, refer_dataset, refer_vectorizer=None, refer_vector=None):
    return gram_vector_template(dataset, refer_dataset, refer_vectorizer, refer_vector, statistic_key='essay_pos', result_key='pos_bigram_vector', step=2)