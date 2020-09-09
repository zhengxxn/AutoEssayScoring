import math
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy.linalg import svd
import numpy as np
from sklearn import decomposition
from Levenshtein import ratio
import random
import matplotlib.pyplot as plt
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel


def lda_vector(dataset: list, refer_dictionary=None, refer_lda_model=None):

    if refer_dictionary is None:
        refer_docs = [
          [token for (i, token) in enumerate(sample['essay_lemma']) if sample['essay_is_stop'][i] is False
           and token not in [',', '.', '?']] for sample in dataset
        ]
        refer_dictionary = Dictionary(refer_docs)
        refer_doc2bow = [refer_dictionary.doc2bow(text) for text in refer_docs]
        refer_lda_model = LdaModel(corpus=refer_doc2bow, id2word=refer_dictionary, num_topics=10, dtype=np.float64, passes=10, minimum_probability=0.0)

    doc = [
        [token for (i, token) in enumerate(sample['essay_lemma']) if sample['essay_is_stop'][i] is False
         and token not in [',', '.', '?']] for sample in dataset
    ]
    doc_bow_s = [refer_dictionary.doc2bow(text) for text in doc]
    doc_vecs = [refer_lda_model[doc_bow] for doc_bow in doc_bow_s]

    for (sample, doc_vec) in zip(dataset, doc_vecs):
        for topic_prob in doc_vec:
            sample['topic'+str(topic_prob[0] + 1)] = topic_prob[1]

    return refer_dictionary, refer_lda_model


def first_sentence_score(dataset: list, refer_dataset):
    """
    no useful
    :param dataset:
    :param refer_dataset:
    :return:
    """
    similarity_score_list = []
    for sample in dataset:
        sample_refer_dataset = random.sample(refer_dataset, 50)
        first_sentence = sample['essay_sent'][0]

        refer_first_sentences = [sample['essay_sent'][0] for sample in sample_refer_dataset]
        similarities = [ratio(first_sentence, sent) for sent in refer_first_sentences]
        scores = [refer_sample['domain1_score'] for refer_sample in sample_refer_dataset]
        similarity_score = np.average([similarity * score for similarity, score in zip(similarities, scores)])
        sample['first_sentence_score'] = similarity_score
        similarity_score_list.append(similarity_score)

    return {'first_sentence_score': {'mean': np.mean(similarity_score_list), 'std': np.std(similarity_score_list)}}


def last_sentence_score(dataset: list, refer_dataset):
    """
    no useful
    :param dataset:
    :param refer_dataset:
    :return:
    """
    similarity_score_list = []
    for sample in dataset:
        sample_refer_dataset = random.sample(refer_dataset, 50)
        last_sentence = sample['essay_sent'][-1]

        refer_last_sentences = [sample['essay_sent'][-1] for sample in sample_refer_dataset]
        similarities = [ratio(last_sentence, sent) for sent in refer_last_sentences]
        scores = [refer_sample['domain1_score'] for refer_sample in sample_refer_dataset]
        similarity_score = np.average([similarity * score for similarity, score in zip(similarities, scores)])
        sample['last_sentence_score'] = similarity_score
        similarity_score_list.append(similarity_score)

    return {'last_sentence_score': {'mean': np.mean(similarity_score_list), 'std': np.std(similarity_score_list)}}


def text_similarity_score(dataset: list, refer_dataset):
    """
    no useful
    :param dataset:
    :param refer_dataset:
    :return:
    """
    similarity_score_list = []
    for sample in dataset:
        sample_refer_dataset = random.sample(refer_dataset, 20)

        similarities = [ratio(sample['essay'], refer_sample['essay']) for refer_sample in sample_refer_dataset]
        # top_k = simi.argsort()[-int(refer_matrix.shape[1]/50):][::-1].tolist()

        scores = [refer_sample['domain1_score'] for refer_sample in sample_refer_dataset]
        similarity_score = np.average([similarity * score for similarity, score in zip(similarities, scores)])
        sample['text_similarity_score'] = similarity_score
        similarity_score_list.append(similarity_score)
        # print(np.mean(similarity_score), sample['domain1_score'])

    return {'text_similarity_score': {'mean': np.mean(similarity_score_list), 'std': np.std(similarity_score_list)}}


def sent_repeat_rate(dataset: list):

    sent_repeat_rate_list = []
    for sample in dataset:

        sents = sample['essay_sent']
        sents_token = [len(sent.split(' ')) for sent in sents]
        sents_token_no_repeat = [len(set(sent.split(' '))) for sent in sents]
        sample['sent_repeat_rate'] = np.average([no_repeat / total for no_repeat, total in zip(sents_token_no_repeat, sents_token)])
        sent_repeat_rate_list.append(sample['sent_repeat_rate'])

    return {'sent_repeat_rate': {'mean': np.mean(sent_repeat_rate_list), 'std': np.std(sent_repeat_rate_list)}}


def essay_length(dataset: list):
    """
    implemented without tokenize
    :param dataset:
    :return:
    """

    all_essay_length = [math.pow(len(sample['essay'].split(' ')), 0.25) for sample in dataset]

    mean_len = np.mean(all_essay_length)
    std_len = np.std(all_essay_length)

    for length, sample in zip(all_essay_length, dataset):
        sample['essay_length'] = length

    return {'essay_length': {'mean': mean_len, 'std': std_len}}


def semantic_vector_similarity(dataset: list, tv=None, refer_matrix=None, refer_dataset=None):

    # essays = []
    # for sample in dataset:
    #     essays.append(" ".join([token for i, token in enumerate(sample['essay_lemma'])
    #                             if sample['essay_is_stop'][i] is False and len(token) > 2]))
    #
    essays = [" ".join(sample['essay_lemma']) for sample in dataset]
    if tv is None:
        tv = TfidfVectorizer(use_idf=True, smooth_idf=True, norm='l2')
        tv.fit(essays)
    tv_fit = tv.transform(essays)
    matrix = tv_fit.toarray().transpose()

    if refer_matrix is None:
        refer_matrix = matrix
    if refer_dataset is None:
        refer_dataset = dataset

    _, _, semantic_matrix = decomposition.randomized_svd(matrix, int(refer_matrix.shape[1]/200))
    _, _, refer_semantic_matrix = decomposition.randomized_svd(refer_matrix, int(refer_matrix.shape[1]/200))
    vector_len = np.sqrt(np.sum(semantic_matrix**2, axis=0))
    semantic_matrix = np.divide(semantic_matrix, vector_len)
    vector_len = np.sqrt(np.sum(refer_semantic_matrix**2, axis=0))
    refer_semantic_matrix = np.divide(refer_semantic_matrix, vector_len)

    result_list = []

    for t, sample in enumerate(dataset):
        sim = np.matmul(semantic_matrix[:, t].T, refer_semantic_matrix)

        # top_k = sim.argsort()[-int(refer_matrix.shape[1]/15):][::-1].tolist()
        top_k = sim.argsort()[:][::-1].tolist()

        weighted_sim = [refer_dataset[k]['domain1_score'] * sim[k] for k in top_k]
        result = np.sum(weighted_sim)

        result_list.append(result)
        sample['semantic_vector_similarity'] = result

    return tv, refer_matrix, {'semantic_vector_similarity': {'mean': np.mean(result_list), 'std': np.std(result_list)}}


def text_coherence(dataset: list):
    pass
