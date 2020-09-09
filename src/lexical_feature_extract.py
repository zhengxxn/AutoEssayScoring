import numpy as np
from spellchecker import SpellChecker
from collections import defaultdict
import math
from sklearn.feature_extraction.text import TfidfVectorizer


def token_counts(dataset: list):

    token_count_list = []
    unique_token_count_list = []
    no_stop_count_list = []
    comma_count_list = []
    special_count_list = []

    for sample in dataset:
        sample['long_token_count'] = len([token for token in sample['essay_token'] if len(token) > 6])
        sample['token_count'] = len(sample['essay_token'])
        sample['unique_token_count'] = len(set(sample['essay_token']))
        sample['no_stop_count'] = sample['essay_is_stop'].count(False)
        sample['comma_count'] = sample['essay_token'].count(',')
        sample['special_count'] = len([token for token in sample['essay_token'] if token[0] == '@'])

        token_count_list.append(sample['token_count'])
        unique_token_count_list.append(sample['unique_token_count'])
        no_stop_count_list.append(sample['no_stop_count'])
        comma_count_list.append(sample['comma_count'])
        special_count_list.append(sample['special_count'])

    return {
        'token_count': {'mean': np.mean(token_count_list), 'std': np.std(token_count_list)},
        'unique_token_count': {'mean': np.mean(unique_token_count_list), 'std': np.std(unique_token_count_list)},
        'no_stop_count': {'mean': np.mean(no_stop_count_list), 'std': np.std(no_stop_count_list)},
        'comma_count': {'mean': np.mean(comma_count_list), 'std': np.std(comma_count_list)},
        'special_count': {'mean': np.mean(special_count_list), 'std': np.std(special_count_list)}
    }


def pos_counts(dataset: list):

    noun_count_list = []
    verb_count_list = []
    adv_count_list = []
    adj_count_list = []
    pron_count_list = []

    for sample in dataset:
        sample['noun_count'] = sample['essay_pos'].count('NOUN')
        sample['verb_count'] = sample['essay_pos'].count('VERB')
        sample['adv_count'] = sample['essay_pos'].count('ADV')
        sample['adj_count'] = sample['essay_pos'].count('ADJ')
        sample['pron_count'] = sample['essay_pos'].count('PRON')

        noun_count_list.append(sample['noun_count'])
        verb_count_list.append(sample['verb_count'])
        adv_count_list.append(sample['adv_count'])
        adj_count_list.append(sample['adj_count'])
        pron_count_list.append(sample['pron_count'])

    return {
        'noun_count': {'mean': np.mean(noun_count_list), 'std': np.std(noun_count_list)},
        'verb_count': {'mean': np.mean(noun_count_list), 'std': np.std(noun_count_list)},
        'adv_count': {'mean': np.mean(adv_count_list), 'std': np.std(adv_count_list)},
        'adj_count': {'mean': np.mean(adj_count_list), 'std': np.std(adj_count_list)},
        'pron_count': {'mean': np.mean(pron_count_list), 'std': np.std(pron_count_list)},
    }


def mean_variance_word_length(dataset: list):
    all_essay_words = [data_sample['essay_token'] for data_sample in dataset]
    all_essay_words_len = [[len(word) for word in essay_words] for essay_words in all_essay_words]

    all_essay_words_len_average = [np.average(essay_words_len) for essay_words_len in all_essay_words_len]
    all_essay_words_len_variance = [np.var(essay_words_len) for essay_words_len in all_essay_words_len]

    avg_len_list = []
    var_len_list = []

    for data_sample, avg_len, var_len in zip(dataset, all_essay_words_len_average, all_essay_words_len_variance):
        data_sample['word_avg_len'] = avg_len
        data_sample['word_var_len'] = var_len
        avg_len_list.append(avg_len)
        var_len_list.append(var_len)

    return {'word_avg_len': {'mean': np.mean(avg_len_list), 'std': np.std(avg_len_list)},
            'word_var_len': {'mean': np.mean(var_len_list), 'std': np.std(var_len_list)}}


def mean_word_level(dataset: list):
    word_level_dict = {}
    with open("../../data/essay_data/word-levels.csv", 'r') as f:
        lines = f.readlines()
        for line in lines:
            tokens = line.split(',')
            word_level_dict[tokens[2]] = int(tokens[-1][:-1])

    avg_word_level_list = []
    var_word_level_list = []
    for sample in dataset:
        tokens = [token.lower() for token in sample['essay_lemma']]
        tokens_level = [word_level_dict[token.lower()] for token in tokens if token.lower() in word_level_dict.keys()]
        average_level = sum(tokens_level) / len(tokens_level)

        sample['mean_word_level'] = average_level
        avg_word_level_list.append(average_level)

    return {'mean_word_level': {'mean': np.mean(avg_word_level_list), 'std': np.std(avg_word_level_list)},
            }


def word_vector_similarity(dataset: list, tv=None, refer_matrix=None, refer_dataset=None):

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

    # Compute similarity
    result_list = []

    for t, sample in enumerate(dataset):
        sim = np.matmul(matrix[:, t].T, refer_matrix)
        top_k = sim.argsort()[-int(refer_matrix.shape[1]/50):][::-1].tolist()
        weighted_sim = [refer_dataset[k]['domain1_score'] * sim[k] for k in top_k]
        result = np.sum(weighted_sim)

        result_list.append(result)
        sample['word_vector_similarity'] = result

    return tv, refer_matrix, {'word_vector_similarity': {'mean': np.mean(result_list), 'std': np.std(result_list)}}


def spelling_errors(dataset: list):
    spell_checker = SpellChecker()

    all_essay_words = [data_sample['essay_token'] for data_sample in dataset]
    _all_essay_error_words = [spell_checker.unknown(essay_words) for essay_words in all_essay_words]

    all_essay_error_words = []
    for essay_error_words in _all_essay_error_words:
        error_words = [error_word for error_word in essay_error_words if error_word[0] != '@' and len(error_word) > 2]
        all_essay_error_words.append(error_words)

    num_of_error_words = [len(error_words) for error_words in all_essay_error_words]
    num_of_total_words = [len(words) for words in all_essay_words]

    spell_error_list = []
    for data_sample, error_words_count, total_word in zip(dataset, num_of_error_words, num_of_total_words):
        data_sample['spelling_error_rate'] = math.pow(error_words_count / total_word, 1)
        spell_error_list.append(data_sample['spelling_error_rate'])

    return {'spelling_error_rate': {'mean': np.mean(spell_error_list), 'std': np.std(spell_error_list)}}