
import regression
import metrics
import copy
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import pickle

import numpy as np

from data import Dataset, split_sentence
import pandas as pd
import itertools

train_dataset = Dataset.load("../../data/essay_data/train-feature-5.p")
dev_dataset = Dataset.load("../../data/essay_data/dev-feature-5.p")
test_dataset = Dataset.load("../../data/essay_data/test-feature-5.p")

feature = [
    'token_count',
    'unique_token_count',
    'no_stop_count',
    'comma_count',
    'special_count',
    'noun_count',
    'verb_count',
    'adv_count',
    'adj_count',
    'pron_count',
    'word_avg_len',
    'word_var_len',
    'spelling_error_rate',
    'mean_word_level',
    'essay_length',
    # 'sent_count',
    # 'sent_avg_len',
    # 'sent_var_len',
    'word_bigram',  # ok
    'word_trigram',  # ok
    'pos_bigram',  # ok
    'pos_trigram',  # ok
    'mean_clause_number',
    'mean_clause_length',
    'mean_sent_depth',
    'mean_sent_level',
    # 'sent_repeat_rate',
    # 'syll_per_word',
    # 'type_token_ratio',
    # 'syllables',
    # 'wordtypes',
    # 'long_words',
    # 'complex_words',
    # 'complex_words_dc',
    # 'tobeverb',
    # 'auxverb',
    # 'pronoun',
    # 'preposition',
    # 'nominalization',
    # 'sentence_beginning_pronoun',
    # 'sentence_beginning_interrogative',
    # 'sentence_beginning_article',
    # 'sentence_beginning_subordination',
    # 'sentence_beginning_conjunction',
    # 'sentence_beginning_preposition',
    # 'FleschReadingEase',
    # 'Kincaid',
    # 'ARI',
    # 'Coleman-Liau',
    # 'GunningFogIndex',
    # 'LIX',
    # 'RIX',
    # 'SMOGIndex',
    # 'DaleChallIndex',
]

# used_set = [
#     [6, 7, 8],  # 1 ok
#     [4, 7],  # 2
#     [4, 5, 6],  # 3
#     [3, 5],  # 4
#     [7],  # 5
#     [3, 5],  # 6
#     [1, 2, 3, 4, 5, 6, 8],
#     [1, 6, 7],
# ]

model_count = [
    120,  # 1
    120,  # 2
    200,  # 3
    200,  # 4
    120,  # 5
    120,  # 6
    100,  # 7
    80,  # 8
]

score_ranges = [
    [2, 12],
    [1, 6],
    [0, 3],
    [0, 3],
    [0, 4],
    [0, 4],
    [0, 30],
    [0, 60],
]


results = []
cross_validate_list = []

for i in range(4, 5):

    _feature = [f for f in feature]

    selected = [j for j in range(1, 9) if i != j]
    all_selected = []
    for j in range(1, 9):
        all_selected.extend(list(itertools.combinations(selected, j)))

    current_results = []
    for selected in all_selected:
        used_set = selected

        x_s = []
        y_s = []
        # print('used_set', used_set)
        for j in used_set:
            x, y = regression.generate_model_input_for_ranges(train_dataset.data[str(j)], _feature, score_ranges[j - 1])
            standerizer = StandardScaler().fit(x)
            x = standerizer.transform(x)

            x_s.append(x)
            y_s.append(y)

        x = np.concatenate(x_s, axis=0)
        y = np.concatenate(y_s, axis=0)

        train_dev_test_x, _ = regression.generate_model_input_for_ranges(
            train_dataset.data[str(i)] + dev_dataset.data[str(i)] + test_dataset.data[str(i)],
            _feature, score_ranges[i - 1])

        dev_test_standerizer = StandardScaler().fit(train_dev_test_x)
        dev_x, dev_y = regression.generate_model_input_for_ranges(dev_dataset.data[str(i)], _feature,
                                                                  score_ranges[i - 1])
        test_x = regression.generate_model_test_input(test_dataset.data[str(i)], _feature)
        dev_x = dev_test_standerizer.transform(dev_x)
        test_x = dev_test_standerizer.transform(test_x)

        regressor = regression.gradient_boosting_regressor(x, y)
                                                           # num_estimators=int(x.shape[0] / 20) + model_count[i - 1])
        predict_dev_y = regressor.predict(dev_x)

        predict_dev_y = [predict * (score_ranges[i - 1][1] - score_ranges[i - 1][0]) + score_ranges[i - 1][0] for predict
                         in predict_dev_y]
        # dev_predict = np.around(dev_predict)
        # print(dev_predict)
        dev_y = [y * (score_ranges[i - 1][1] - score_ranges[i - 1][0]) + score_ranges[i - 1][0] for y in dev_y]

        dev_result = metrics.kappa(y_true=dev_y, y_pred=predict_dev_y, weights='quadratic')
        current_results.append(dev_result)
        # print(dev_result)

    print(max(current_results))
    index = current_results.index(max(current_results))
    print(all_selected[index])
    results.append(max(current_results))


print(np.average(results))
# print(np.average(results))