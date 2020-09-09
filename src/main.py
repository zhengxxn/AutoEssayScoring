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
import operator


def predict_bag(dev_predict_list):
    count = dev_predict_list[0].shape[0]
    dev_predict_ = []
    for i in range(0, count):
        select_result = {}
        for dev_predict in dev_predict_list:
            if int(dev_predict[i]) not in select_result.keys():
                select_result[int(dev_predict[i])] = 0
            select_result[int(dev_predict[i])] += 1
        # print(select_result)
        dev_predict_.append(max(select_result.items(), key=operator.itemgetter(1))[0])

    return dev_predict_


def save_to_tsv(samples: list, tsv_file):
    raw_data = {
        'id': [sample['essay_id'] for sample in samples],
        'set': [sample['essay_set'] for sample in samples],
        'score': [sample['domain1_score'] for sample in samples]
    }
    df = pd.DataFrame(raw_data)
    df.to_csv(tsv_file, sep='\t', index=False, header=False)


def preprocess():
    _train_dataset = Dataset()
    _train_dataset.load_from_raw_file('../data/essay_data/train.tsv',
                                      ['essay_set', 'essay_id', 'essay', 'domain1_score'])
    split_sentence(_train_dataset)
    Dataset.save(_train_dataset, '../data/essay_data/train-preprocess.pickle')

    _dev_dataset = Dataset()
    _dev_dataset.load_from_raw_file('../data/essay_data/dev.tsv', ['essay_set', 'essay_id', 'essay', 'domain1_score'])
    split_sentence(_dev_dataset)
    Dataset.save(_dev_dataset, '../data/essay_data/dev-preprocess.pickle')

    _test_dataset = Dataset()
    _test_dataset.load_from_raw_file('../data/essay_data/test.tsv', ['essay_set', 'essay_id', 'essay'])
    split_sentence(_test_dataset)
    Dataset.save(_test_dataset, '../data/essay_data/test-preprocess.pickle')


# load_from_raw_file includes tokenize process, which is time consuming
#
# train_dataset = Dataset()
# train_dataset.load_from_raw_file('../data/essay_data/train.tsv', ['essay_set', 'essay_id', 'essay', 'domain1_score'])
# Dataset.save(train_dataset, '../data/essay_data/train.p')
#
# dev_dataset = Dataset()
# dev_dataset.load_from_raw_file('../data/essay_data/dev.tsv', ['essay_set', 'essay_id', 'essay', 'domain1_score'])
# Dataset.save(dev_dataset, '../data/essay_data/dev.p')
#
# test_dataset = Dataset()
# test_dataset.load_from_raw_file('../data/essay_data/test.tsv', ['essay_set', 'essay_id', 'essay'])
# Dataset.save(test_dataset, '../data/essay_data/test.p')

# split_sentence(train_dataset)
# split_sentence(dev_dataset)
# split_sentence(test_dataset)
# Dataset.save(train_dataset, '../data/essay_data/train.p')
# Dataset.save(dev_dataset, '../data/essay_data/dev.p')
# Dataset.save(test_dataset, '../data/essay_data/test.p')

# dataset.data is a dictionary, keys are {1, 2, 3..., 8} means eight essay sets.
# the value of dict is a list, contains the samples of each essay set
# the element of each list is a dictionary, keys are attribute
#
# dataset.data = {'1': , '2': , ..., '8': }
#
# dataset.data['1'] = [
# {'essay_id':, 'essay': , 'domain1_score': , 'word_avg_len': , 'word_var_len': , ...},
# {'essay_id':, 'essay': , 'domain1_score': , 'word_avg_len': , 'word_var_len': , ...},
# {'essay_id':, 'essay': , 'domain1_score': , 'word_avg_len': , 'word_var_len': , ...},
# ...
# ]
#
# dataset.data['2'] = [
# {'essay_id':, 'essay': , 'domain1_score': , 'word_avg_len': , 'word_var_len': , ...},
# {'essay_id':, 'essay': , 'domain1_score': , 'word_avg_len': , 'word_var_len': , ...},
# {'essay_id':, 'essay': , 'domain1_score': , 'word_avg_len': , 'word_var_len': , ...},
# ...
# ]
# ...
#

# this is time consuming


predict_list = []
label_list = []
all_dev_samples = []
all_test_samples = []
extract_feature = [
    'token_count',
    'pos_count',
    'word_len',
    'spelling_error',
    'vector_similarity',
    'mean_word_level',
    'essay_len',
    'sent_len',
    'sent_count',
    'word_bigram',
    'word_trigram',
    'pos_bigram',
    'pos_trigram',
    # 'pos_bigram_vector',
    # 'mean_clause_number',
    # 'mean_clause_depth',
    # 'mean_sent_depth',
    # 'mean_sent_level'
    'sent_repeat_rate',
    # 'text_similarity_score',
    # 'lda_vector',
    # 'first_sentence_score',
    # 'last_sentence_score',
    'readability',
]
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
    'sent_count',
    'sent_avg_len',
    'sent_var_len',
    'word_bigram',  # ok
    'word_trigram',  # ok
    'pos_bigram',  # ok
    'pos_trigram',  # ok
    'mean_clause_number',
    'mean_clause_length',
    'mean_sent_depth',
    'mean_sent_level',
    'sent_repeat_rate',
    'syll_per_word',
    'type_token_ratio',
    'syllables',
    'wordtypes',
    'long_words',
    'complex_words',
    'complex_words_dc',
    'tobeverb',
    'auxverb',
    'pronoun',
    'preposition',
    'nominalization',
    'sentence_beginning_pronoun',
    'sentence_beginning_interrogative',
    'sentence_beginning_article',
    'sentence_beginning_subordination',
    'sentence_beginning_conjunction',
    'sentence_beginning_preposition',
    'FleschReadingEase',
    'Kincaid',
    'ARI',
    'Coleman-Liau',
    'GunningFogIndex',
    'LIX',
    'RIX',
    'SMOGIndex',
    'DaleChallIndex',
]


def output_feature(dataset, cover_feature_names, file):
    dataset_features = {}
    for i in range(1, 9):
        for sample in dataset.data[str(i)]:
            features = []
            # features.append(sample['essay_id'])
            # features.append(sample['essay_set'])
            for feature_name in cover_feature_names:
                if feature_name in sample.keys():
                    features.append(sample[feature_name])
            dataset_features[sample['essay_id']] = features

    with open(file, 'wb') as f:
        pickle.dump(dataset_features, f)


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

used_set = [
    [3, 5, 6, 8],  # 1 ok
    [4, 7],  # 2
    [4, 5, 6],  # 3
    [3, 5],  # 4
    [1, 3, 4, 6],  # 5
    [1, 3, 4, 5],  # 6
    [1, 2, 3, 4, 6, 8],
    [1, 7],
]

model_count = [
    120,  # 1
    120,  # 2
    120,  # 3
    120,  # 4
    200,  # 5
    200,  # 6
    80,  # 7
    80,  # 8
]


def main():
    results = []
    cross_validate_list = []

    for i in range(1, 9):

        _feature = feature

        # dev_x, dev_y = regression.generate_model_input(dev_dataset.data[str(i)][:], _feature)
        # x, y = regression.generate_model_input(train_dataset.data[str(i-1)] + dev_dataset.data[str(i-1)][:-100], _feature)
        # x, y = regression.generate_model_input(train_dataset.data[str(i + 1)],
        #                                        _feature)
        # test_x = regression.generate_model_test_input(test_dataset.data[str(i)], _feature)

        x_s = []
        y_s = []
        print('used_set', used_set[i - 1])
        for j in used_set[i - 1]:
            x, y = regression.generate_model_input_for_ranges(train_dataset.data[str(j)], _feature, score_ranges[j - 1])
            standerizer = StandardScaler().fit(x)
            x = standerizer.transform(x)

            x_s.append(x)
            y_s.append(y)

        x = np.concatenate(x_s, axis=0)
        y = np.concatenate(y_s, axis=0)
        # print(x[0])

        train_dev_test_x, _ = regression.generate_model_input_for_ranges(
            train_dataset.data[str(i)] + dev_dataset.data[str(i)] + test_dataset.data[str(i)],
            _feature, score_ranges[i - 1])

        dev_test_standerizer = StandardScaler().fit(train_dev_test_x)
        dev_x, dev_y = regression.generate_model_input_for_ranges(dev_dataset.data[str(i)], _feature,
                                                                  score_ranges[i - 1])
        test_x = regression.generate_model_test_input(test_dataset.data[str(i)], _feature)
        dev_x = dev_test_standerizer.transform(dev_x)
        test_x = dev_test_standerizer.transform(test_x)

        # standerizer = StandardScaler().fit(x)
        # x = standerizer.transform(x)

        # standerizer = StandardScaler().fit(x)
        # _train_x = standerizer.transform(x)
        # _dev_x = standerizer.transform(dev_x)
        # _test_x = standerizer.transform(test_x)

        kf = KFold(n_splits=9)

        dev_predict_list = []
        model_list = []
        test_list = []
        dev_result_list = []
        test_result_list = []

        for train_index, test_index in kf.split(x, y):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # print(x_train[0])
            # print('train size', x_train.shape[0])

            standerizer = StandardScaler().fit(x_train)

            _x_train = standerizer.transform(x_train)
            _x_test = standerizer.transform(x_test)
            _dev_x = standerizer.transform(dev_x)
            _test_x = standerizer.transform(test_x)

            # _x_train = x_train
            # _x_test = x_test
            # _dev_x = dev_x
            # _test_x = test_x

            # regressor = regression.svr_regressor(_x_train, y_train)
            regressor = regression.gradient_boosting_regressor(_x_train, y_train,
                                                               num_estimators=int(_x_train.shape[0] / 20) + model_count[
                                                                   i - 1])
            # regressor = regression.gradient_boosting_regressor(_x_train, y_train)
            model_list.append(regressor)

            predict_dev_y = regressor.predict(_dev_x)

            _dev_y = [predict * (score_ranges[i - 1][1] - score_ranges[i - 1][0]) + score_ranges[i - 1][0] for
                      predict in dev_y]
            _predict_dev_y = [predict * (score_ranges[i - 1][1] - score_ranges[i - 1][0]) + score_ranges[i - 1][0] for
                              predict in predict_dev_y]

            dev_result = metrics.kappa(y_true=_dev_y, y_pred=_predict_dev_y, weights='quadratic')

            # print(dev_result)
            dev_result_list.append(dev_result)
            # dev_predict_list.append(np.around(_predict_dev_y))
            dev_predict_list.append(_predict_dev_y)

            # y_test_predict = regressor.predict(_x_test)
            # test_result = metrics.kappa(y_true=y_test, y_pred=y_test_predict, weights='quadratic')
            # test_result_list.append(test_result)
            #
            test_predict_y = regressor.predict(_test_x)
            # test_predict_y = [predict * (score_ranges[i - 1][1] - score_ranges[i - 1][0]) + score_ranges[i - 1][0] for
            #                   predict in test_predict_y]
            test_list.append(test_predict_y)
            # test_list.append(np.around(test_predict_y))

        dev_predict = [sum(x) / len(dev_predict_list) for x in zip(*dev_predict_list)]
        # dev_predict = predict_bag(dev_predict_list)
        # dev_predict = [predict * (score_ranges[i - 1][1] - score_ranges[i - 1][0]) + score_ranges[i - 1][0] for predict
        #                in dev_predict]
        dev_predict = np.around(dev_predict)
        # print(dev_predict)
        dev_y = [y * (score_ranges[i - 1][1] - score_ranges[i - 1][0]) + score_ranges[i - 1][0] for y in dev_y]

        dev_result = metrics.kappa(y_true=dev_y, y_pred=dev_predict, weights='quadratic')

        # print('cross validate average', np.average(test_result_list))
        print('dev ', i, ' ', dev_result)
        # cross_validate_list.append(np.average(test_result_list))
        results.append(dev_result)

        # test_predict_y = predict_bag(test_list)
        #
        test_predict_y = [sum(x) / len(test_list) for x in zip(*test_list)]
        test_predict_y = [predict * (score_ranges[i - 1][1] - score_ranges[i - 1][0]) + score_ranges[i - 1][0] for
                          predict
                          in test_predict_y]
        # print('average ', np.average(test_predict_y))

        # test_predict_y = np.around(test_predict_y)

        for idx, sample in enumerate(test_dataset.data[str(i)]):
            sample['domain1_score'] = int(test_predict_y[idx])
        all_test_samples.extend(test_dataset.data[str(i)])

        save_to_tsv(test_dataset.data[str(i)], '../' + str(i) + '.tsv')
    print(np.average(results))


# train_dataset = Dataset.load("../data/essay_data/train-preprocess.pickle")
# dev_dataset = Dataset.load("../data/essay_data/dev-preprocess.pickle")
# test_dataset = Dataset.load("../data/essay_data/test-preprocess.pickle")
train_dataset = Dataset.load("../data/train-feature-5.p")
dev_dataset = Dataset.load("../data/dev-feature-5.p")
test_dataset = Dataset.load("../data/test-feature-5.p")
#
# train_dataset = Dataset.load("../../data/essay_data/train-entire.p")
# dev_dataset = Dataset.load("../../data/essay_data/dev-entire.p")
# test_dataset = Dataset.load("../../data/essay_data/test-entire.p")
# #
# print('load complete')

# _extract_feature()
# parser()
#
# for feature_name in train_dataset.data['1'][0].keys():
#     if 'topic' in feature:
#         if feature_name.find('topic') != -1:
#             feature.append(feature_name)
#     if 'pos_bigram_vector' in feature:
#         if feature_name.find('pos_bigram_vector') != -1:
#             feature.append(feature_name)

main()
# output_feature(train_dataset, feature, '/Users/zx/Documents/Project/data/train.feature')
# output_feature(dev_dataset, feature, '/Users/zx/Documents/Project/data/dev.feature')
# output_feature(test_dataset, feature, '/Users/zx/Documents/Project/data/test.feature')
#
# Dataset.save(train_dataset, '../data/essay_data/train-entire.p')
# Dataset.save(dev_dataset, '../data/essay_data/dev-entire.p')
# Dataset.save(test_dataset, '../data/essay_data/test-entire.p')

# Dataset.save_feature(train_dataset, '../../data/essay_data/train-feature-5.p')
# Dataset.save_feature(dev_dataset, '../../data/essay_data/dev-feature-5.p')
# Dataset.save_feature(test_dataset, '../../data/essay_data/test-feature-5.p')


# with open('/Users/zx/Documents/Project/data/dev.feature', 'rb') as f:
#     data = {}
#     data = pickle.load(f)
#     print(len(data[21115]))
