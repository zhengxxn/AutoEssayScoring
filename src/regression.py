from sklearn.svm import SVR, SVC
import numpy as np
import copy

import xgboost as xgb
from xgboost.sklearn import XGBRegressor, XGBClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from ranking import RankSVM
from xgboostextension import XGBRanker


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


def generate_model_input_for_ranges(dataset: list, cover_feature_names, score_range):
    x = []
    y = []
    # print(score_range)
    feature_name = cover_feature_names
    # feature_name.sort()

    for sample in dataset:
        y.append((sample['domain1_score'] - score_range[0]) / (score_range[1] - score_range[0]))

        # name_list = []
        # for name in sample.keys():
        #     if name in feature_name:
        #         name_list.append(name)
        # print(name_list)

        feature = [sample[name] for name in feature_name if name in sample.keys()]
        # feature = [sample[name] for name in sample.key() if name in feature_name]

        x.append(feature)

    _x = np.array(x)
    _target = np.array(y)

    return _x, _target


def generate_model_input(dataset: list, cover_feature_names):
    x = []
    y = []

    feature_name = cover_feature_names
    feature_name.sort()

    for sample in dataset:
        y.append(sample['domain1_score'])

        feature = [sample[name] for name in feature_name if name in sample.keys()]
        # feature = [sample[name] for name in sample.key() if name in feature_name]

        x.append(feature)

    _x = np.array(x)
    _target = np.array(y)

    return _x, _target


def generate_model_test_input(dataset: list, cover_feature_names):
    x = []

    feature_name = cover_feature_names
    # feature_name.sort()

    for sample in dataset:
        feature = [sample[name] for name in feature_name if name in sample.keys()]
        x.append(feature)

    _x = np.array(x)
    return _x


def rank_svm(x, y):
    clf = RankSVM()
    clf.fit(x, y)
    return clf


def svr_regressor(x, y):
    clf = Pipeline([('feature_selection', SelectFromModel(Lasso(), threshold='median', max_features=8)),
                    ('regression', SVR(kernel='rbf', gamma='scale', C=0.5, epsilon=0.2))])
    clf.fit(x, y)
    return clf


def xgb_classifier(x, y):

    model = XGBClassifier(learning_rate=0.1, subsample=0.6, max_depth=1, objective='rank:pairwise', n_estimators=100, random_state=0)
    model.fit(x, y)
    return model


def xgb_ranker(x, y):
    ranker = XGBRanker(n_estimators=500, learning_rate=0.1, subsample=0.9)
    ranker.fit(x, y, eval_metric=['ndcg', 'map@5-'])
    return ranker

    '''
    model = XGBRegressor (learning_rate=0.1,
                          subsample=0.8,
                          max_depth=3,
                          objective='rank:pairwise',
                          n_estimators=100,
                          random_state=0,
                          colsample_bytree=1,
                          gamma=0,
                          reg_alpha=0,
                          reg_lambda=1,
                          max_delta_step=0,
                          scale_pos_weight=1)

    watch_list = [(x_train, y_train), (x_test, y_test)]
    model.fit(x_train, y_train, eval_set=watch_list, early_stopping_rounds=10)
    return model
    '''


def xgb_regressor(x, y):
    model = XGBRegressor(n_estimators=33, learning_rate=0.1, subsample=0.6, max_depth=1, objective='rank:pairwise', random_state=0)
    model.fit(x, y)
    return model


def svr_classifier(x, y):
    model = SVC(gamma='auto')
    model.fit(x, y)
    return model


def gradient_boosting_regressor(x, y, num_estimators=200):

    model = GradientBoostingRegressor(n_estimators=num_estimators, learning_rate=0.1, max_depth=1, subsample=0.55, random_state=3, loss='huber')
    model.fit(x, y)
    return model


def gradient_boosting_classifier(x, y):

    model = GradientBoostingClassifier(n_estimators=120, learning_rate=0.1, subsample=0.55, random_state=3, max_depth=3)
    model.fit(x, y)
    return model


def ridge_regression(x, y):
    model = Ridge()
    model.fit(x, y)
    return model


def random_forest_regression(x, y):
    model = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)
    model.fit(x, y)
    return model


def xgb_regression(x, y):
    model = xgb.XGBRegressor()
    model.fit(x, y)
    return model


def random_forest_classifier(x, y):
    model = RandomForestClassifier()
    model.fit(x, y)
    return model


def elastic_net(x, y):
    paramgrid = {'l1_ratio': [.01, .1, .5, .9], 'alpha': [0.01, .1, 1]}
    model = GridSearchCV(ElasticNet(max_iter=100000, random_state=26),
                      param_grid=paramgrid,
                      cv=5)
    model.fit(x, y)
    return model


def bagging_regression(x, y):
    model = BaggingRegressor(base_estimator=SVR(kernel='rbf', gamma='scale', C=1.0, epsilon=0.2),
                             n_estimators=66,
                             max_samples=0.9,
                             max_features=0.9,
                             n_jobs=-1,
                             random_state=0)
    model.fit(x, y)
    return model


def get_real_y_predict(y_predict, y_refer):
    y_range = [np.min(y_refer), np.max(y_refer)]
    y_count = [y_refer.tolist().count(i) for i in range(y_range[0], y_range[1] + 1)]
    print('range', y_range)
    print('count', y_count)

    y_rate = [count / sum(y_count) for count in y_count]
    predict_y_count = [y_predict.shape[0] * rate for rate in y_rate]
    predict_y_count = np.ceil(predict_y_count)
    predict_y_count = [int(x) for x in predict_y_count]
    print(predict_y_count)
    index = np.argsort(y_predict)
    y_true_prdict = np.zeros_like(y_predict)
    i = 0
    score = y_range[0]
    for count in predict_y_count:
        for j in range(0, count):
            if i >= len(index):
                break
            y_true_prdict[index[i]] = score
            i += 1
        score += 1

    return y_true_prdict