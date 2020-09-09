import pandas as pd
# from util.metrics import kappa
import numpy as np
from scipy.stats import normaltest
import math
import matplotlib.pyplot as plt
import seaborn as sns
import random
import statistics
from scipy.stats import wasserstein_distance
import itertools

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

train_dataset_file = '/Users/zx/Documents/课程/文章自动评分/essay_data/train.tsv'
dev_dataset_file = '/Users/zx/Documents/课程/文章自动评分/essay_data/dev.tsv'
train_dataset = pd.read_csv(train_dataset_file, delimiter='\t',
                          usecols=['essay_set', 'essay_id', 'domain1_score'])
dev_dataset = pd.read_csv(dev_dataset_file, delimiter='\t',
                          usecols=['essay_set', 'essay_id', 'domain1_score'])

k_s = []
dev_true_s = []
train_true_s = []
for i in range(1, 9):
    # if i in [3, 4, 5, 6]:
    #     continue
    dev_in_set = dev_dataset[dev_dataset.essay_set == i]
    train_in_set = train_dataset[train_dataset.essay_set == i]
    # print(dev_in_set.domain1_score.values)

    dev_true = dev_in_set.domain1_score.values
    dev_true = [(v - score_ranges[i-1][0]) / (score_ranges[i-1][1] - score_ranges[i-1][0]) for v in dev_true]
    print(np.average(dev_true))
    dev_true_s.append(dev_true)

    train_true = train_in_set.domain1_score.values
    train_true = [(v - score_ranges[i-1][0]) / (score_ranges[i-1][1] - score_ranges[i-1][0]) for v in train_true]
    print(np.average(train_true))
    train_true_s.append(train_true)

    print()


for i in range(0, 8):
    selected = [j for j in range(0, 8) if i != j]
    all_selected = []
    for j in range(1, 8):
        all_selected.extend(list(itertools.combinations(selected, j)))
    # print(all_selected)
    # print(len(all_selected))
    all_selected_distance = []
    for selected in all_selected:
        current_set_label = train_true_s[i]
        refer_label = []
        for j in selected:
            refer_label.extend(train_true_s[j])
        distance = wasserstein_distance(current_set_label, refer_label)
        all_selected_distance.append(distance)
        # print([element + 1 for element in selected], distance)

    min_selected = all_selected[all_selected_distance.index(min(all_selected_distance))]
    min_selected = [element + 1 for element in min_selected]
    print(min_selected, min(all_selected_distance))
    # print(selected)


# for dev_true in dev_true_s:
#     distances = []
#     for train_true in train_true_s:
#         distances.append(wasserstein_distance(dev_true, train_true))
#     print(distances)