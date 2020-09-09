from nltk import tokenize
import numpy as np
from stanfordcorenlp import StanfordCoreNLP
from nltk.tree import Tree
nlp = StanfordCoreNLP(r'../stanford_corenlp_full', lang='en', memory='8g')


def mean_variance_sent_length(dataset: list):

    mean_len_list = []
    var_len_list = []

    for sample in dataset:
        sentences = sample['essay_sent']
        sentences_len = [len(sentence.split(' ')) for sentence in sentences]
        mean_len = np.average(sentences_len)
        var_len = np.var(sentences_len)
        mean_len_list.append(mean_len)
        var_len_list.append(var_len)
        sample['sent_avg_len'] = mean_len
        sample['sent_var_len'] = var_len

    return {'sent_avg_len': {'mean': np.mean(mean_len_list), 'std': np.std(mean_len_list)},
            'sent_var_len': {'mean': np.mean(var_len_list), 'std': np.std(var_len_list)}}


def sent_count(dataset: list):
    count_list = []
    for sample in dataset:
        sentence_count = len(sample['essay_sent'])
        sample['sent_count'] = sentence_count
        count_list.append(sentence_count)

    return {'sent_count': {'mean': np.mean(count_list), 'std': np.std(count_list)}}


def mean_clause_length(dataset: list):
    pass


def mean_clause_number(dataset: list):
    pass


def mean_sentence_depth(dataset: list):
    pass


def mean_sentence_level(dataset: list):
    pass


def means_of_parsing_trees(dataset: list):

    mean_clause_num_list = []
    mean_clause_length_list = []
    mean_sent_depth_list = []
    mean_sent_level_list = []

    count = 0
    for sample in dataset:
        print(count)
        count += 1

        mean_clause_num = 0.0
        mean_clause_len = 0.0
        mean_sen_dep = 0.0
        mean_sen_lev = 0.0

        for i, sentence in enumerate(sample['essay_sent']):  # 对每句话分别做检查并更新
            # print(sentence)
            if len(sentence.split(' ')) >= 60:
                clause_count = 0
                clause_len = 0
                sen_dep = 0
                sen_lev = 0
            else:
                tree = Tree.fromstring(nlp.parse(sentence))
                clause_count, clause_len = clause_count_len_of_sen(tree)
                clause_len = clause_len / clause_count if clause_count != 0 else clause_len
                sen_dep = depth_of_sen(tree)
                sen_lev = tree.height()

            mean_clause_num = (mean_clause_num * i + clause_count) / (i + 1)
            mean_clause_len = (mean_clause_len * i + clause_len) / (i + 1)
            mean_sen_dep = (mean_sen_dep * i + sen_dep) / (i + 1)
            mean_sen_lev = (mean_sen_lev * i + sen_lev) / (i + 1)

        sample['mean_clause_number'] = mean_clause_num
        sample['mean_clause_length'] = mean_clause_len
        sample['mean_sent_depth'] = mean_sen_dep
        sample['mean_sent_level'] = mean_sen_lev

        mean_clause_num_list.append(mean_clause_num)
        mean_clause_length_list.append(mean_clause_len)
        mean_sent_depth_list.append(mean_sen_dep)
        mean_sent_level_list.append(mean_sen_lev)

    # nlp.close()
    return {
        'mean_clause_number': {'mean': np.mean(mean_clause_num_list), 'std': np.std(mean_clause_num_list)},
        'mean_clause_length': {'mean': np.mean(mean_clause_length_list), 'std': np.std(mean_clause_length_list)},
        'mean_sent_depth': {'mean': np.mean(mean_sent_depth_list), 'std': np.std(mean_sent_depth_list)},
        'mean_sent_level': {'mean': np.mean(mean_sent_level_list), 'std': np.std(mean_sent_level_list)}
    }


def clause_count_len_of_sen(tree):
    clause_count = 0
    clause_len = 0
    for child in tree:
        if isinstance(child, Tree):
            count_of_children, len_of_children = clause_count_len_of_sen(child)

            if child.label() == 'SBAR':   ## 这里只分析了'SBAR'
                clause_count += 1
                clause_len += len(child.leaves()) - len_of_children

            clause_count += count_of_children
            clause_len += len_of_children

    return clause_count, clause_len


def depth_of_sen(tree, dep=0):
    depth_count = 0
    for child in tree:
        if isinstance(child, Tree):
            depth_count += depth_of_sen(child, dep + 1)
        else:
            depth_count += dep + 1
    return depth_count


