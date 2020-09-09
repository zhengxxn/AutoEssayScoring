import lexical_feature_extract
import content_organization_feature_extract
import syntactic_feature_extract
import grammatical_feature_extract
import readability_feature_extract
import pickle

import numpy as np

from data import Dataset, split_sentence
import pandas as pd


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


def parser():
    for i in range(1, 9):
        normalize_dict = syntactic_feature_extract.means_of_parsing_trees(train_dataset.data[str(i)])
        train_dataset.normalize_dict[str(i)].update(normalize_dict)
        train_dataset.normalize_feature(str(i), 'mean_clause_number')
        train_dataset.normalize_feature(str(i), 'mean_clause_length')
        train_dataset.normalize_feature(str(i), 'mean_sent_depth')
        train_dataset.normalize_feature(str(i), 'mean_sent_level')

        syntactic_feature_extract.means_of_parsing_trees(dev_dataset.data[str(i)])
        dev_dataset.normalize_feature(str(i), 'mean_clause_number', normalize_dict)
        dev_dataset.normalize_feature(str(i), 'mean_clause_length', normalize_dict)
        dev_dataset.normalize_feature(str(i), 'mean_sent_depth', normalize_dict)
        dev_dataset.normalize_feature(str(i), 'mean_sent_level', normalize_dict)

        syntactic_feature_extract.means_of_parsing_trees(test_dataset.data[str(i)])
        test_dataset.normalize_feature(str(i), 'mean_clause_number', normalize_dict)
        test_dataset.normalize_feature(str(i), 'mean_clause_length', normalize_dict)
        test_dataset.normalize_feature(str(i), 'mean_sent_depth', normalize_dict)
        test_dataset.normalize_feature(str(i), 'mean_sent_level', normalize_dict)

        Dataset.save(train_dataset, '../data/essay_data/train-parse-2.p')
        Dataset.save(dev_dataset, '../data/essay_data/dev-parse-2.p')
        Dataset.save(test_dataset, '../data/essay_data/test-parse-2.p')


def word_vector():
    for i in range(1, 9):
        if 'word_vector_similarity' in extract_feature:
            tv, train_matrix, normalize_dict = lexical_feature_extract.word_vector_similarity(
                train_dataset.data[str(i)])
            train_dataset.normalize_dict[str(i)].update(normalize_dict)
            train_dataset.normalize_feature(str(i), 'word_vector_similarity')

            lexical_feature_extract.word_vector_similarity(dev_dataset.data[str(i)], tv, refer_matrix=train_matrix,
                                                           refer_dataset=train_dataset.data[str(i)])
            dev_dataset.normalize_feature(str(i), 'word_vector_similarity', normalize_dict)

            lexical_feature_extract.word_vector_similarity(test_dataset.data[str(i)], tv, refer_matrix=train_matrix,
                                                           refer_dataset=train_dataset.data[str(i)])
            test_dataset.normalize_feature(str(i), 'word_vector_similarity', normalize_dict)


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


def _extract_feature():
    for i in range(1, 9):

        if 'lda_vector' in extract_feature:
            dictionary, lda = content_organization_feature_extract.lda_vector(
                train_dataset.data[str(i)] + dev_dataset.data[str(i)])
            # content_organization_feature_extract.lda_vector(dev_dataset.data[str(i)], dictionary, lda)
            content_organization_feature_extract.lda_vector(test_dataset.data[str(i)], dictionary, lda)

        if 'token_count' in extract_feature:
            # normalize_dict = lexical_feature_extract.token_counts(train_dataset.data[str(i)] + dev_dataset.data[str(i)])
            lexical_feature_extract.token_counts(train_dataset.data[str(i)])
            lexical_feature_extract.token_counts(dev_dataset.data[str(i)])
            lexical_feature_extract.token_counts(test_dataset.data[str(i)])

        # noun verb count
        if 'pos_count' in extract_feature:
            # normalize_dict = lexical_feature_extract.pos_counts(train_dataset.data[str(i)] + dev_dataset.data[str(i)])
            # _ = lexical_feature_extract.pos_counts(test_dataset.data[str(i)])

            lexical_feature_extract.pos_counts(train_dataset.data[str(i)])
            lexical_feature_extract.pos_counts(dev_dataset.data[str(i)])
            lexical_feature_extract.pos_counts(test_dataset.data[str(i)])

        # tf idf similarity
        # lexical_feature_extract.word_vector_similarity(train_dataset.data[str(i)])
        if 'vector_similarity' in extract_feature:
            tv, train_matrix, normalize_dict = lexical_feature_extract.word_vector_similarity(
                train_dataset.data[str(i)] + dev_dataset.data[str(i)])
            # train_dataset.normalize_dict[str(i)].update(normalize_dict)
            # train_dataset.normalize_feature(str(i), 'word_vector_similarity')

            # lexical_feature_extract.word_vector_similarity(dev_dataset.data[str(i)], tv, refer_matrix=train_matrix, refer_dataset=train_dataset.data[str(i)])
            # dev_dataset.normalize_feature(str(i), 'word_vector_similarity', normalize_dict)

            lexical_feature_extract.word_vector_similarity(test_dataset.data[str(i)], tv, refer_matrix=train_matrix,
                                                           refer_dataset=train_dataset.data[str(i)] + dev_dataset.data[
                                                               str(i)])
            # test_dataset.normalize_feature(str(i), 'word_vector_similarity', normalize_dict)

            _, _, normalize_dict = content_organization_feature_extract.semantic_vector_similarity(
                train_dataset.data[str(i)] + dev_dataset.data[str(i)], tv=tv, refer_matrix=train_matrix,
                refer_dataset=train_dataset.data[str(i)] + dev_dataset.data[str(i)])

            # content_organization_feature_extract.semantic_vector_similarity(dev_dataset.data[str(i)], tv, refer_matrix=train_matrix, refer_dataset=train_dataset.data[str(i)])
            # dev_dataset.normalize_feature(str(i), 'semantic_vector_similarity', normalize_dict)

            content_organization_feature_extract.semantic_vector_similarity(test_dataset.data[str(i)], tv,
                                                                            refer_matrix=train_matrix,
                                                                            refer_dataset=train_dataset.data[str(i)] +
                                                                                          dev_dataset.data[str(i)])
            # test_dataset.normalize_feature(str(i), 'semantic_vector_similarity', normalize_dict)

        if 'text_similarity_score' in extract_feature and i in [3, 4, 5, 6, 7]:
            normalize_dict = content_organization_feature_extract.text_similarity_score(train_dataset.data[str(i)],
                                                                                        train_dataset.data[str(i)])
            # train_dataset.normalize_dict[str(i)].update(normalize_dict)
            # train_dataset.normalize_feature(str(i), 'text_similarity_score')
            #
            # _ = content_organization_feature_extract.text_similarity_score(dev_dataset.data[str(i)], train_dataset.data[str(i)])
            # dev_dataset.normalize_feature(str(i), 'text_similarity_score', normalize_dict)

            _ = content_organization_feature_extract.text_similarity_score(test_dataset.data[str(i)],
                                                                           train_dataset.data[str(i)])
            # test_dataset.normalize_feature(str(i), 'text_similarity_score', normalize_dict)

        if 'first_sentence_score' in extract_feature:
            normalize_dict = content_organization_feature_extract.first_sentence_score(train_dataset.data[str(i)],
                                                                                       train_dataset.data[str(i)])
            train_dataset.normalize_dict[str(i)].update(normalize_dict)
            train_dataset.normalize_feature(str(i), 'first_sentence_score')

            _ = content_organization_feature_extract.first_sentence_score(dev_dataset.data[str(i)],
                                                                          train_dataset.data[str(i)])
            dev_dataset.normalize_feature(str(i), 'first_sentence_score', normalize_dict)

            _ = content_organization_feature_extract.first_sentence_score(test_dataset.data[str(i)],
                                                                          train_dataset.data[str(i)])
            test_dataset.normalize_feature(str(i), 'first_sentence_score', normalize_dict)

        if 'last_sentence_score' in extract_feature:
            normalize_dict = content_organization_feature_extract.last_sentence_score(train_dataset.data[str(i)],
                                                                                      train_dataset.data[str(i)])
            train_dataset.normalize_dict[str(i)].update(normalize_dict)
            train_dataset.normalize_feature(str(i), 'last_sentence_score')

            _ = content_organization_feature_extract.last_sentence_score(dev_dataset.data[str(i)],
                                                                         train_dataset.data[str(i)])
            dev_dataset.normalize_feature(str(i), 'last_sentence_score', normalize_dict)

            _ = content_organization_feature_extract.last_sentence_score(test_dataset.data[str(i)],
                                                                         train_dataset.data[str(i)])
            test_dataset.normalize_feature(str(i), 'last_sentence_score', normalize_dict)

        # mean word len, variance word len -----------
        if 'word_len' in extract_feature:
            # normalize_dict = lexical_feature_extract.mean_variance_word_length(train_dataset.data[str(i)] + dev_dataset.data[str(i)])
            #
            # _ = lexical_feature_extract.mean_variance_word_length(test_dataset.data[str(i)])
            lexical_feature_extract.mean_variance_word_length(train_dataset.data[str(i)])
            lexical_feature_extract.mean_variance_word_length(dev_dataset.data[str(i)])
            lexical_feature_extract.mean_variance_word_length(test_dataset.data[str(i)])

        # spelling errors ----------
        if 'spelling_error' in extract_feature:
            # normalize_dict = lexical_feature_extract.spelling_errors(train_dataset.data[str(i)] + dev_dataset.data[str(i)])
            lexical_feature_extract.spelling_errors(train_dataset.data[str(i)])
            lexical_feature_extract.spelling_errors(dev_dataset.data[str(i)])
            lexical_feature_extract.spelling_errors(test_dataset.data[str(i)])

        if 'mean_word_level' in extract_feature:
            # normalize_dict = lexical_feature_extract.mean_word_level(train_dataset.data[str(i)] + dev_dataset.data[str(i)])
            lexical_feature_extract.mean_word_level(train_dataset.data[str(i)])
            lexical_feature_extract.mean_word_level(dev_dataset.data[str(i)])
            lexical_feature_extract.mean_word_level(test_dataset.data[str(i)])

        # essay_length ----------
        if 'essay_len' in extract_feature:
            # normalize_dict = content_organization_feature_extract.essay_length(train_dataset.data[str(i)] + dev_dataset.data[str(i)])

            content_organization_feature_extract.essay_length(train_dataset.data[str(i)])
            content_organization_feature_extract.essay_length(dev_dataset.data[str(i)])
            content_organization_feature_extract.essay_length(test_dataset.data[str(i)])
            # test_dataset.normalize_feature(str(i), 'essay_length', normalize_dict)

        # mean sent len, variance sent len -----------
        if 'sent_len' in extract_feature:
            syntactic_feature_extract.mean_variance_sent_length(train_dataset.data[str(i)])
            syntactic_feature_extract.mean_variance_sent_length(dev_dataset.data[str(i)])
            syntactic_feature_extract.mean_variance_sent_length(test_dataset.data[str(i)])

        if 'sent_count' in extract_feature:
            # normalize_dict = syntactic_feature_extract.sent_count(train_dataset.data[str(i)] + dev_dataset.data[str(i)])

            syntactic_feature_extract.sent_count(train_dataset.data[str(i)])
            syntactic_feature_extract.sent_count(dev_dataset.data[str(i)])
            syntactic_feature_extract.sent_count(test_dataset.data[str(i)])
            # test_dataset.normalize_feature(str(i), 'sent_count', normalize_dict)

        if 'sent_repeat_rate' in extract_feature:
            # normalize_dict = content_organization_feature_extract.sent_repeat_rate(train_dataset.data[str(i)] + dev_dataset.data[str(i)])

            content_organization_feature_extract.sent_repeat_rate(train_dataset.data[str(i)])
            content_organization_feature_extract.sent_repeat_rate(dev_dataset.data[str(i)])
            content_organization_feature_extract.sent_repeat_rate(test_dataset.data[str(i)])
            # test_dataset.normalize_feature(str(i), 'sent_repeat_rate', normalize_dict)

        # word bigram -------
        if 'word_bigram' in extract_feature:
            # train_refer_term, normalize_dict = grammatical_feature_extract.word_bigram(train_dataset.data[str(i)] + dev_dataset.data[str(i)], train_dataset.data[str(i)] + dev_dataset.data[str(i)], None)

            #
            train_refer_term, _ = grammatical_feature_extract.word_bigram(train_dataset.data[str(i)],
                                                                          train_dataset.data[str(i)], None)
            grammatical_feature_extract.word_bigram(dev_dataset.data[str(i)], None, refer_term=train_refer_term)
            grammatical_feature_extract.word_bigram(test_dataset.data[str(i)], None, refer_term=train_refer_term)
            # test_dataset.normalize_feature(str(i), 'word_bigram', normalize_dict)

        # word trigram -------
        if 'word_trigram' in extract_feature:
            # train_refer_term, normalize_dict = grammatical_feature_extract.word_trigram(train_dataset.data[str(i)] + dev_dataset.data[str(i)], train_dataset.data[str(i)] + dev_dataset.data[str(i)], None)

            train_refer_term, _ = grammatical_feature_extract.word_trigram(train_dataset.data[str(i)],
                                                                           train_dataset.data[str(i)], None)
            grammatical_feature_extract.word_trigram(dev_dataset.data[str(i)], None, refer_term=train_refer_term)
            grammatical_feature_extract.word_trigram(test_dataset.data[str(i)], None, refer_term=train_refer_term)

        # pos bigram --------
        if 'pos_bigram' in extract_feature:
            # train_refer_term, normalize_dict = grammatical_feature_extract.pos_bigram(train_dataset.data[str(i)] + dev_dataset.data[str(i)], train_dataset.data[str(i)] + dev_dataset.data[str(i)], None)

            train_refer_term, _ = grammatical_feature_extract.pos_bigram(train_dataset.data[str(i)],
                                                                         train_dataset.data[str(i)], None)
            grammatical_feature_extract.pos_bigram(dev_dataset.data[str(i)], None, refer_term=train_refer_term)
            grammatical_feature_extract.pos_bigram(test_dataset.data[str(i)], None, refer_term=train_refer_term)

        # pos trigram --------
        if 'pos_trigram' in extract_feature:
            # train_refer_term, normalize_dict = grammatical_feature_extract.pos_trigram(train_dataset.data[str(i)] + dev_dataset.data[str(i)], train_dataset.data[str(i)] + dev_dataset.data[str(i)], None)

            train_refer_term, _ = grammatical_feature_extract.pos_trigram(train_dataset.data[str(i)],
                                                                          train_dataset.data[str(i)], None)
            grammatical_feature_extract.pos_trigram(dev_dataset.data[str(i)], None, refer_term=train_refer_term)
            grammatical_feature_extract.pos_trigram(test_dataset.data[str(i)], None, refer_term=train_refer_term)

        if 'pos_bigram_vector' in extract_feature:
            vectorizer, vector = grammatical_feature_extract.pos_bigram_vector(
                train_dataset.data[str(i)] + dev_dataset.data[str(i)],
                train_dataset.data[str(i)] + dev_dataset.data[str(i)], None)
            # grammatical_feature_extract.pos_bigram_vector(dev_dataset.data[str(i)], None, refer_vectorizer=vectorizer, refer_vector=vector)
            grammatical_feature_extract.pos_bigram_vector(test_dataset.data[str(i)], None, refer_vectorizer=vectorizer,
                                                          refer_vector=vector)

        if 'readability' in extract_feature:
            print('preprocess readability')
            readability_feature_extract.readability_attributes(train_dataset.data[str(i)])
            readability_feature_extract.readability_attributes(dev_dataset.data[str(i)])
            readability_feature_extract.readability_attributes(test_dataset.data[str(i)])


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


train_dataset = Dataset.load("../../data/essay_data/train-entire.p")
dev_dataset = Dataset.load("../../data/essay_data/dev-entire.p")
test_dataset = Dataset.load("../../data/essay_data/test-entire.p")


_extract_feature()

Dataset.save_feature(train_dataset, '../../data/essay_data/train-feature-5.p')
Dataset.save_feature(dev_dataset, '../../data/essay_data/dev-feature-5.p')
Dataset.save_feature(test_dataset, '../../data/essay_data/test-feature-5.p')
