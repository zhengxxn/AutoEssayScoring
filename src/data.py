import pandas as pd
import spacy
from tqdm import tqdm
import pickle
from nltk import tokenize


class Dataset:

    def __init__(self):
        self.data = {}
        self.normalize_dict = {}

    def load_from_raw_file(self, filename, field_require):
        nlp = spacy.load("en_core_web_sm")

        data = pd.read_csv(filename, delimiter='\t')
        essay_set = set(data['essay_set'])

        for set_id in tqdm(essay_set):

            set_df = data[data.essay_set == set_id]
            fields = [set_df[field] for field in field_require]

            self.normalize_dict[str(set_id)] = {}
            self.data[str(set_id)] = []

            for values in tqdm(zip(*fields), total=len(fields[0])):
                sample_dict = {}
                for i, field in (enumerate(field_require)):
                    if field == 'essay':
                        doc = nlp(values[i])

                        tokens = [token.text for token in doc]
                        sample_dict[field + '_token'] = tokens

                        tokens_pos = [token.pos_ for token in doc]
                        sample_dict[field + '_pos'] = tokens_pos

                        tokens_lemma = [token.lemma_ for token in doc]
                        sample_dict[field+'_lemma'] = tokens_lemma

                        tokens_stop = [token.is_stop for token in doc]
                        sample_dict[field+'_is_stop'] = tokens_stop

                    sample_dict[field] = values[i]

                self.data[str(set_id)].append(sample_dict)

    def normalize_feature(self, set_id, field, normalize_dict=None):
        """
        we change it to standardize
        :param set_id:
        :param field:
        :param normalize_dict:
        :return:
        """
        if normalize_dict is None:
            normalize_dict = self.normalize_dict[set_id]

        # min_value = normalize_dict[field]['min']
        # max_value = normalize_dict[field]['max']
        mean_value = normalize_dict[field]['mean']
        std_value = normalize_dict[field]['std']
        for sample in self.data[set_id]:
            # sample[field] = (sample[field] - min_value) / (max_value - min_value)
            sample[field] = (sample[field] - mean_value) / std_value

    @staticmethod
    def save(dataset, path):
        with open(path, 'wb') as f:

            pickle.dump(dataset, f)

    @staticmethod
    def save_feature(dataset, path):
        with open(path, 'wb') as f:
            for i in range(1, 9):
                for sample in dataset.data[str(i)]:
                    if 'essay' in sample.keys():
                        del sample['essay']
                    if 'essay_token' in sample.keys():
                        del sample['essay_token']
                    if 'essay_pos' in sample.keys():
                        del sample['essay_pos']
                    if 'essay_sent' in sample.keys():
                        del sample['essay_sent']
                    if 'essay_lemma' in sample.keys():
                        del sample['essay_lemma']
                    if 'essay_is_stop' in sample.keys():
                        del sample['essay_is_stop']

            pickle.dump(dataset, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            dataset = pickle.load(f)
        return dataset


def split_sentence(dataset):
    for i in range(1, 9):
        for sample in dataset.data[str(i)]:
            sentences = tokenize.sent_tokenize(sample['essay'])
            sample['essay_sent'] = sentences

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
#
# test_dataset = Dataset()
# test_dataset.load_from_raw_file('../data/essay_data/test.tsv', ['essay_set', 'essay_id', 'essay'])
# Dataset.save(test_dataset, '../data/essay_data/test.p')


# train_dataset = Dataset.load("../data/essay_data/train.p")
# dev_dataset = Dataset.load("../data/essay_data/dev.p")
# test_dataset = Dataset.load("../data/essay_data/test.p")

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