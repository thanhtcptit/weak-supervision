import glob

import numpy as np
import pandas as pd

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.utils import shuffle
from sklearn import model_selection as cross_validation
from sklearn.feature_extraction.text import CountVectorizer


YOUTUBE_DATASET_DIR = "/home/s2210421/dataset/youtube_spam_cmt/data"


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]


def load_youtube_dataset():
    filenames = sorted(glob.glob(f"{YOUTUBE_DATASET_DIR}/Youtube*.csv"))

    dfs = []
    for i, filename in enumerate(filenames, start=1):
        df = pd.read_csv(filename)
        df.columns = map(str.lower, df.columns)
        df = df.drop("comment_id", axis=1)
        df["video"] = [i] * len(df)
        df = df.rename(columns={"class": "label", "content": "text"})
        df = df.sample(frac=1, random_state=123).reset_index(drop=True)
        dfs.append(df)

    df_train = pd.concat(dfs[:4])
    labels = np.asarray(df_train['label'])
    labels[np.where(labels == 0)] = -1
    plots, labels = np.asarray(df_train['text']), labels

    return plots, labels


def split_data(X, plots, y, split_val=0.1):
    np.random.seed(1234)
    num_test = 500
    X, plots, y = shuffle(X, plots, y, random_state=25)

    X_test = X[0:num_test, :]
    X_train = X[num_test:, :]
    plots_train = plots[num_test:]
    plots_test = plots[0:num_test]

    y_test = y[0:num_test]
    y_train = y[num_test:]

    test_ratio = split_val
    X_tr, X_te, y_tr, y_te, plots_tr, plots_te = cross_validation.train_test_split(
        X_train, y_train, plots_train, test_size=test_ratio, random_state=25)

    print('Train ', len(y_tr), 'Valid ', len(y_te))
    return np.array(X_tr.todense()), np.array(X_te.todense()), np.array(X_test.todense()), \
        np.array(y_tr), np.array(y_te), np.array(y_test), plots_tr, plots_te, plots_test


class DataLoader(object):
    def prune_features(self, val_primitive_matrix, train_primitive_matrix, thresh=0.01):
        val_sum = np.sum(np.abs(val_primitive_matrix), axis=0)
        train_sum = np.sum(np.abs(train_primitive_matrix), axis=0)

        train_idx = np.where(
            (train_sum >= thresh*np.shape(train_primitive_matrix)[0]))[0]
        val_idx = np.where(
            (val_sum >= thresh*np.shape(val_primitive_matrix)[0]))[0]
        common_idx = list(set(train_idx) & set(val_idx))

        return common_idx

    def load_data(self, dataset, data_path='', split_val=0.1, feat='count'):
        plots, labels = load_youtube_dataset()

        if feat == 'count':
            vectorizer = CountVectorizer(min_df=1, binary=True, stop_words='english',
                                         decode_error='ignore', strip_accents='ascii', ngram_range=(1, 2))
        elif feat == 'lemma':
            vectorizer = CountVectorizer(min_df=1, binary=True,   decode_error='ignore', ngram_range=(1, 2),
                                         tokenizer=LemmaTokenizer(), strip_accents='unicode', stop_words='english',
                                         lowercase=True)
        else:
            vectorizer = CountVectorizer(min_df=1, binary=True, stop_words='english',
                                         decode_error='ignore', strip_accents='ascii', ngram_range=(1, 2))

        X = vectorizer.fit_transform(plots)
        valid_feats = np.where(np.sum(X, 0) > 2)[1]
        X = X[:, valid_feats]

        train_primitive_matrix, val_primitive_matrix, test_primitive_matrix, train_ground, val_ground, \
            test_ground, train_plots, val_plots, test_plots = split_data(X, plots, labels, split_val)
        common_idx = self.prune_features(val_primitive_matrix, train_primitive_matrix)

        return train_primitive_matrix[:, common_idx], val_primitive_matrix[:, common_idx], \
            test_primitive_matrix[:, common_idx], np.array(train_ground), np.array(val_ground), \
            np.array(test_ground), vectorizer, valid_feats, common_idx, train_plots, val_plots, test_plots
