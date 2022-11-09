import os
import re
import glob
import string
import subprocess

from collections import OrderedDict

import numpy as np
import pandas as pd

from pathlib import Path
from nltk import corpus
from sklearn.model_selection import train_test_split

from src.utils import load_txt


SEED = 123

PUNCTUATION = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + \
    '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
SIMPLE_PUNCTUATION = "'.,!?():" + '"'

SPECIAL_CHARS_MAP = {
    "\ufeff" : "", "\u200b": " ", "&#39;": "'",
    "&gt": " ", "&lt": " ", "&amp": " "
}

ENG_WORDS = set(corpus.words.words())
STOPWORDS = set(corpus.stopwords.words('english'))

def is_word(word):
    return all(c in string.ascii_lowercase for c in word)


def generate_unigram(text):
    text = text.strip().lower()
    for k, v in SPECIAL_CHARS_MAP.items():
        text = text.replace(k, v)
    for p in SIMPLE_PUNCTUATION:
        text = text.replace(p, f" {p} ")
    text = re.sub("\s+", " ", text).strip()
    return " ".join([w for w in text.split(" ")
                     if is_word(w) and len(w) > 2 and \
                        w in ENG_WORDS and \
                        w not in STOPWORDS])


def load_youtube_spam_dataset(data_path, transform_unigrams=False):
    filenames = sorted(glob.glob(f"{data_path}/Youtube*.csv"))

    dfs = []
    for i, filename in enumerate(filenames, start=1):
        df = pd.read_csv(filename)
        df.columns = map(str.lower, df.columns)
        df = df.drop("comment_id", axis=1)
        df["video"] = [i] * len(df)
        df = df.rename(columns={"class": "label", "content": "text"})
        df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
        if transform_unigrams:
            df["text"] = df["text"].apply(generate_unigram)
        dfs.append(df)

    df_train = pd.concat(dfs[:4])
    df_train = df_train.sample(frac=1, random_state=SEED)

    df_valid_test = dfs[4]
    df_val, df_test = train_test_split(
        df_valid_test, test_size=195, random_state=SEED, stratify=df_valid_test.label
    )

    return df_train, df_val, df_test


def load_imdb_review_dataset(data_path, transform_unigrams=False):
    data_path = Path(data_path)
    pos_dir, neg_dir = data_path / "pos", data_path / "neg"

    data = {"text": [], "label": []}
    for d, l in zip([neg_dir, pos_dir], [0, 1]):
        files = sorted(os.listdir(d))
        for f in files:
            text = " ".join(load_txt(d / f))
            data["text"].append(text)
            data["label"].append(l)

    df = pd.DataFrame.from_dict(data)
    if transform_unigrams:
        df["text"] = df["text"].apply(generate_unigram)

    df_train, df_val_test = train_test_split(
        df, test_size=5000, random_state=SEED, stratify=df.label
    )
    df_val, df_test = train_test_split(
        df_val_test, test_size=2500, random_state=SEED, stratify=df_val_test.label
    )
    return df_train.reset_index(drop=True), df_val, df_test


def preview_tfs(df, tfs):
    transformed_examples = []
    for f in tfs:
        for _, row in df.sample(frac=1, random_state=2).iterrows():
            transformed_or_none = f(row)
            if transformed_or_none is not None:
                transformed_examples.append(
                    OrderedDict(
                        {
                            "TF Name": f.name,
                            "Original Text": row.text,
                            "Transformed Text": transformed_or_none.text,
                        }
                    )
                )
                break
    return pd.DataFrame(transformed_examples)
