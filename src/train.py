import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import copy
import shutil
import pickle
import collections

import numpy as np
import pandas as pd
pd.set_option("display.max_colwidth", 0)

from tqdm import tqdm
from pathlib import Path

from snorkel.labeling import LabelingFunction, PandasLFApplier, LFAnalysis, \
    filter_unlabeled_dataframe
from snorkel.labeling.model import LabelModel
from snorkel.utils import probs_to_preds

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.utils import Params
from src.utils.logger import Logger
from src.data.utils import load_youtube_spam_dataset


def keyword_lookup(x, keywords, label):
    if any(word in x.text.lower() for word in keywords):
        return label
    return -1


def make_keyword_lf(keywords, label):
    return LabelingFunction(
        name=f"keyword_{keywords[0]}",
        f=keyword_lookup,
        resources=dict(keywords=keywords, label=label))


def select_dev_example(df, lfs=[], method="random"):
    retry_count = 0
    while True:
        example = df.sample(n=1).iloc[0]
        if len(lfs) and retry_count <= 5:
            if method == "abs":
                has_label = False
                for lf in lfs:
                    if lf(example) != -1:
                        has_label = True
                        break
                if has_label:
                    retry_count += 1
                    continue
            elif method == "dis":
                labels = []
                for lf in lfs:
                    l = lf(example)
                    if l != -1:
                        labels.append(l)
                if len(labels) != 0:
                    conf = np.mean(labels)
                    if conf < 0.4 or conf > 0.7:
                        retry_count += 1
                        continue

        return example


def select_primitive(example, prim_set, prim_freq, prim_thresh, use_weight_probs=False):
    words = example.text.lower().split()
    cands, cands_w = [], []
    for w in words:
        if w not in prim_freq:
            continue
        if (w, example.label) in prim_set:
            continue

        if (example.label == 0 and prim_freq[w] < 1 - prim_thresh) or \
           (example.label == 1 and prim_freq[w] > prim_thresh):
            cands.append(w)
            if example.label == 1:
                cands_w.append(prim_freq[w])
            else:
                cands_w.append(1 - prim_freq[w])
    if len(cands) == 0:
        return None
    cands_prob = np.exp(cands_w) / np.sum(np.exp(cands_w), axis=0)
    if use_weight_probs:
        prim = np.random.choice(cands, size=1, p=cands_prob)[0]
    else:
        prim = np.random.choice(cands, size=1)[0]
    return prim


def build_primitive_lf(df, lfs, select_method, prim_set, prim_freq, prim_thres,
                       use_weight_probs=False):
    prim, example = None, None
    while prim is None:
        example = select_dev_example(df, lfs, select_method)
        prim = select_primitive(example, prim_set, prim_freq, prim_thres, use_weight_probs)
    return make_keyword_lf(keywords=[prim], label=example.label), (prim, example.label)


def eval_lm_em(df_train, df_test, lfs):
    applier = PandasLFApplier(lfs=lfs)
    L_train = applier.apply(df=df_train, progress_bar=False)
    L_test = applier.apply(df=df_test, progress_bar=False)

    label_model = LabelModel(cardinality=2, verbose=False)
    label_model.fit(L_train=L_train, n_epochs=500, progress_bar=False)
    lm_acc = label_model.score(
        L=L_test, Y=df_test.label.values, tie_break_policy="random")["accuracy"] * 100

    prob_labels = label_model.predict_proba(L=L_train)
    df_train_filtered, prob_labels_filtered = filter_unlabeled_dataframe(
        X=df_train, y=prob_labels, L=L_train)

    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(df_train_filtered.text.tolist())
    X_test = vectorizer.transform(df_test.text.tolist())

    pred_labels_filtered = probs_to_preds(probs=prob_labels_filtered)
    end_model = LogisticRegression(C=1e3, solver="liblinear")
    end_model.fit(X=X_train, y=pred_labels_filtered)
    em_acc = end_model.score(X=X_test, y=df_test.label.values) * 100

    lf_analysis = LFAnalysis(L=L_train, lfs=lfs)
    return lm_acc, em_acc, label_model, end_model, lf_analysis


def train(config_path, save_dir=None, recover=False, force=False, verbose=False):
    config = Params.from_file(config_path)
    if save_dir:
        if os.path.exists(save_dir):
            if force:
                shutil.rmtree(save_dir)
            else:
                raise ValueError(save_dir + " existed & force = False")

        os.makedirs(save_dir)
        save_dir = Path(save_dir)
        log_path = save_dir / "log"
    else:
        log_path = ""

    logger = Logger(log_path, stdout=verbose)

    if config["dataset"] == "ytb":
        df_train, df_val, df_test = load_youtube_spam_dataset(
            Path(config["dataset_path"]) / "data", transform_unigrams=True)
    elif config["dataset"] == "imbd":
        df_train, df_val, df_test = None, None, None
    else:
        raise ValueError(config["dataset"])

    logger.log(f"Train: {len(df_train)} - Val: {len(df_val)} - Test: {len(df_test)}")

    df_train_spam = df_train[df_train.label == 1]
    df_train_ham = df_train[df_train.label == 0]

    unigrams = collections.defaultdict(list)
    for i, row in df_train.iterrows():
        words = list(set(row["text"].split()))
        for w in words:
            unigrams[w].append(row["label"])

    unigrams_freq = {}
    for w, l in unigrams.items():
        unigrams_freq[w] = sum(l) / len(l)

    avg_acc = []
    best_em_acc = 0
    best_lfs, best_lm, best_em = None, None, None

    lfs, primitive_set = [], set()
    use_weight_probs = False

    segment_df = None
    segment_flag = False

    for iter in tqdm(range(1, config["num_iter"] + 1)):
        if iter == 1:
            for _ in range(3):
                if config["alternative_draw"]:
                    segment_df = df_train_spam if segment_flag else df_train_ham
                    segment_flag = not segment_flag
                else:
                    segment_df = df_train
                lf, prim = build_primitive_lf(segment_df, lfs, config["select_method"], primitive_set, unigrams_freq,
                                              config["primitive_threshold"], use_weight_probs)
                lfs.append(lf)
                primitive_set.add(prim)
        else:
            if config["alternative_draw"]:
                segment_df = df_train_spam if segment_flag else df_train_ham
                segment_flag = not segment_flag
            else:
                segment_df = df_train
            lf, prim = build_primitive_lf(segment_df, lfs, config["select_method"], primitive_set, unigrams_freq,
                                          config["primitive_threshold"], use_weight_probs)
            lfs.append(lf)
            primitive_set.add(prim)

        if iter % config["eval_iter_mod"] == 0:
            try:
                _, em_acc, lm, em, _ = eval_lm_em(df_train, df_val, lfs)
            except ValueError:
                em_acc = 0
            if em_acc > best_em_acc:
                best_em_acc = em_acc
                best_lfs = copy.deepcopy(lfs)
                best_lm = lm
                best_em = em
            avg_acc.append(em_acc)

    final_lm_acc, final_em_acc, _, _, lf_analysis = eval_lm_em(df_train, df_test, best_lfs)
    logger.log(f"LM Acc: {final_lm_acc}\nEM Acc: {final_em_acc}\n{lf_analysis.lf_summary()}")

    if save_dir:
        config.to_file(save_dir / "config.json")
        with open(save_dir / "lm.pkl", "wb") as f:
            pickle.dump(best_lm, f)
        with open(save_dir / "em.pkl", "wb") as f:
            pickle.dump(best_em, f)
    return final_lm_acc, final_em_acc


def eval(config_path):
    lm_acc, em_acc = train(config_path)
    print(f"{lm_acc:.4f}\n{em_acc:.4f}\n")
