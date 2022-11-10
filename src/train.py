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

from src.utils import Params, get_basename
from src.utils.logger import Logger
from src.data.utils import load_youtube_spam_dataset, load_imdb_review_dataset


def keyword_lookup(x, keywords, label):
    if any(word in x.text.lower() for word in keywords):
        return label
    return -1


def make_keyword_lf(keywords, label):
    return LabelingFunction(
        name=f"keyword_{keywords[0]}",
        f=keyword_lookup,
        resources=dict(keywords=keywords, label=label))


def seu(df, lfs, prim_examples, selected_prim, label_list):
    applier = PandasLFApplier(lfs=lfs)
    L_train = applier.apply(df=df, progress_bar=False)

    label_model = LabelModel(cardinality=2, verbose=False)
    label_model.fit(L_train=L_train, n_epochs=500, progress_bar=False)
    prob_labels = label_model.predict_proba(L=L_train)

    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(df.text.tolist())
    end_model = LogisticRegression(C=1e3, solver="liblinear")
    end_model.fit(X=X_train, y=probs_to_preds(prob_labels))
    pred_labels = probs_to_preds(end_model.predict_proba(X_train))

    lm_uncer = -np.sum(prob_labels * np.log(prob_labels), axis=-1)
    cand_lfs_acc, cand_lfs_util_score = {}, {}
    for prim, ex_list in prim_examples.items():
        if prim in selected_prim:
            continue

        for l in label_list:
            corr = pred_labels[ex_list] == np.array([l] * len(ex_list))
            cand_lfs_acc[(prim, l)] = np.sum(corr) / corr.shape[0]

            _w = np.where(corr == 1, corr, -1)
            util_score = np.sum(lm_uncer[ex_list] * _w)
            cand_lfs_util_score[(prim, l)] = util_score

    best_ex_score = 0
    best_ex_ind = -1
    for i, row in df.reset_index(drop=True).iterrows():
        words = list(set(row["text"].split()))

        cand_prims = [w for w in words if w in prim_examples and w not in selected_prim]
        if len(cand_prims) == 0:
            continue
        for l in label_list:
            acc = np.array([cand_lfs_acc[(w, l)] for w in cand_prims])
            agg = np.sum(acc)
            p = 0 if agg == 0 else acc / agg
            util_score = np.array([cand_lfs_util_score[(w, l)] for w in cand_prims])
            ex_score = np.sum(p * util_score)
            if ex_score > best_ex_score:
                best_ex_score = ex_score
                best_ex_ind = i

    assert best_ex_ind != -1

    return df.iloc[best_ex_ind], best_ex_ind


def select_dev_example(df, lfs, method, prim_examples, selected_prim, label_list):
    retry_count = 0
    while True:
        if method.startswith("random"):
            index = np.random.randint(len(df))
            example = df.iloc[index]
            if len(lfs) and retry_count <= 5:
                if method == "random_abs":
                    has_label = False
                    for lf in lfs:
                        if lf(example) != -1:
                            has_label = True
                            break
                    if has_label:
                        retry_count += 1
                        continue
                elif method == "random_dis":
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
        elif method == "seu":
            example, index = seu(df, lfs, prim_examples, selected_prim, label_list)

        return example, index


def select_primitive(example, selected_prim, prim_probs, use_weight_probs=False):
    words = example.text.lower().split()
    cands, cands_w = [], []
    for w in words:
        if w not in prim_probs or w in selected_prim:
            continue

        cands.append(w)
        if example.label == 1:
            cands_w.append(prim_probs[w])
        else:
            cands_w.append(1 - prim_probs[w])
    if len(cands) == 0:
        return None

    if use_weight_probs:
        cands_prob = np.exp(cands_w) / np.sum(np.exp(cands_w), axis=0)
        prim = np.random.choice(cands, size=1, p=cands_prob)[0]
    else:
        prim = np.random.choice(cands, size=1)[0]
    return prim


def build_primitive_lf(df, lfs, select_method, selected_prim, prim_probs,
                       prim_examples, label_list, use_weight_probs=False):
    prim, example = None, None
    retry_count = 0
    last_example = None
    while prim is None:
        if retry_count > 5:
            select_method = "random"
        example, index = select_dev_example(df, lfs, select_method, prim_examples, selected_prim,
                                     label_list)
        if last_example is not None and last_example.equals(example):
            select_method = "random"
            continue

        last_example = example
        prim = select_primitive(example, selected_prim, prim_probs, use_weight_probs)
        retry_count += 1
    return make_keyword_lf(keywords=[prim], label=example.label), (prim, index, example.label), retry_count


def filter_L(L, selected_prim, deactive_examples):
    for i, (_, v) in enumerate(selected_prim.items()):
        noisy_ex = deactive_examples[v[0]]
        if len(noisy_ex) != 0:
            L[noisy_ex, [i] * len(noisy_ex)] = -1
    return L


def eval_lm_em(df_train, df_test, lfs, selected_prim, contextual_learning=False,
               deactive_examples=None):
    applier = PandasLFApplier(lfs=lfs)
    L_train = applier.apply(df=df_train, progress_bar=False)
    L_test = applier.apply(df=df_test, progress_bar=False)
    if contextual_learning:
        L_train = filter_L(L_train, selected_prim, deactive_examples)

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

    lm_labels = probs_to_preds(probs=prob_labels_filtered)
    end_model = LogisticRegression(C=1e3, solver="liblinear")
    end_model.fit(X=X_train, y=lm_labels)
    em_acc = end_model.score(X=X_test, y=df_test.label.values) * 100

    lf_analysis = LFAnalysis(L=L_train, lfs=lfs)
    return lm_acc, em_acc, label_model, end_model, lf_analysis


def train(config_path, save_dir=None, recover=False, force=False, verbose=False):
    config = Params.from_file(config_path)
    
    if save_dir is not None:
        if save_dir == "":
            save_dir = os.path.join("train_logs", config["dataset"],
                                    get_basename(config_path, remove_ext=True))
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
            config["dataset_path"], transform_unigrams=True)
    elif config["dataset"] == "imdb":
        df_train, df_val, df_test = load_imdb_review_dataset(
            config["dataset_path"], transform_unigrams=True)
    else:
        raise ValueError(config["dataset"])
    label_list = list(set(df_train["label"]))

    logger.log(f"Train: {len(df_train)} - Val: {len(df_val)} - Test: {len(df_test)}")

    df_train_pos = df_train[df_train.label == 1]
    df_train_neg = df_train[df_train.label == 0]

    primitive_examples, primitive_labels = collections.defaultdict(list), \
        collections.defaultdict(list)
    for i, row in df_train.iterrows():
        words = list(set(row["text"].split()))
        for w in words:
            primitive_examples[w].append(i)
            primitive_labels[w].append(row["label"])

    primitive_probs = {}
    for w in primitive_labels.keys():
        freq = sum(primitive_labels[w]) / len(primitive_labels[w])
        if freq > (1 - config["primitive_threshold"]) and freq < config["primitive_threshold"]:
            continue
        primitive_examples[w] = list(set(primitive_examples[w]))
        primitive_probs[w] = freq
    logger.log(f"Num primitives: {len(primitive_probs)}")

    if "contextual_learning" in config:
        contextual_learning = True
        percentile = config["contextual_learning"]["percentile"]

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(df_train.text.tolist()).toarray()
        X_norm = X / (np.linalg.norm(X, axis=-1) + 1e-5)[:, np.newaxis]
        dist = 1 - X_norm @ X_norm.T
        r = np.percentile(dist, percentile, axis=-1)
        deactive_examples = {i: [j for j in range(len(df_train)) if dist[i, j] > r[i]]
                             for i in range(len(df_train))}
    else:
        contextual_learning = False
        deactive_examples = None

    avg_acc = []
    lm_acc, em_acc, best_em_acc = 0, 0, 0
    best_lfs, best_prim, best_lm, best_em = None, None, None, None

    lfs, selected_prim = [], {}
    use_weight_probs = False

    segment_df = None
    segment_flag = False

    pbar = tqdm(range(1, config["num_iter"] + 1))
    for iter in pbar:
        if iter == 1:
            for _ in range(3):
                if config["alternative_draw"]:
                    segment_df = df_train_pos if segment_flag else df_train_neg
                    segment_flag = not segment_flag
                else:
                    segment_df = df_train
                lf, prim, retry = build_primitive_lf(
                    segment_df, lfs, "random", selected_prim, primitive_probs,
                    primitive_examples, label_list, use_weight_probs)
                lfs.append(lf)
                selected_prim[prim[0]] = (prim[1], prim[2])
        else:
            if config["alternative_draw"]:
                segment_df = df_train_pos if segment_flag else df_train_neg
                segment_flag = not segment_flag
            else:
                segment_df = df_train
            lf, prim, retry = build_primitive_lf(
                segment_df, lfs, config["select_method"], selected_prim, primitive_probs,
                primitive_examples, label_list, use_weight_probs)
            lfs.append(lf)
            selected_prim[prim[0]] = (prim[1], prim[2])

        if iter % config["eval_iter_mod"] == 0:
            try:
                lm_acc, em_acc, lm, em, _ = eval_lm_em(
                    df_train, df_val, lfs, selected_prim, contextual_learning,
                    deactive_examples)
                logger.log(f"Iter {iter} - LM Acc: {lm_acc} - EM Acc: {em_acc}\n")
            except Exception as e:
                logger.log(e)
                em_acc = 0
                continue

            if em_acc > best_em_acc:
                best_em_acc = em_acc
                best_lfs = copy.deepcopy(lfs)
                best_prim = copy.deepcopy(selected_prim)
                best_lm = lm
                best_em = em
            avg_acc.append(em_acc)
        pbar.set_postfix({"prim": prim[0], "retry": retry, "lm_acc": lm_acc, "em_acc": em_acc})

    final_lm_acc, final_em_acc, _, _, lf_analysis = eval_lm_em(
        df_train, df_test, best_lfs, best_prim, contextual_learning,
        deactive_examples)
    logger.log(f"[Test set]\nLM Acc: {final_lm_acc}\nEM Acc: {final_em_acc}\n{lf_analysis.lf_summary()}")

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
