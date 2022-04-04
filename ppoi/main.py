# -*- coding: utf-8 -*-
import numpy as np
from sklearn.linear_model import LogisticRegression
import os
from .user import make_features as _make_features
from time import perf_counter
from random import random

_HERE = os.path.dirname(os.path.abspath(__file__))
_POSITIVE = os.path.join(_HERE, "positive.txt")
_NEGATIVE = os.path.join(_HERE, "negative.txt")
_NEUTRAL = os.path.join(_HERE, "neutral.txt")
_UNKNOWN = os.path.join(_HERE, "unknown.txt")
_model = None


def _take_examples(message):
    examples = []
    print(message)
    while True:
        line = input("> ")
        if not line and examples:
            return examples
        examples.append(line)


def _initialize():
    positives = _take_examples(
        "Enter at least one positive examples (empty to exit)")
    negatives = _take_examples(
        "Enter at least one negative examples (empty to exit)")
    fo = open(_POSITIVE, "w")
    fo.writelines([line + "\n" for line in positives])
    fo.close()
    fo = open(_NEGATIVE, "w")
    fo.writelines([line + "\n" for line in negatives])
    fo.close()
    fo = open(_NEUTRAL, "w")
    fo.close()
    _learn()
    _describe()


def _make_training_data():
    pos_x = open(_POSITIVE).readlines()
    neg_x = open(_NEGATIVE).readlines()
    X = []
    N = len(pos_x) + len(neg_x)
    Y = np.zeros((N, ))
    _last_learn_info["num_positive"] = len(pos_x)
    _last_learn_info["num_negative"] = len(neg_x)

    for i, line in enumerate(pos_x):
        X.append(_make_features(line))
        Y[i] = 1

    offset = len(pos_x)
    for i_, line in enumerate(neg_x):
        i = offset + i_
        X.append(_make_features(line))
        Y[i] = 0

    X = np.array(X)
    return X, Y


def _learn():
    global _model, _last_learn_info
    _last_learn_info = {}
    _model = LogisticRegression()
    X, Y = _make_training_data()
    _model.fit(X, Y)


def _interactive():
    down_sampling = 1.0
    while True:
        _learn()
        start_time = perf_counter()
        scored_lines = _get_scored_lines(down_sampling)
        scored_lines.sort(key=lambda x: abs(x[0] - 0.5))
        # 約1秒で結果が返るようにダウンサンプリングする
        down_sampling = (perf_counter() - start_time) * down_sampling

        score, line = scored_lines[0]
        print("{}: {:.4f}".format(line, score))
        while True:
            s = input("negative(z), neutral(x), positive(c), quit(q)>")
            if s == "q":
                return
            if s == "z":
                open(_NEGATIVE, "a").write(line + "\n")
                break
            if s == "x":
                open(_NEUTRAL, "a").write(line + "\n")
                break
            if s == "c":
                open(_POSITIVE, "a").write(line + "\n")
                break


def _find():
    query = input("input query> ")
    pos = open(_POSITIVE).readlines()
    neg = open(_NEGATIVE).readlines()
    neu = open(_NEUTRAL).readlines()
    unknowns = open(_UNKNOWN).readlines()

    for line in unknowns:
        if query not in line: continue
        if line in pos: continue
        if line in neg: continue
        if line in neu: continue
        line = line.rstrip()
        print(line)
        while True:
            s = input("negative(z), neutral(x), positive(c), quit(q)>")
            if s == "q":
                return
            if s == "z":
                open(_NEGATIVE, "a").write(line + "\n")
                break
            if s == "x":
                open(_NEUTRAL, "a").write(line + "\n")
                break
            if s == "c":
                open(_POSITIVE, "a").write(line + "\n")
                break


def _get_scored_lines(down_sampling=1.0):
    pos = open(_POSITIVE).readlines()
    neg = open(_NEGATIVE).readlines()
    neu = open(_NEUTRAL).readlines()
    unknowns = open(_UNKNOWN).readlines()
    if down_sampling > 1.0:
        p = 1.0 / down_sampling
        unknowns = [line for line in unknowns if random() < p]

    X = []
    used_lines = []
    for line in unknowns:
        if line in pos: continue
        if line in neg: continue
        if line in neu: continue
        line = line.rstrip()
        X.append(_make_features(line))
        used_lines.append(line)

    if len(X) == 0:
        raise RuntimeError("Error: unknown.txtに教師データに含まれないデータがない")

    X = np.array(X)
    probs = _model.predict_proba(X)
    scored_lines = []
    for i in range(len(used_lines)):
        scored_lines.append((probs[i, 1], used_lines[i]))

    return scored_lines


def _describe():
    scored_lines = _get_scored_lines()
    scored_lines.sort()
    print("BEST 5")
    for score, line in scored_lines[-1:-6:-1]:
        print("{}: {:.4f}".format(line, score))

    print("\nWORST 5")
    for score, line in scored_lines[:5]:
        print("{}: {:.4f}".format(line, score))

    print("\nLESS CONFIDENT")
    scored_lines.sort(key=lambda x: abs(x[0] - 0.5))
    for score, line in scored_lines[:5]:
        print("{}: {:.4f}".format(line, score))


def ppoi(line):
    return to_bool(line)


def to_prob(line):
    if not _model:
        _learn()

    return _model.predict_proba(_make_features(line).reshape(1, -1))[0, 1]


def to_bool(line):
    prob = to_prob(line)
    return (prob > 0.5)



