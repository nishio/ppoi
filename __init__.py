# -*- coding: utf-8 -*-
import numpy as np
from sklearn.linear_model import LogisticRegression
import argparse
import os

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


def _make_features(s):
    "take a string, return np.array"
    BEST100 = (" のーンenoia0スるrトtルい1イクとアな,をリsにはッ2シc"
               "的学知ラlテジプすpしhタグデdてィ.か・たm化3ロフ理でレ"
               "ドu生がニ性人4マコSムg95法きりエョ識yP分76-8bれミ自経えT論く")
    s = set(s)
    return np.array([(1 if c in s else 0) for c in BEST100])


def _initialize():
    positives = _take_examples("Enter at least one positive examples (empty to exit)")
    negatives = _take_examples("Enter at least one positive examples (empty to exit)")
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


def _describe():
    pos = open(_POSITIVE).readlines()
    neg = open(_NEGATIVE).readlines()
    neu = open(_NEUTRAL).readlines()
    unknowns = open(_UNKNOWN).readlines()

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
        print("Error: unknown.txtに教師データに含まれないデータがない")
        return

    X = np.array(X)
    probs = _model.predict_proba(X)
    scored_lines = []
    for i in range(len(used_lines)):
        scored_lines.append((probs[i, 1], used_lines[i]))

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
    if not _model:
        _learn()

    prob = _model.predict_proba(_make_features(line).reshape(1, -1))[0, 1]
    return (prob > 0.5)


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--initialize', action='store_true')
    parser.add_argument('--learn', action='store_true')
    parser.add_argument('--describe', action='store_true')
    args = parser.parse_args()

    if args.initialize:
        _initialize()

    if args.learn:
        _learn()

    if args.describe:
        _learn()
        _describe()


if __name__ == "__main__":
    _main()
