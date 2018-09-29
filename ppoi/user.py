# -*- coding: utf-8 -*-
import numpy as np

def make_features(s):
    "take a string, return np.array"

    try:
        nnl = int(s.split("\t")[-1])
        nnl = [int(nnl == 0), int(nnl == 1), int(nnl > 1), 0]
    except:
        nnl = [0, 0, 0, 1]

    extra = [int("..." in s), int("\t注" in s), int("・" in s)]
    BEST100 = (" のーンenoia0スるrトtルい1イクとアな,をリsにはッ2シc"
               "的学知ラlテジプすpしhタグデdてィ.か・たm化3ロフ理でレ"
               "ドu生がニ性人4マコSムg95法きりエョ識yP分76-8bれミ自経えT論く")
    s = set(s)
    return np.array([(1 if c in s else 0) for c in BEST100] + nnl + extra)


def default_make_features(s):
    "take a string, return np.array"
    BEST100 = (" のーンenoia0スるrトtルい1イクとアな,をリsにはッ2シc"
               "的学知ラlテジプすpしhタグデdてィ.か・たm化3ロフ理でレ"
               "ドu生がニ性人4マコSムg95法きりエョ識yP分76-8bれミ自経えT論く")
    s = set(s)
    return np.array([(1 if c in s else 0) for c in BEST100])
    

make_features = default_make_features
