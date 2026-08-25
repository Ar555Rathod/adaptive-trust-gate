"""Shared blending formula used by every hybrid model (3-7):

    r_hat(u,i) = g(u,i) * CF_score(u,i) + (1 - g(u,i)) * Content_score(u,i)

Only g(u,i) changes between models; this combiner stays fixed so the
comparison across gating mechanisms is apples-to-apples.
"""
import numpy as np

from atg import config


def blend_scores(cf_scores, cb_scores, g, clip: bool = True):
    cf_scores = np.asarray(cf_scores, dtype=float)
    cb_scores = np.asarray(cb_scores, dtype=float)
    g = np.asarray(g, dtype=float)
    pred = g * cf_scores + (1 - g) * cb_scores
    if clip:
        pred = np.clip(pred, config.RATING_MIN, config.RATING_MAX)
    return pred
