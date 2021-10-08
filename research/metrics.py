import numpy as np

def batch_jaccard(y_true, y_pred):
    intersection = (y_true * y_pred).sum((-1, -2, -3))
    union = y_true.sum((-1, -2, -3)) + y_pred.sum((-1, -2, -3)) - intersection
    return list((intersection + 1e-15) / (union + 1e-15))

def batch_dice(y_true, y_pred):
    return list((2 * (y_true * y_pred).sum((-1, -2, -3)) + 1e-15) /\
           (y_true.sum((-1, -2, -3)) + y_pred.sum((-1, -2, -3)) + 1e-15))


def jaccard(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)


def dice(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)
