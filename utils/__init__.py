import os
import torch
import numpy as np
import pandas as pd
import random
from cdt.metrics import SID, SHD

def backRE(tar_DAG, P_KCI):
    sid_val = np.max([SID(tar_DAG, P_KCI), SID(P_KCI, tar_DAG)])
    shd_val = SHD(tar_DAG, P_KCI)
    return [shd_val, sid_val]

def f1_score(y_true, y_pred):
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    precision = true_positive / (np.sum(y_pred == 1) + 1e-10)
    recall = true_positive / (np.sum(y_true == 1) + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    return precision, recall, f1

def l2_distance(point1, point2):
    point1 = np.array(point1)
    point2 = np.array(point2)
    distance = np.sqrt(np.sum((point1 - point2)**2))
    return distance

