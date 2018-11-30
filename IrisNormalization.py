import pandas as pd
import numpy as np
import cv2
import math
import scipy.signal
import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

def iris_normalization(Xp,Yp,Rp,Xi,Yi,Ri,eye):
    M, N = 64, 512
    eye_norm = np.zeros((M,N))
    theta = (2*math.pi)/N
    delta = (Ri-Rp)/M
    i = 0
    for r in np.arange(Rp,Rp + eye_norm.shape[0]*delta, delta):
        j = 0
        for angle in np.arange(0,eye_norm.shape[1]*theta,theta):
            x = int(round(Xp + r*math.cos(angle)))
            y = int(round(Yp + r*math.sin(angle)))
            if y < 280 and x < 320:
                eye_norm[i][j] = eye[y][x]
            elif y > 280 and x < 320:
                eye_norm[i][j] = eye[279][x]
            elif y < 280 and x > 320:
                eye_norm[i][j] = eye[y][319]
            else:
                eye_norm[i][j] = eye[279][319]
            j += 1
        i += 1
    return eye_norm