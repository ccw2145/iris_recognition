import pandas as pd
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.preprocessing import StandardScaler

def iris_matching(X_train, y_train, X_test, y_test, reduced_feature = False):
    
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    X_train_trans = X_train
    X_test_trans = X_test
    
    if reduced_feature == True:
        lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
        X_train_trans = lda.fit_transform(X_train,y_train)
        X_test_trans = lda.transform(X_test)

    elif reduced_feature > 1:
        lda = LinearDiscriminantAnalysis(n_components = reduced_feature ,solver='eigen', shrinkage='auto')
        X_train_trans = lda.fit_transform(X_train,y_train)
        X_test_trans = lda.transform(X_test)
        
     
    #run KNN classifier to get predicted labels
    # using manhattan_distance (l1)
    knn_l1 = KNeighborsClassifier(n_neighbors = 1, metric='manhattan')
    knn_l1.fit(X_train_trans,y_train)
    y_pred_l1 = knn_l1.predict(X_test_trans)
    
    # using euclidean_distance (l2).
    knn_l2 = KNeighborsClassifier(n_neighbors = 1,metric='euclidean')
    knn_l2.fit(X_train_trans,y_train)
    y_pred_l2 = knn_l2.predict(X_test_trans)
    
    # nearest centroid using cosine similarity
    knn_cos = KNeighborsClassifier(n_neighbors = 1,metric='cosine')
    knn_cos.fit(X_train_trans,y_train)
    y_pred_cos =knn_cos.predict(X_test_trans)
    
    return y_pred_l1,  y_pred_l2,  y_pred_cos

    