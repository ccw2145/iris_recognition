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

def iris_localization(img):
    #Step 1 
    #project the image in vertical and horizontal direction to estimate the center coordinates of the pupil
    
    #convert to grayscale
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #projection
    height = img.shape[0]
    width = img.shape[1]
    #vertical projection
    v_proj = np.array([y/255/width for y in img.sum(axis=1)])
    #horizontal projection
    h_proj = np.array([x/255/height for x in img.sum(axis=0)])
    #center coordinates
    Y_p = np.argmin(v_proj)
    X_p = np.argmin(h_proj)
    center = (X_p, Y_p)
    
    
    #Step 2
    #CROP a 120*120 region center at (X_p, Y_p) with threshold 
    new_img = img[Y_p-60:Y_p+60, X_p-60:X_p+60]
    #plt.imshow(new_img)
    #plt.show()
    
    #plot histogram to get threshold
    hist = cv2.calcHist([new_img],[0],None,[256],[0,256])  

    #binarize the image based on the above threshold
    thresh = 50
    bi_img = cv2.threshold(new_img, thresh, 255, cv2.THRESH_BINARY)[1]

    
    #estimate of center coordinates based on binarized image
    height_bi = bi_img.shape[0]
    width_bi = bi_img.shape[1]
  
    v_proj_bi = np.array([y/255/width_bi for y in bi_img.sum(axis=1)])
    #horizontal projection
    h_proj_bi = np.array([x/255/height_bi for x in bi_img.sum(axis=0)])
    #center coordinates
    Y_p_bi = np.argmin(v_proj_bi)
    X_p_bi = np.argmin(h_proj_bi)
    center_bi = (X_p_bi, Y_p_bi)
    
    #Step 3
    #calculate the exact parameters of the two circles using edge detection and hough transformation
    
    #remove noise, apply median filter then do canny detection on pupil
    new_img_copy = new_img.copy()
    med_img = cv2.medianBlur(new_img_copy,11)
    canny_edge_p = cv2.Canny(med_img, 35, 35)
    #plt.imshow(canny_edge_p)
    #plt.show()
    
    #hough transform and get the exact parameters for the pupil circle
    edge_circ_p = cv2.HoughCircles(canny_edge_p,cv2.HOUGH_GRADIENT,5,350,param1=50,param2=50,minRadius=40,maxRadius=50)
    
    #plot the detected circle
    if edge_circ_p is not None:
        edge_circ_p = np.round(edge_circ_p[0, :]).astype("int")
        for (x, y, r) in edge_circ_p:
            # draw the pupil circle      
            cv2.circle(new_img_copy, (x, y), r, (0, 255, 0), 3)  
  
    #get parameters for the pupil in the original image
    X_pupil = edge_circ_p[0][0]+X_p-60
    Y_pupil = edge_circ_p[0][1]+Y_p-60
    r_pupil = edge_circ_p[0][2]
    
    #run canny edge detection on iris
    img_copy = img.copy()
    med_img = cv2.medianBlur(img_copy,11)
    canny_edge_i = cv2.Canny(med_img, 35, 35)
   
    #remove pupil
    canny_edge_i[Y_pupil-r_pupil-30:Y_pupil+r_pupil+30,X_pupil-r_pupil-30:X_pupil+r_pupil+30] = 0
    
    edge_circ_i = cv2.HoughCircles(canny_edge_i,cv2.HOUGH_GRADIENT,6.5,350,param1=150,param2=250,minRadius=r_pupil+40,maxRadius=r_pupil+75)
    
    #plot the detected circle
    if edge_circ_i is not None:
        edge_circ_i = np.round(edge_circ_i[0, :]).astype("int")
        for (x, y, r) in edge_circ_i:
            # draw the iris circle     
            cv2.circle(img_copy, (x, y), r, (0, 255, 0), 3)        
   
    #get parameters for iris in the original image
    X_iris = edge_circ_i[0][0]
    Y_iris = edge_circ_i[0][1]
    r_iris = edge_circ_i[0][2]

    
    return(X_pupil, Y_pupil, r_pupil, X_iris, Y_iris, r_iris)