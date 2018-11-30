
# coding: utf-8
'''
Iris Recognition
'''
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import warnings
from tabulate import tabulate
warnings.filterwarnings('ignore')


from IrisLocalization import iris_localization
from IrisNormalization import iris_normalization
from ImageEnhancement import image_enhancement
from FeatureExtraction import feature_extraction
from IrisMatching import iris_matching
from PerformanceEvaluation import performance_evaluation

'''
main function
'''
def main():
    
    image_train, good_img, y_train, image_test, good_img_test, y_test = load_data(folder = 'CASIA Iris Image Database (version 1.0)')
    #train data
    print('Preprocessing images...')
    X_train = []
    for i in good_img:
        img = cv2.cvtColor(image_train[i], cv2.COLOR_BGR2GRAY)
        #localization
        X_pupil, Y_pupil, r_pupil, X_iris, Y_iris, r_iris = iris_localization(img)
        #normalization
        img_norm = iris_normalization(X_pupil, Y_pupil, r_pupil, X_iris, Y_iris, r_iris, img)
        #enhancement
        img_enhance = image_enhancement(img_norm, img)
        #feature extraction
        feature_vec = feature_extraction(img_enhance)
        X_train.append(feature_vec)

    #test data
    X_test = []
   
    for i in good_img_test:
        img = cv2.cvtColor(image_test[i], cv2.COLOR_BGR2GRAY)
        #localization
        X_pupil, Y_pupil, r_pupil, X_iris, Y_iris, r_iris = iris_localization(img)
        #normalization
        img_norm = iris_normalization(X_pupil, Y_pupil, r_pupil, X_iris, Y_iris, r_iris, img)
        #enhancement
        img_enhance = image_enhancement(img_norm, img)
        #feature extraction
        feature_vec = feature_extraction(img_enhance)
        X_test.append(feature_vec)
    
    print(' \nFeature extraction done')
    #train model
    # try using different feature lengths
    print (' \nTrying different feature dimensions...')
    crr =[]
    num_feature = [20,60,70,80,90,105,107]
    for n in num_feature:
        y_pred_l1,  y_pred_l2,  y_pred_cos = iris_matching(X_train, y_train, X_test, y_test, reduced_feature= n)
        _, _, CRR_cos = performance_evaluation(y_test, y_pred_l1, y_pred_l2, y_pred_cos, plot = False)
        crr.append(CRR_cos)
        
    print (' \nGenerating fig 10...')    
    f10 = plt.figure(10)
    plt.plot(num_feature,crr)
    plt.xlabel('Dimensionality of the feature vector')
    plt.ylabel('Correct recognition rate')
    plt.savefig("fig10.png")
    print(' \nSaved plot as fig10.png')

    #use reduced feature set
    print(' \nUsing reduced features...')
    y_pred_l1,  y_pred_l2,  y_pred_cos = iris_matching(X_train, y_train, X_test, y_test, reduced_feature= True)
    print(' \nGenerating table 4 and fig 13...')
    CRR_l1_r, CRR_l2_r, CRR_cos_r = performance_evaluation(y_test, y_pred_l1, y_pred_l2, y_pred_cos, plot = True)

    #use original feature set
    y_pred_l1,  y_pred_l2,  y_pred_cos = iris_matching(X_train, y_train, X_test, y_test)
    #evaluate
    CRR_l1_o, CRR_l2_o, CRR_cos_o = performance_evaluation(y_test, y_pred_l1, y_pred_l2, y_pred_cos)
    #generate table 3 in the paper
    print(' \nGenerating table 3...')
    print(tabulate([['L1 distancs',CRR_l1_o,CRR_l1_r,],\
                    ['L2 distance',CRR_l2_o, CRR_l2_r],\
                    ['Cosine similarity',CRR_cos_o, CRR_cos_r]],\
                ['Similarity Measure','CRR with original feature','CRR wit reduced features']))
    print(' \nClose plot window to exit program')
    plt.show()


### Helper Functions
def load_data(folder = 'CASIA Iris Image Database (version 1.0)'):
    #load training images
    image_train = []
    label_train = []

    for i in range(1,109):
        for j in range(1,4):
            if i <= 9:
                train_path = folder + '/00' + str(i) + '/1' + '/00' + str(i) + '_1_' + str(j) + '.bmp'
            elif i>9 and i<=99:
                train_path = folder + '/0' + str(i) + '/1' + '/0' + str(i) + '_1_' + str(j) + '.bmp'
            else:
                train_path = folder + '/' + str(i) + '/1' + '/' + str(i) + '_1_' + str(j) + '.bmp'
            img = cv2.imread(train_path)
            image_train.append(img)
            label_train.append(i) 

    #get the idx of the images where iris localization works
    good_img = []
    error = []
    for i in range(0,len(image_train)):
        try:
            img = cv2.cvtColor(image_train[i], cv2.COLOR_BGR2GRAY)
            X_pupil, Y_pupil, r_pupil, X_iris, Y_iris, r_iris = iris_localization(img)
            good_img.append(i)
        except:
            error.append(i)
  
    #get y train
    y_train = []
    for i in good_img:
        y = label_train[i]
        y_train.append(y)

    #load test images
    image_test = []
    label_test = []

    for i in range(1,109):
        for j in range(1,5):
            if i <= 9:
                test_path = folder + '/00' + str(i) + '/2' + '/00' + str(i) + '_2_' + str(j) + '.bmp'
            elif i>9 and i<=99:
                test_path = folder + '/0' + str(i) + '/2' + '/0' + str(i) + '_2_' + str(j) + '.bmp'
            else:
                test_path = folder  + '/' + str(i) + '/2' + '/' + str(i) + '_2_' + str(j) + '.bmp'
            img = cv2.imread(test_path)
            image_test.append(img)
            label_test.append(i)      

    #get the idx of the images where iris localization works
    good_img_test = []
    error_test = []
    for i in range(0,len(image_test)):
        try:
            img = cv2.cvtColor(image_test[i], cv2.COLOR_BGR2GRAY)
            X_pupil, Y_pupil, r_pupil, X_iris, Y_iris, r_iris = iris_localization(img)
            good_img_test.append(i)
        except:
            error_test.append(i)
    
    #get y test
    y_test = []
    for i in good_img_test:
        y = label_test[i]
        y_test.append(y)
     
    return image_train, good_img, y_train, image_test, good_img_test,y_test


if __name__ == '__main__':
    main()