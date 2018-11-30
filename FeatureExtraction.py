import pandas as pd
import numpy as np
import math
import scipy.signal

def feature_extraction(img):
    
    #channel definition constants given in the paper
    dx_1, dy = 3, 1.5
    dx_2 = 4.5
    
    #frequecy
    f = 1/ dy
    
    #filter size
    h = 8
    w = 8

    # create 8 by 8 gabor filters for 2 channels
    filter_c1 = np.zeros((w,h))   
    filter_c2 = np.zeros((w,h))   

    for i in range(w):
        for j in range(h):
            filter_c1[i, j] = gabor_filter(i+1,j+1,dx_1,dy,f)
            filter_c2[i, j] = gabor_filter(i+1,j+1,dx_2,dy,f)
    
    filter_c1 = filter_c1.T
    filter_c2 = filter_c2.T
    
    vec1 = features_by_block(img, filter_c1)
    
    vec2 = features_by_block(img, filter_c2)
    
    features = np.append(vec1,vec2)
    
    return features

## Helper functions
# get gabor filter with defined modulating function
def gabor_filter(x, y, dx, dy, f):
    
    # M1 modulating function in Li's paper
    m = math.cos(2 * math.pi * f * (math.sqrt(x ** 2 + y ** 2)))
    
    # Gabor filter with defined modulating function
    gabor = (1/(2 * math.pi * dx * dy)) * math.exp(-1/2 * (x ** 2 / dx ** 2 +  y ** 2 / dy ** 2)) * m
    
    return gabor


# generate filtered image by convolution and get feature vector for filtered image
def features_by_block(img, kernel):
    
    vec = np.empty((0))
    i = 0
    filtered_img = np.zeros_like(img)
    
    #convolution with 8 by 8 block with stride = 1
    #filter size
    h = 8
    w = 8
    while i <= img.shape[0]-8:
        j = 0
        while j <= img.shape[1]-8:
            
            block = img[i:i+8,j:j+8]
            #filtered_block = cv2.filter2D(block, kernel=kernel,  ddepth=-1)
            filtered_block = scipy.signal.convolve2d(block, kernel, mode='same')
            filtered_img[i:i+8,j:j+8] = filtered_block

            j = j + 1

        i = i + 1

    
    #get features by taking mean and standard deviation of every 8 by 8 block
    for i in np.arange(0,img.shape[0],w):
        
         for j in np.arange(0,img.shape[1],h):
            
            block = filtered_img[i:i+w,j:j+h]
            mean = block.mean()
            st = block.std()
            vec = np.append(vec,[mean, st])
            
    return vec