import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

def performance_evaluation(y_test, y_pred_l1, y_pred_l2, y_pred_cos, plot= False):
    
    cnt_l1 = 0
    cnt_l2 = 0
    cnt_cos = 0
    
    for k in range(len(y_test)):
        if y_test[k] == y_pred_l1[k]:
            cnt_l1 += 1
        if y_test[k] == y_pred_l2[k]:
            cnt_l2 += 1    
        if y_test[k] == y_pred_cos[k]:
            cnt_cos += 1    
       
    CRR_l1 = round(cnt_l1/len(y_test), 4)
    CRR_l2 = round(cnt_l2/len(y_test), 4)
    CRR_cos = round(cnt_cos/len(y_test), 4)
        
    #plot false match rate vs. false nonmatch rate as mentioned 
    if plot == True:
        plot_roc(y_test, [y_pred_l1, y_pred_l2, y_pred_cos])
        
    
    return CRR_l1, CRR_l2, CRR_cos

## Helper function
def plot_roc(y_test, y_preds):
   
    f13 = plt.figure(13)
    classes = np.unique(y_test)

    y_test = label_binarize(y_test, classes = classes)
   
    names=['l1 dist','l2 dist','cosine ']
    print('')
    print ('Measure__Threshold__FMR___FNMR')
    for n,pred in enumerate(y_preds):
        
        pred = label_binarize(list(pred), classes = classes)
      
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        thresh = dict()
        roc_auc = dict()
        
        # Compute roc for classification
        # fnr = 1 - tpr
        fpr["micro"], tpr["micro"], thresh["micro"] = roc_curve(y_test.ravel(), pred.ravel())
        roc_auc["micro"] = auc(fpr["micro"], 1-tpr["micro"])
        
        print ('{}_____{}____{:0.2}___{:0.2}'.format(names[n],thresh["micro"][1],fpr["micro"][1], 1-tpr["micro"][1]))
        
        plt.plot(fpr["micro"], 1-tpr["micro"], label = 'AUC = %0.2f (%s)' % (roc_auc["micro"],names[n]))
        
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc = 'lower right')
    plt.plot([1, 0], [0, 1],'r--')
    plt.xlim([0, 0.5])
    plt.ylim([0, 0.5])
    plt.ylabel('False Non-Match Rate')
    plt.xlabel('False Match Rate')
    plt.savefig('fig13.png')
    print(' \nSaved plot as fig13.png')
