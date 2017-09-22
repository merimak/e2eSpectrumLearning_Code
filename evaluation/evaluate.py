# -*- coding: utf-8 -*-

import numpy as np
import cPickle
from sys import platform

def dump_result(path, name, var):
    res_path=path + name
    fs = open(res_path, 'wb')
    cPickle.dump(var, fs)
    fs.close()


def calc_predictions(model, classes, X_test, Y_test, path=None):

    #All SNR confusion matrix
    Y_pred=[] #Integer representations of predicted labels
    Y_truth=[] #Integer representations of true labels
    test_Y_hat = model.predict(X_test, batch_size=1024)
    conf = np.zeros([len(classes),len(classes)])
    confnorm = np.zeros([len(classes),len(classes)])
    for i in range(0, X_test.shape[0]):
        j = list(Y_test[i,:]).index(1) #find the truth modulation
        Y_truth.append(j)
        k = int(np.argmax(test_Y_hat[i,:])) #find the estimated/predicted modulation
        Y_pred.append(k)
        conf[j,k] = conf[j,k] + 1
    for i in range(0,len(classes)):
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
    
    dump_result(path, '.snr_confn_results.dat', confnorm)
    dump_result(path, '.Y_truth.dat', Y_truth)
    dump_result(path, '.Y_pred.dat', Y_pred)
    
    return Y_truth, Y_pred

def calc_acc_per_snr(model, snrs, classes, test_SNRs, X_test, Y_test, path=None):

    # Accuracy per SNR
    acc = {}
    for snr in snrs:
        # extract classes @ SNR
        #test_SNRs = map(lambda x: lbl[x][1], test_idx)
        test_X_i = X_test[np.where(np.array(test_SNRs)==snr)]
        test_Y_i = Y_test[np.where(np.array(test_SNRs)==snr)]
        # estimate classes
        test_Y_i_hat = model.predict(test_X_i)
        conf = np.zeros([len(classes),len(classes)])
        for i in range(0,test_X_i.shape[0]):
            j = list(test_Y_i[i,:]).index(1)
            k = int(np.argmax(test_Y_i_hat[i,:]))
            conf[j,k] = conf[j,k] + 1
    
        cor = np.sum(np.diag(conf))
        ncor = np.sum(conf) - cor
        print "Overall Accuracy: ", cor / (cor+ncor)
        acc[snr] = 1.0*cor/(cor+ncor)
     
    dump_result(path, '.snr_acc_results.dat', acc) 

    