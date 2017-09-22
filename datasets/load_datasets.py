# -*- coding: utf-8 -*-

'''
Created on Nov 24, 2016

@author: Merima Kulin
'''
import os
from os.path import dirname
import cPickle
from sys import platform
import numpy as np


def load_modRec_rth(dtype=1):
    """Load and return the amplitude/phase spectrum data

    """
    base_path = dirname(__file__)
    # Load the datasets 
    if platform == "linux" or platform == "linux2":
        if dtype==1:
            with open(base_path + '/data/RadioML_Xrth_traindata.dat') as fid:
                X_train = cPickle.load(fid)
                fid.close()
            with open(base_path + '/data/RadioML_Xrth_testdata.dat') as fid:
                X_test = cPickle.load(fid)
                fid.close()   
            with open(base_path + '/data/RadioML_Y_traindata.dat') as fid:
                Y_train = cPickle.load(fid)
                fid.close()
            with open(base_path + '/data/RadioML_Y_testdata.dat') as fid:
                Y_test = cPickle.load(fid)
                fid.close()   
            with open(base_path + '/data/RadioML_testsnrs.dat') as fid:
                test_SNRs = cPickle.load(fid)
                fid.close()   
        elif dtype==2:
            with open(base_path + '/data/ISM_interference_data/Interference_Xrth_traindata.dat') as fid:
                X_train = cPickle.load(fid)
                fid.close()
            with open(base_path + '/data/ISM_interference_data/Interference_Xrth_testdata.dat') as fid:
                X_test = cPickle.load(fid)
                fid.close()   
            with open(base_path + '/data/ISM_interference_data/Interference_Y_traindata.dat') as fid:
                Y_train = cPickle.load(fid)
                fid.close()
            with open(base_path + '/data/ISM_interference_data/Interference_Y_testdata.dat') as fid:
                Y_test = cPickle.load(fid)
                fid.close()   
            with open(base_path + '/data/ISM_interference_data/Interference_testsnrs.dat') as fid:
                test_SNRs = cPickle.load(fid)
                fid.close()
        
    elif platform == "win32":
    # Load the datasets 
        with open(base_path + '\\data\\RadioML_Xrth_traindata.dat') as fid:
            X_train = cPickle.load(fid)
            fid.close()
        with open(base_path + '\\data\\RadioML_Xrth_testdata.dat') as fid:
            X_test = cPickle.load(fid)
            fid.close()   
        with open(base_path + '\\data\\RadioML_Y_traindata.dat') as fid:
            Y_train = cPickle.load(fid)
            fid.close()
        with open(base_path + '\\data\\RadioML_Y_testdata.dat') as fid:
            Y_test = cPickle.load(fid)
            fid.close()   
        with open(base_path + '\\data\\RadioML_testsnrs.dat') as fid:
            test_SNRs = cPickle.load(fid)
            fid.close()   

    return X_train, X_test, Y_train, Y_test, test_SNRs
      
def load_modRec_fft(dtype=1):
    """Load and return the frequency domain spectrum data

    """
    base_path = dirname(__file__)
    # Load the datasets 
    if platform == "linux" or platform == "linux2":
        if dtype==1:
            with open(base_path + '/data/RadioML_Xfft_traindata.dat') as fid:
                X_train = cPickle.load(fid)
                fid.close()
            with open(base_path + '/data/RadioML_Xfft_testdata.dat') as fid:
                X_test = cPickle.load(fid)
                fid.close()   
            with open(base_path + '/data/RadioML_Y_traindata.dat') as fid:
                Y_train = cPickle.load(fid)
                fid.close()
            with open(base_path + '/data/RadioML_Y_testdata.dat') as fid:
                Y_test = cPickle.load(fid)
                fid.close()   
            with open(base_path + '/data/RadioML_testsnrs.dat') as fid:
                test_SNRs = cPickle.load(fid)
                fid.close()  
        elif dtype==2:
            with open(base_path + '/data/ISM_interference_data/Interference_Xfft_traindata.dat') as fid:
                X_train = cPickle.load(fid)
                fid.close()
            with open(base_path + '/data/ISM_interference_data/Interference_Xfft_testdata.dat') as fid:
                X_test = cPickle.load(fid)
                fid.close()   
            with open(base_path + '/data/ISM_interference_data/Interference_Y_traindata.dat') as fid:
                Y_train = cPickle.load(fid)
                fid.close()
            with open(base_path + '/data/ISM_interference_data/Interference_Y_testdata.dat') as fid:
                Y_test = cPickle.load(fid)
                fid.close()   
            with open(base_path + '/data/ISM_interference_data/Interference_testsnrs.dat') as fid:
                test_SNRs = cPickle.load(fid)
                fid.close()
   
    elif platform == "win32":
        with open(base_path + '\\data\\RadioML_Xfft_traindata.dat') as fid:
            X_train = cPickle.load(fid)
            fid.close()
        with open(base_path + '\\data\\RadioML_Xfft_testdata.dat') as fid:
            X_test = cPickle.load(fid)
            fid.close()   
        with open(base_path + '\\data\\RadioML_Y_traindata.dat') as fid:
            Y_train = cPickle.load(fid)
            fid.close()
        with open(base_path + '\\data\\RadioML_Y_testdata.dat') as fid:
            Y_test = cPickle.load(fid)
            fid.close()   
        with open(base_path + '\\data\\RadioML_testsnrs.dat') as fid:
            test_SNRs = cPickle.load(fid)
            fid.close()   

    return X_train, X_test, Y_train, Y_test, test_SNRs

        
def load_modRec_iq(dtype=1):
    """Load and return the complex-temporal spectrum data
        1-modulation recognition data, 2-interference detection data
    """
    base_path = dirname(__file__)
    # Load the datasets 
    if platform == "linux" or platform == "linux2":
        if dtype==1:
            with open(base_path + '/data/RadioML_Xiq_traindata.dat') as fid:
                X_train = cPickle.load(fid)
                fid.close()
            with open(base_path + '/data/RadioML_Xiq_testdata.dat') as fid:
                X_test = cPickle.load(fid)
                fid.close()   
            with open(base_path + '/data/RadioML_Y_traindata.dat') as fid:
                Y_train = cPickle.load(fid)
                fid.close()
            with open(base_path + '/data/RadioML_Y_testdata.dat') as fid:
                Y_test = cPickle.load(fid)
                fid.close()   
            with open(base_path + '/data/RadioML_testsnrs.dat') as fid:
                test_SNRs = cPickle.load(fid)
                fid.close()  
            
        elif dtype==2:
            with open(base_path + '/data/ISM_interference_data/Interference_Xiq_traindata.dat') as fid:
                X_train = cPickle.load(fid)
                fid.close()
            with open(base_path + '/data/ISM_interference_data/Interference_Xiq_testdata.dat') as fid:
                X_test = cPickle.load(fid)
                fid.close()   
            with open(base_path + '/data/ISM_interference_data/Interference_Y_traindata.dat') as fid:
                Y_train = cPickle.load(fid)
                fid.close()
            with open(base_path + '/data/ISM_interference_data/Interference_Y_testdata.dat') as fid:
                Y_test = cPickle.load(fid)
                fid.close()   
            with open(base_path + '/data/ISM_interference_data/Interference_testsnrs.dat') as fid:
                test_SNRs = cPickle.load(fid)
                fid.close()
   
    elif platform == "win32":
        if dtype==1:
            with open(base_path + '\\data\\RadioML_Xiq_traindata.dat') as fid:
                X_train = cPickle.load(fid)
                fid.close()
            with open(base_path + '\\data\\RadioML_Xiq_testdata.dat') as fid:
                X_test = cPickle.load(fid)
                fid.close()   
            with open(base_path + '\\data\\RadioML_Y_traindata.dat') as fid:
                Y_train = cPickle.load(fid)
                fid.close()
            with open(base_path + '\\data\\RadioML_Y_testdata.dat') as fid:
                Y_test = cPickle.load(fid)
                fid.close()   
            with open(base_path + '\\data\\RadioML_testsnrs.dat') as fid:
                test_SNRs = cPickle.load(fid)
                fid.close()   
                
        elif dtype==2:
            with open(base_path + '\\data\\ISM_interference_data\\Interference_Xiq_traindata.dat') as fid:
                X_train = cPickle.load(fid)
                fid.close()
            with open(base_path + '\\data\\ISM_interference_data\\Interference_Xiq_testdata.dat') as fid:
                X_test = cPickle.load(fid)
                fid.close()   
            with open(base_path + '\\data\\ISM_interference_data\\Interference_Y_traindata.dat') as fid:
                Y_train = cPickle.load(fid)
                fid.close()
            with open(base_path + '\\data\\ISM_interference_data\\Interference_Y_testdata.dat') as fid:
                Y_test = cPickle.load(fid)
                fid.close()   
            with open(base_path + '\\data\\ISM_interference_data\\Interference_testsnrs.dat') as fid:
                test_SNRs = cPickle.load(fid)
                fid.close()
        
    return X_train, X_test, Y_train, Y_test, test_SNRs