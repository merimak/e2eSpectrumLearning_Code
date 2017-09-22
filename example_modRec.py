# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 19:38:54 2017

@author: mkulin
"""

# import the necessary packages
from networks.modnet import ModNet2
import numpy as np
import argparse
from datasets.load_datasets import load_modRec_rth, load_modRec_fft, load_modRec_iq
from evaluation.evaluate import calc_predictions, calc_acc_per_snr

 
"""
Main driver program for training and collecting results
"""
if __name__ == '__main__':
    
    #Construct the argument parse
    arg = argparse.ArgumentParser()
    arg.add_argument("-l", "--load-model", type=int, default=-1, help="(optional) whether or not pre-trained model should be loaded", dest="load_model")
    arg.add_argument("-w", "--weights", type=str, help="(optional) path to weights file or folder", dest="weights")
    arg.add_argument("-d", "--load_data", type=int, help="1-iq, 2-rth, 3-fft", dest="load_data")
    
    args = vars(arg.parse_args())
    #D:\mkulin\Documents\Software\ipython\radioml_paper\output\
    
    data=""
    #Load data
    print("[INFO] loading datasets...")
    if args["load_data"]==1:
        X_train, X_test, Y_train, Y_test, test_SNRs = load_modRec_iq()
        data="iq"
    elif args["load_data"]==2:
        X_train, X_test, Y_train, Y_test, test_SNRs = load_modRec_rth()
        data="rth"
    elif args["load_data"]==3:
        X_train, X_test, Y_train, Y_test, test_SNRs = load_modRec_fft()
        data="fft"
    
    #Constants
    snrs=[-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    classes=['8PSK', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK']
    path=args["weights"] + 'CNN2' + '_' + data
    
    #Initialize the optimizer and model
    print("[INFO] compiling model...")

    cnn= ModNet2(0.0001, 0.5)
    cnn.build(list(X_train.shape[1:]), classes, weightsPath=args["weights"] if args["load_model"] > 0 else None)

    #Train the model if a pre-existing model is not loaded
    if args["load_model"] < 0:
        print("[INFO] training...")
        cnn.train(X_train, Y_train, X_test, Y_test, nb_epoch=50, batch_size=1024, basepath=path)
        
        # show the accuracy on the testing set
        print("[INFO] evaluating...")
        loss = cnn.model.evaluate(X_test,  Y_test, batch_size=1024, verbose=1)
        #print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))
        print("[INFO] loss: {:.2f}%".format(loss * 100))
    
    #Collect results for performance evaluation
    Y_truth, Y_pred=calc_predictions(cnn.model, classes, X_test, Y_test, path=path)
    calc_acc_per_snr(cnn.model, snrs, classes, test_SNRs, X_test, Y_test, path=path)
	
	
	
