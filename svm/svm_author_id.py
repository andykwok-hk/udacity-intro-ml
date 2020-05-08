#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]

clf = SVC(kernel="rbf", C=10**4)
print("Start training")
t0 = time()
clf.fit(features_train, labels_train)
print((time()-t0))
print("Start prediting")
t1 = time()
pred = clf.predict(features_test)
print(time()-t1)
print(accuracy_score(labels_test,pred))
print((pred==1).sum())
#########################################################
