#!/usr/bin/python3
from sklearn.svm import SVC

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


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###
svClassifier = SVC(kernel='linear')

t0 = time()

svClassifier.fit(features_train, labels_train)
print("Training time: ", round(time()-t0, 3), "s")

svClassifier.predict(features_test)
print("Predicting time: ", round(time()-t0, 3), "s")

print(svClassifier.score(features_test, labels_test))

#########################################################

#########################################################
'''
You'll be Provided similar code in the Quiz
But the Code provided in Quiz has an Indexing issue
The Code Below solves that issue, So use this one
'''

# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]

#########################################################
