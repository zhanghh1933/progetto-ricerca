import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spy
import pickle
import time
import datetime
import random

from sklearn import datasets, linear_model, svm, model_selection, preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, classification_report
from sklearn.externals import joblib



lr = linear_model.LogisticRegression()
lsvm = svm.LinearSVC()
rf = RandomForestClassifier(n_estimators=100)

f = open('debug.log', 'a')
f.write(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
f.write("\n")



#load all data in a time
a_add = spy.loadmat('D:\\Top Secret\\Università\\progetto_ricerca\\script\\output\\addi_features_all.mat')
a_dct = spy.loadmat('D:\\Top Secret\\Università\\progetto_ricerca\\script\\output\\deq_dct_coef_all.mat')
a_add_x = a_add['features_all']
a_dct_x = a_dct['features_all']
a = np.column_stack((a_add_x, a_dct_x))
target = a_add['target']
target = np.array(target.ravel()).astype(int)
dataset = np.column_stack((target, a))
dataset = np.delete(dataset,(8), axis=0)
#np.random.shuffle(dataset)

Y = dataset[:, :1]
Y = np.array(Y.ravel()).astype(int)
X = dataset[:, 1:]
X = preprocessing.scale(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
scoring = 'accuracy'
kfold = model_selection.KFold(n_splits=10, random_state=7)

print('logistic regression \n')
f.write('logistic regression \n')
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
confusion_lr = confusion_matrix(y_test, y_pred_lr)
print(confusion_lr)
print('\n')
print(classification_report(y_test, y_pred_lr))
print('\n')
result_lr = model_selection.cross_val_score(lr, X_train, y_train, cv=kfold, scoring=scoring)
print(result_lr.mean())
f.write("%f" %result_lr.mean())
f.write("\n")
print('\n')

print('linear support vector machine \n')
f.write('linear support vector machine \n')
lsvm.fit(X_train, y_train)
y_pred_lsvm = lsvm.predict(X_test)
confusion_lsvm = confusion_matrix(y_test, y_pred_lsvm)
print(confusion_lsvm)
print('\n')
print(classification_report(y_test, y_pred_lsvm))
print('\n')
result_lsvm = model_selection.cross_val_score(lsvm, X_train, y_train, cv=kfold, scoring=scoring)
print(result_lsvm.mean())
f.write("%f" %result_lsvm.mean())
f.write("\n")
print('\n')

print('random forest \n')
f.write('random forest \n')
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
confusion_rf = confusion_matrix(y_test, y_pred_rf)
print(confusion_rf)
print('\n')
print(classification_report(y_test, y_pred_rf))
print('\n')
result_rf = model_selection.cross_val_score(rf, X_train, y_train, cv=kfold, scoring=scoring)
print(result_rf.mean())
f.write("%f" %result_rf.mean())
f.write("\n")
print('\n')


f.close()
