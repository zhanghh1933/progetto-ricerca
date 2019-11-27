import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spy
import pickle
import time
import datetime
import random

from sklearn import datasets, linear_model, svm, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.externals import joblib



lr = linear_model.LogisticRegression()
lsvm = svm.LinearSVC()
rf = RandomForestClassifier(n_estimators=100)

f = open('debug.log', 'a')
f.write(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
f.close()

#load data of messenger
M = spy.loadmat('D:\\Top Secret\\Università\\progetto_ricerca\\script\\output\\addi_features_m.mat')
M_dct = spy.loadmat('D:\\Top Secret\\Università\\progetto_ricerca\\script\\output\\deq_dct_coef_m.mat')
#divide dataset in features and label
M_add_x = M['Features_M']
M_dct_x = M_dct['Hist_M']
M_y = M['M']
M_x = np.column_stack((M_add_x, M_dct_x))


#load data for telegram
T = spy.loadmat('D:\\Top Secret\\Università\\progetto_ricerca\\script\\output\\addi_features_t.mat')
T_dct = spy.loadmat('D:\\Top Secret\\Università\\progetto_ricerca\\script\\output\\deq_dct_coef_t.mat')
#divide dataset in features and label
T_add_x = T['Features_T']
T_dct_x = T_dct['Hist_T']
T_y = T['T']
T_x = np.column_stack((T_add_x, T_dct_x))


#load data for whatsapp
W = spy.loadmat('D:\\Top Secret\\Università\\progetto_ricerca\\script\\output\\addi_features_w.mat')
W_dct = spy.loadmat('D:\\Top Secret\\Università\\progetto_ricerca\\script\\output\\deq_dct_coef_w.mat')
#divide dataset in features and label
W_add_x = W['Features_W']
W_dct_x = W_dct['Hist_W']
W_y = W['W']
W_x = np.column_stack((W_add_x, W_dct_x))


#load data for original
O = spy.loadmat('D:\\Top Secret\\Università\\progetto_ricerca\\script\\output\\addi_features_o.mat')
O_dct = spy.loadmat('D:\\Top Secret\\Università\\progetto_ricerca\\script\\output\\deq_dct_coef_o.mat')
#divide dataset in features and label
O_add_x = O['Features_O']
O_dct_x = O_dct['Hist_O']
O_y = O['O']
O_x = np.column_stack((O_add_x, O_dct_x))



total_precision_lr = 0
total_precision_lsvm = 0
total_precision_rf = 0


dataset = np.concatenate((M_x, T_x, W_x, O_x))
dataset = preprocessing.scale(dataset)
label = np.concatenate((M_y, T_y, W_y, O_y))
label = np.array(label.ravel()).astype(int)
dataset = np.column_stack((label, dataset))
np.random.shuffle(dataset)

index = 0
step = 1
train_size = int(len(dataset)*0.3)
lenght = len(dataset)

while (index + train_size < len(dataset)):
    dataset_train = dataset[index:index + train_size]
    dataset_test = np.concatenate((dataset[0:index], dataset[index + train_size:lenght]))

    x_train = dataset_train[: , 1:]
    y_train = dataset_train[:, :1]
    y_train = np.array(y_train.ravel()).astype(int)
    
    x_test = dataset_test[: , 1:]
    y_test = dataset_test[:, :1]
    y_test = np.array(y_test.ravel()).astype(int)
    


    lr.fit(x_train, y_train)
    lsvm.fit(x_train, y_train)
    rf.fit(x_train, y_train)

    f = open('debug1.log', 'a')

    pr_lr = lr.predict(x_test)
    cf_lr = confusion_matrix(y_test, pr_lr)
    a_precision_lr = precision_score(y_test, pr_lr, average='macro')
    total_precision_lr = total_precision_lr + a_precision_lr
    print('Precision logistic regression at step ', step , ' is ' , a_precision_lr , '\n')
    f.write("Precision logistic regression at step %f"  %step)
    f.write(" is %f"  %a_precision_lr)
    f.write("\n")

    pr_lsvm = lsvm.predict(x_test)
    cf_lsvm = confusion_matrix(y_test, pr_lsvm)
    a_precision_lsvm = precision_score(y_test, pr_lsvm, average='macro')
    total_precision_lsvm = total_precision_lsvm + a_precision_lsvm
    print('Precision Linear SVM at step ', step , ' is ' , a_precision_lsvm , '\n')
    f.write("Precision linear SVM at step %f"  %step)
    f.write(" is %f"  %a_precision_lsvm)
    f.write("\n")

    pr_rf = rf.predict(x_test)
    cf_rf = confusion_matrix(y_test, pr_rf)
    a_precision_rf = precision_score(y_test, pr_rf, average='macro')
    total_precision_rf = total_precision_rf + a_precision_rf
    print('Precision random forest at step ', step , ' is ' , a_precision_rf , '\n')
    f.write("Precision random forest at step %f"  %step)
    f.write(" is %f"  %a_precision_rf)
    f.write("\n")
    print('\n\n')
    f.close()
    index = index + 5

    

lr_precision = total_precision_lr / step
lsvm_precision = total_precision_lsvm / step
rf_precision = total_precision_rf / step

print('final precision score: \n')
print('logistic regression: ', lr_precision , '\n')
print('linear svm: ' , lsvm_precision , '\n')
print('random forest: ' , rf_precision , '\n')

f = open('debug.log', 'a')
f.write("final precision score: \n")
f.write("logistic regression: %f" %lr_precision)
f.write("\n")
f.write("linear svm: %f" %lsvm_precision)
f.write("\n")
f.write("random forest: %f" %rf_precision)
f.write("\n")
f.close()


joblib.dump(rf, 'random_forest_single_scenario.pkl')
joblib.dump(lsvm, 'linear_svm_single_scenario.pkl')
joblib.dump(lr, 'logistic_single_scenario.pkl')






