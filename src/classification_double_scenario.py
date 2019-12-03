import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spy
import pickle
import time
import datetime
import random
import itertools

from src.plot_confusion_matrix import plot_confusion_matrix, write_matrix, write_row, write_figure, write_log_table

from sklearn import datasets, linear_model, svm, model_selection, preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, classification_report
from sklearn.externals import joblib
import src.settings as sett


###########################################################################################################
#
#   not need anymore, do not run
#
############################################################################################################

file_name = 'report_double.tex'
file_add = 'addi_features_all_double.mat'

file_dct_Y = 'deq_dct_coef_all_double.mat'

file_dct_CbCr = 'deq_dct_coef_all_double1.mat'

# f = open('D:\\Top Secret\\Università\\progetto_ricerca\\script\\report\\report_double2.tex', 'w')
f = open('D:\\Top Secret\\Università\\progetto_ricerca\\script\\report3\\' + file_name, 'w')

#################################

# modificato al volo per eliminare le classi dove vengono sharate per la stessa app

lr = linear_model.LogisticRegression()
lsvm = svm.LinearSVC()
rf = RandomForestClassifier(n_estimators=100)

np.set_printoptions(precision=2)

# load all data in a time
a_add = spy.loadmat('D:\\Top Secret\\Università\\progetto_ricerca\\script\\output\\' + file_add)
a_dct = spy.loadmat('D:\\Top Secret\\Università\\progetto_ricerca\\script\\output\\' + file_dct_Y)
a_add_x = a_add['features_all']
a_dct_x = a_dct['features_all']
a = np.column_stack((a_add_x, a_dct_x))
target = a_add['target']
target = np.array(target.ravel()).astype(int)
dataset = np.column_stack((target, a))

Y = dataset[:, :1]
Y = np.array(Y.ravel()).astype(int)
X = dataset[:, 1:]
X = preprocessing.scale(X)

f.write('\\chapter{Double Scenario Classification of the last shared app, KFold Validation}')

Y[:1050] = 1
Y[1050:2100] = 2
Y[2100:3150] = 3

simple_class = ['1', '2', '3', '4']
simple_class_num = [1, 2, 3, 4]
simple_class_name = ['messenger', 'telegram', 'whatsapp', 'original']
simple_class_name_up = ['', 'messenger', 'telegram', 'whatsapp', 'original']
simple_class_name_up = np.array(simple_class_name_up)
simple_class_name_up = np.reshape(simple_class_name_up, [1, 5])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.7, random_state=0)
scoring = 'accuracy'
kfold = model_selection.KFold(n_splits=10, random_state=7)
f.write(
    'Starting with fitting randomly the classifiers, there are some statistics of the data used for the first test: \\\\')

stat = [
    ['', 'count train', 'count test'],
    ['messenger', list(y_train).count(1), list(y_test).count(1)],
    ['telegram', list(y_train).count(2), list(y_test).count(2)],
    ['whatsapp', list(y_train).count(3), list(y_test).count(3)],
    ['original', list(y_train).count(4), list(y_test).count(4)]]

write_matrix(f, np.array(stat))

# Logistic Regression
f.write('\\section{Logistic regression results:} \n')
f.write('Confusion matrix with number of sample and with normalization:')

lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
confusion_lr = confusion_matrix(y_test, y_pred_lr)

plt.figure()
plot_confusion_matrix(confusion_lr, simple_class_name, title='logistic confusion matrix, last app', normalize=True)
plt.savefig('D:\\Top Secret\\Università\\progetto_ricerca\\script\\report\\lr_initial_double_simple.png')
plt.close()

confusion_lr = np.column_stack((np.array(simple_class_name), confusion_lr))
confusion_lr = np.append(simple_class_name_up, confusion_lr, axis=0)
write_matrix(f, confusion_lr)

write_figure(f, 'lr_initial_double_simple.png', caption='logistic regression, last app classified')
f.write('\n\nResult of the KFold validation with 10 bins:')

result_lr = model_selection.cross_val_score(lr, X_train, y_train, cv=kfold, scoring=scoring)
write_row(f, result_lr.transpose())
f.write('\nThe mean is : %f' % result_lr.mean())

# Linear SVM

f.write('\\section{Linear Support Vector Machine results:} \n')
f.write('Confusion matrix with number of sample and with normalization:')

lsvm.fit(X_train, y_train)
y_pred_lsvm = lsvm.predict(X_test)
confusion_lsvm = confusion_matrix(y_test, y_pred_lsvm)

plt.figure()
plot_confusion_matrix(confusion_lsvm, simple_class_name, title='SVM confusion matrix', normalize=True)
plt.savefig('D:\\Top Secret\\Università\\progetto_ricerca\\script\\report\\lsvm_initial_double_simple.png')
plt.close()

confusion_lsvm = np.column_stack((np.array(simple_class_name), confusion_lsvm))
confusion_lsvm = np.append(simple_class_name_up, confusion_lsvm, axis=0)
write_matrix(f, confusion_lsvm)

write_figure(f, 'lsvm_initial_double_simple.png', caption='linear SVM, last app classified')
f.write('\n\nResult of the KFold validation with 10 bins:')

result_lsvm = model_selection.cross_val_score(lsvm, X_train, y_train, cv=kfold, scoring=scoring)
write_row(f, result_lsvm.transpose())
f.write('\nThe mean is : %f' % result_lsvm.mean())

# random forest

f.write('\\section{Random forest results:} \n')
f.write('Confusion matrix with number of sample and with normalization:')

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
confusion_rf = confusion_matrix(y_test, y_pred_rf)

plt.figure()
plot_confusion_matrix(confusion_rf, simple_class_name, title='Random Forest confusion matrix', normalize=True)
plt.savefig('D:\\Top Secret\\Università\\progetto_ricerca\\script\\report\\rf_initial_double_simple.png')
plt.close()

confusion_rf = np.column_stack((np.array(simple_class_name), confusion_rf))
confusion_rf = np.append(simple_class_name_up, confusion_rf, axis=0)
write_matrix(f, confusion_rf)

write_figure(f, 'rf_initial_double_simple.png', caption='random forest, last app classified')
f.write('\n\nResult of the KFold validation with 10 bins:')

result_rf = model_selection.cross_val_score(rf, X_train, y_train, cv=kfold, scoring=scoring)
write_row(f, result_rf.transpose())
f.write('\nThe mean is : %f' % result_rf.mean())

#######################################################################################à

# classification combined with first and last app


Y = dataset[:, :1]
Y = np.array(Y.ravel()).astype(int)

f.write('\n\n\\chapter{Double Scenario Classification of the first and last shared app, KFold Validation}\n\n')

complete_class = ['11', '12', '13', '21', '22', '23', '31', '32', '33', '4']
complete_class_num = [11, 12, 13, 21, 22, 23, 31, 32, 33, 4]
complete_class_name = ['messenger\_mess', 'messenger\_tele', 'messenger\_what', 'telegram\_mess', 'telegram\_tele',
                       'telegram\_what', 'whatsapp\_mess', 'whatsapp\_tele', 'whatsapp\_what', 'original']
complete_class_name_up = ['', 'messenger_mess', 'messenger_tele', 'messenger_what', 'telegram_mess', 'telegram_tele',
                          'telegram_what', 'whatsapp_mess', 'whatsapp_tele', 'whatsapp_what', 'original']
complete_class_name_short = ['', 'm\_m', 'm\_t', 'm\_w', 't\_m', 't\_t', 't\_w', 'w\_m', 'w\_t', 'w\_w', 'original']
complete_class_name_up = np.array(complete_class_name_up)
complete_class_name_up = np.reshape(complete_class_name_up, [1, 11])
complete_class_name_short = np.array(complete_class_name_short)
complete_class_name_short = np.reshape(complete_class_name_short, [1, 11])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.7, random_state=0)

scoring = 'accuracy'
kfold = model_selection.KFold(n_splits=10, random_state=7)
f.write(
    'Starting with fitting randomly the classifiers, there are some statistics of the data used for the first test: \\\\')

stat = [
    ['', 'count train', 'count test'],
    ['messenger\_mess', list(y_train).count(11), list(y_test).count(11)],
    ['messenger\_tele', list(y_train).count(12), list(y_test).count(12)],
    ['messenger\_what', list(y_train).count(13), list(y_test).count(13)],
    ['telegram\_mess', list(y_train).count(21), list(y_test).count(21)],
    ['telegram\_tele', list(y_train).count(22), list(y_test).count(22)],
    ['telegram\_what', list(y_train).count(23), list(y_test).count(23)],
    ['whatsapp\_mess', list(y_train).count(31), list(y_test).count(31)],
    ['whatsapp\_tele', list(y_train).count(32), list(y_test).count(32)],
    ['whatsapp\_what', list(y_train).count(33), list(y_test).count(33)],
    ['original', list(y_train).count(4), list(y_test).count(4)]]

write_matrix(f, np.array(stat))

# logistic regression

f.write('\\section{Logistic regression results:} \n')
f.write('Confusion matrix with number of sample and with normalization:')

lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
confusion_lr = confusion_matrix(y_test, y_pred_lr, labels=complete_class_num)

plt.figure()
plot_confusion_matrix(confusion_lr, complete_class_name, title='logistic confusion matrix, double app', normalize=True)
plt.savefig('D:\\Top Secret\\Università\\progetto_ricerca\\script\\report\\lr_initial_double_complete.png')
plt.close()

confusion_lr = np.column_stack((np.array(complete_class_name), confusion_lr))
confusion_lr = np.append(complete_class_name_short, confusion_lr, axis=0)
write_matrix(f, confusion_lr)

write_figure(f, 'lr_initial_double_complete.png', caption='logistic regression, last app classified')
f.write('\n\nResult of the KFold validation with 10 bins:')

result_lr = model_selection.cross_val_score(lr, X_train, y_train, cv=kfold, scoring=scoring)
write_row(f, result_lr.transpose())
f.write('\nThe mean is : %f' % result_lr.mean())

# Linear SVM

f.write('\\section{Linear Support Vector Machine results:} \n')
f.write('Confusion matrix with number of sample and with normalization:')

lsvm.fit(X_train, y_train)
y_pred_lsvm = lsvm.predict(X_test)
confusion_lsvm = confusion_matrix(y_test, y_pred_lsvm, labels=complete_class_num)

plt.figure()
plot_confusion_matrix(confusion_lsvm, complete_class_name, title='SVM confusion matrix', normalize=True)
plt.savefig('D:\\Top Secret\\Università\\progetto_ricerca\\script\\report\\lsvm_initial_double_complete.png')
plt.close()

confusion_lsvm = np.column_stack((np.array(complete_class_name), confusion_lsvm))
confusion_lsvm = np.append(complete_class_name_short, confusion_lsvm, axis=0)
write_matrix(f, confusion_lsvm)

write_figure(f, 'lsvm_initial_double_complete.png', caption='linear SVM, last app classified')
f.write('\n\nResult of the KFold validation with 10 bins:')

result_lsvm = model_selection.cross_val_score(lsvm, X_train, y_train, cv=kfold, scoring=scoring)
write_row(f, result_lsvm.transpose())
f.write('\nThe mean is : %f' % result_lsvm.mean())

# random forest

f.write('\\section{Random forest results:} \n')
f.write('Confusion matrix with number of sample and with normalization:')

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
confusion_rf = confusion_matrix(y_test, y_pred_rf, labels=complete_class_num)

plt.figure()
plot_confusion_matrix(confusion_rf, complete_class_name, title='Random Forest confusion matrix', normalize=True)
plt.savefig('D:\\Top Secret\\Università\\progetto_ricerca\\script\\report\\rf_initial_double_complete.png')
plt.close()

confusion_rf = np.column_stack((np.array(complete_class_name), confusion_rf))
confusion_rf = np.append(complete_class_name_short, confusion_rf, axis=0)
write_matrix(f, confusion_rf)

write_figure(f, 'rf_initial_double_complete.png', caption='random forest, last app classified')
f.write('\n\nResult of the KFold validation with 10 bins:')

result_rf = model_selection.cross_val_score(rf, X_train, y_train, cv=kfold, scoring=scoring)
write_row(f, result_rf.transpose())
f.write('\nThe mean is : %f' % result_rf.mean())

'''

#########################################################
#classification with different features, the dct is taken from the other channel not for the luminance

f.write('\n\n\\chapter{Double Scenario Classification of the first and last shared app whith cannel CbCr, KFold Validation}\n\n')

#load all data in a time
a_add = spy.loadmat('D:\\Top Secret\\Università\\progetto_ricerca\\script\\output\\'+file_add)
a_dct = spy.loadmat('D:\\Top Secret\\Università\\progetto_ricerca\\script\\output\\'+file_dct_CbCr)
a_add_x = a_add['features_all']
a_dct_x = a_dct['features_all']
a = np.column_stack((a_add_x, a_dct_x))
target = a_add['target']
target = np.array(target.ravel()).astype(int)
dataset = np.column_stack((target, a))


Y = dataset[:, :1]
Y = np.array(Y.ravel()).astype(int)
X = dataset[:, 1:]
X = preprocessing.scale(X)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=0)


scoring = 'accuracy'
kfold = model_selection.KFold(n_splits=10, random_state=7)
f.write('Starting with fitting randomly the classifiers, there are some statistics of the data used for the first test: \\\\')

stat = [
    ['','count train','count test'], 
    ['messenger\_mess', list(y_train).count(11),list(y_test).count(11)],
    ['messenger\_tele', list(y_train).count(12),list(y_test).count(12)],
    ['messenger\_what', list(y_train).count(13),list(y_test).count(13)], 
    ['telegram\_mess',list(y_train).count(21), list(y_test).count(21)],
    ['telegram\_tele',list(y_train).count(22), list(y_test).count(22)],
    ['telegram\_what',list(y_train).count(23), list(y_test).count(23)], 
    ['whatsapp\_mess', list(y_train).count(31), list(y_test).count(31)],
    ['whatsapp\_tele', list(y_train).count(32), list(y_test).count(32)],
    ['whatsapp\_what', list(y_train).count(33), list(y_test).count(33)], 
    ['original', list(y_train).count(4) , list(y_test).count(4)]]

write_matrix(f, np.array(stat))


#logistic regression

f.write('\\section{Logistic regression results:} \n')
f.write('Confusion matrix with number of sample and with normalization:')


lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
confusion_lr = confusion_matrix(y_test, y_pred_lr, labels=complete_class_num)


plt.figure()
plot_confusion_matrix(confusion_lr, complete_class_name, title='logistic confusion matrix, double app', normalize=True)
plt.savefig('D:\\Top Secret\\Università\\progetto_ricerca\\script\\report\\lr_initial_double_complete1.png')
plt.close()

confusion_lr= np.column_stack((np.array(complete_class_name), confusion_lr))
confusion_lr = np.append(complete_class_name_short, confusion_lr, axis=0)
write_matrix(f, confusion_lr)

write_figure(f, 'lr_initial_double_complete1.png', caption='logistic regression, last app classified')
f.write('\n\nResult of the KFold validation with 10 bins:')

result_lr = model_selection.cross_val_score(lr, X_train, y_train, cv=kfold, scoring=scoring)
write_row(f, result_lr.transpose())
f.write('\nThe mean is : %f' %result_lr.mean())



#Linear SVM

f.write('\\section{Linear Support Vector Machine results:} \n')
f.write('Confusion matrix with number of sample and with normalization:')

lsvm.fit(X_train, y_train)
y_pred_lsvm = lsvm.predict(X_test)
confusion_lsvm = confusion_matrix(y_test, y_pred_lsvm, labels=complete_class_num)


plt.figure()
plot_confusion_matrix(confusion_lsvm, complete_class_name, title='SVM confusion matrix', normalize=True)
plt.savefig('D:\\Top Secret\\Università\\progetto_ricerca\\script\\report\\lsvm_initial_double_complete1.png')
plt.close()

confusion_lsvm= np.column_stack((np.array(complete_class_name), confusion_lsvm))
confusion_lsvm = np.append(complete_class_name_short, confusion_lsvm, axis=0)
write_matrix(f, confusion_lsvm)

write_figure(f, 'lsvm_initial_double_complete1.png', caption='linear SVM, last app classified')
f.write('\n\nResult of the KFold validation with 10 bins:')


result_lsvm = model_selection.cross_val_score(lsvm, X_train, y_train, cv=kfold, scoring=scoring)
write_row(f, result_lsvm.transpose())
f.write('\nThe mean is : %f' %result_lsvm.mean())




#random forest

f.write('\\section{Random forest results:} \n')
f.write('Confusion matrix with number of sample and with normalization:')

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
confusion_rf = confusion_matrix(y_test, y_pred_rf, labels=complete_class_num)

plt.figure()
plot_confusion_matrix(confusion_rf, complete_class_name, title='Random Forest confusion matrix', normalize=True)
plt.savefig('D:\\Top Secret\\Università\\progetto_ricerca\\script\\report\\rf_initial_double_complete1.png')
plt.close()

confusion_rf= np.column_stack((np.array(complete_class_name), confusion_rf))
confusion_rf = np.append(complete_class_name_short, confusion_rf, axis=0)
write_matrix(f, confusion_rf)

write_figure(f, 'rf_initial_double_complete1.png', caption='random forest, last app classified')
f.write('\n\nResult of the KFold validation with 10 bins:')

result_rf = model_selection.cross_val_score(rf, X_train, y_train, cv=kfold, scoring=scoring)
write_row(f, result_rf.transpose())
f.write('\nThe mean is : %f' %result_rf.mean())




#########################################################
#classification with different features, the dct is taken from the other channel not for the luminance

f.write('\n\n\\chapter{Double Scenario Classification of the first and last shared app whith cannel CbCr, KFold Validation}\n\n')

#load all data in a time
a_add = spy.loadmat('D:\\Top Secret\\Università\\progetto_ricerca\\script\\output\\'+file_add)
a_dct = spy.loadmat('D:\\Top Secret\\Università\\progetto_ricerca\\script\\output\\'+file_dct_Y)
a_dct2 = spy.loadmat('D:\\Top Secret\\Università\\progetto_ricerca\\script\\output\\'+file_dct_CbCr)
a_add_x = a_add['features_all']
a_dct_x = a_dct['features_all']
a_dct2_x = a_dct2['features_all']
a = np.column_stack((a_add_x, a_dct_x, a_dct2_x))
target = a_add['target']
target = np.array(target.ravel()).astype(int)
dataset = np.column_stack((target, a))


Y = dataset[:, :1]
Y = np.array(Y.ravel()).astype(int)
X = dataset[:, 1:]
X = preprocessing.scale(X)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=0)




scoring = 'accuracy'
kfold = model_selection.KFold(n_splits=10, random_state=7)
f.write('Starting with fitting randomly the classifiers, there are some statistics of the data used for the first test: \\\\')

stat = [
    ['','count train','count test'], 
    ['messenger\_mess', list(y_train).count(11),list(y_test).count(11)],
    ['messenger\_tele', list(y_train).count(12),list(y_test).count(12)],
    ['messenger\_what', list(y_train).count(13),list(y_test).count(13)], 
    ['telegram\_mess',list(y_train).count(21), list(y_test).count(21)],
    ['telegram\_tele',list(y_train).count(22), list(y_test).count(22)],
    ['telegram\_what',list(y_train).count(23), list(y_test).count(23)], 
    ['whatsapp\_mess', list(y_train).count(31), list(y_test).count(31)],
    ['whatsapp\_tele', list(y_train).count(32), list(y_test).count(32)],
    ['whatsapp\_what', list(y_train).count(33), list(y_test).count(33)], 
    ['original', list(y_train).count(4) , list(y_test).count(4)]]

write_matrix(f, np.array(stat))


#logistic regression

f.write('\\section{Logistic regression results:} \n')
f.write('Confusion matrix with number of sample and with normalization:')


lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
confusion_lr = confusion_matrix(y_test, y_pred_lr, labels=complete_class_num)


plt.figure()
plot_confusion_matrix(confusion_lr, complete_class_name, title='logistic confusion matrix, double app', normalize=True)
plt.savefig('D:\\Top Secret\\Università\\progetto_ricerca\\script\\report\\lr_initial_double_complete2.png')
plt.close()

confusion_lr= np.column_stack((np.array(complete_class_name), confusion_lr))
confusion_lr = np.append(complete_class_name_short, confusion_lr, axis=0)
write_matrix(f, confusion_lr)

write_figure(f, 'lr_initial_double_complete2.png', caption='logistic regression, last app classified')
f.write('\n\nResult of the KFold validation with 10 bins:')

result_lr = model_selection.cross_val_score(lr, X_train, y_train, cv=kfold, scoring=scoring)
write_row(f, result_lr.transpose())
f.write('\nThe mean is : %f' %result_lr.mean())



#Linear SVM

f.write('\\section{Linear Support Vector Machine results:} \n')
f.write('Confusion matrix with number of sample and with normalization:')

lsvm.fit(X_train, y_train)
y_pred_lsvm = lsvm.predict(X_test)
confusion_lsvm = confusion_matrix(y_test, y_pred_lsvm, labels=complete_class_num)


plt.figure()
plot_confusion_matrix(confusion_lsvm, complete_class_name, title='SVM confusion matrix', normalize=True)
plt.savefig('D:\\Top Secret\\Università\\progetto_ricerca\\script\\report\\lsvm_initial_double_complete2.png')
plt.close()

confusion_lsvm= np.column_stack((np.array(complete_class_name), confusion_lsvm))
confusion_lsvm = np.append(complete_class_name_short, confusion_lsvm, axis=0)
write_matrix(f, confusion_lsvm)

write_figure(f, 'lsvm_initial_double_complete2.png', caption='linear SVM, last app classified')
f.write('\n\nResult of the KFold validation with 10 bins:')


result_lsvm = model_selection.cross_val_score(lsvm, X_train, y_train, cv=kfold, scoring=scoring)
write_row(f, result_lsvm.transpose())
f.write('\nThe mean is : %f' %result_lsvm.mean())




#random forest

f.write('\\section{Random forest results:} \n')
f.write('Confusion matrix with number of sample and with normalization:')

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
confusion_rf = confusion_matrix(y_test, y_pred_rf, labels=complete_class_num)

plt.figure()
plot_confusion_matrix(confusion_rf, complete_class_name, title='Random Forest confusion matrix', normalize=True)
plt.savefig('D:\\Top Secret\\Università\\progetto_ricerca\\script\\report\\rf_initial_double_complete2.png')
plt.close()

confusion_rf= np.column_stack((np.array(complete_class_name), confusion_rf))
confusion_rf = np.append(complete_class_name_short, confusion_rf, axis=0)
write_matrix(f, confusion_rf)

write_figure(f, 'rf_initial_double_complete2.png', caption='random forest, last app classified')
f.write('\n\nResult of the KFold validation with 10 bins:')

result_rf = model_selection.cross_val_score(rf, X_train, y_train, cv=kfold, scoring=scoring)
write_row(f, result_rf.transpose())
f.write('\nThe mean is : %f' %result_rf.mean())

'''

f.close()

'''

#################################################################################################
#here start the part where we circulary shift the dataset 
f.write('\n')
f.write('\n\\chapter{Double Scenario Classification of the last shared app, Circularly Validation}\n\n')
f.write('Here was used the same dataset as before but the training used a 0.3 of the dataset, and it is shifted circulary to cover all the dataset.')
f.write('Here is the table of all steps calculated \\\\')


X_m = X[:1050, :]
X_t = X[1050:2100, :]
X_w = X[2100:3150, :]
X_o = X[3150:, :]

Y_m = Y[:1050]
Y_t = Y[1050:2100]
Y_w = Y[2100:3150]
Y_o = Y[3150:]


length = len(X_m)
size = int(length * 0.3)
shift = 5
index = int(length / shift)
#index = 5
M_index = 0
T_index = 50
W_index = 100
O_index = 150

total_precision_lr = 0
total_precision_lsvm = 0
total_precision_rf = 0
total_cf_lr = [[]]
total_cf_lsvm = [[]]
total_cf_rf = [[]]

precisions = [['step', 'logistic', 'linear SVM', 'random fo.']]

#deleteting figure 9 because is fake
Y_m = np.delete(Y_m, (8), axis=0)
X_m = np.delete(X_m, (8), axis=0)

for x in range (0, index):
    print('make %d cicle' %x)
    M_index = (M_index + shift) % length
    T_index = (T_index + shift) % length
    W_index = (W_index + shift) % length
    O_index = (O_index + shift) % length
    

    if M_index + size < len(X_m):
        M_x_train = X_m[M_index:M_index+size, :]
        M_x_test = np.concatenate((X_m[:M_index, :], X_m[M_index + size : , :]))
        M_y_train = Y_m[:105, :]
        M_y_test = Y_m[105:, :]
    else:
        left = (M_index+size)-len(X_m)
        M_x_train = np.concatenate((X_m[M_index:, :], X_m[:left, :]))
        M_x_test = X_m[left:M_index, :]
        M_y_train = Y_m[:105, :]
        M_y_test = Y_m[105:, :]

    if T_index + size < length:
        T_x_train = X_t[T_index:T_index+size, :]
        T_x_test = np.concatenate((X_t[:T_index, :], X_t[T_index + size : , :]))
        T_y_train = Y_t[:105, :]
        T_y_test = Y_t[105:, :]
    else:
        left = (T_index+size) - length
        T_x_train = np.concatenate((X_t[T_index:, :], X_t[:left, :]))
        T_x_test = X_t[left:T_index, :]
        T_y_train = Y_t[:105, :]
        T_y_test = Y_t[105:, :]

    if W_index + size < length:
        W_x_train = X_w[W_index:W_index+size, :]
        W_x_test = np.concatenate((X_w[:W_index, :], X_w[W_index + size : , :]))
        W_y_train = Y_w[:105, :]
        W_y_test = Y_w[105:, :]
    else:
        left = (W_index + size) - length
        W_x_train = np.concatenate((X_w[W_index:, :], X_w[:left, :]))
        W_x_test = X_w[left:W_index, :]
        W_y_train = Y_w[:105, :]
        W_y_test = Y_w[105:, :]

    if O_index + size < length:
        O_x_train = X_o[O_index:O_index+size, :]
        O_x_test = np.concatenate((X_o[:O_index, :], X_o[O_index + size : , :]))
        O_y_train = Y_o[:105, :]
        O_y_test = Y_o[105:, :]
    else:
        left = (O_index + size) - length
        O_x_train = np.concatenate((X_o[O_index:, :], X_o[:left, :]))
        O_x_test = X_o[left:O_index, :]
        O_y_train = Y_o[:105, :]
        O_y_test = Y_o[105:, :]

    
    
    X_train = np.concatenate((M_x_train, T_x_train, W_x_train, O_x_train))
    Y_train = np.concatenate((M_y_train, T_y_train, W_y_train, O_y_train))
    Y_train = np.array(Y_train.ravel()).astype(int)
    X_test = np.concatenate((M_x_test, T_x_test, W_x_test, O_x_test))
    Y_test = np.concatenate((M_y_test, T_y_test, W_y_test, O_y_test))
    Y_test = np.array(Y_test.ravel()).astype(int)


    lr.fit(X_train, Y_train)
    lsvm.fit(X_train, Y_train)
    rf.fit(X_train, Y_train)

    
    pr_lr = lr.predict(X_test)
    cf_lr = confusion_matrix(Y_test, pr_lr)
    if x == 0 :
        total_cf_lr = cf_lr
    else:
        total_cf_lr = total_cf_lr + cf_lr
    a_precision_lr = precision_score(Y_test, pr_lr, average='macro')
    total_precision_lr = total_precision_lr + a_precision_lr
    

    pr_lsvm = lsvm.predict(X_test)
    cf_lsvm = confusion_matrix(Y_test, pr_lsvm)
    if x == 0:
        total_cf_lsvm = cf_lsvm
    else:
        total_cf_lsvm = total_cf_lsvm + cf_lsvm
    a_precision_lsvm = precision_score(Y_test, pr_lsvm, average='macro')
    total_precision_lsvm = total_precision_lsvm + a_precision_lsvm
    

    pr_rf = rf.predict(X_test)
    cf_rf = confusion_matrix(Y_test, pr_rf)
    if x == 0:
        total_cf_rf = cf_rf
    else:
        total_cf_rf = total_cf_rf + cf_rf
    a_precision_rf = precision_score(Y_test, pr_rf, average='macro')
    total_precision_rf = total_precision_rf + a_precision_rf
    

    precisions.append([x, a_precision_lr, a_precision_lsvm, a_precision_rf])
    
write_log_table(f, np.array(precisions))

lr_precision = total_precision_lr / index
lsvm_precision = total_precision_lsvm / index
rf_precision = total_precision_rf / index

aver = [['logistic r.','linear SVM','random f.'], 
        [lr_precision, lsvm_precision, rf_precision] ]

f.write('Average of all steps: \n')
write_matrix(f, np.array(aver))

f.write('Confusion matrix estimated on overall tests: \n')

#total confusion matrix plot
plt.figure()
plot_confusion_matrix(total_cf_lr, ['messenger','telegram','whatsapp','original'], title='logistic regression confusion matrix', normalize=True)
plt.savefig('D:\\Top Secret\\Università\\progetto_ricerca\\script\\report\\logistic_total.png')
plt.close()
write_figure(f, 'logistic_total.png', caption='logistic regression')

plt.figure()
plot_confusion_matrix(total_cf_lsvm, ['messenger','telegram','whatsapp','original'], title='linear suppurt vector machine confusion matrix', normalize=True)
plt.savefig('D:\\Top Secret\\Università\\progetto_ricerca\\script\\report\\lsvm_total.png')
plt.close()
write_figure(f, 'lsvm_total.png', caption='linear SVM')


plt.figure()
plot_confusion_matrix(total_cf_rf, ['messenger','telegram','whatsapp','original'], title='random forest confusion matrix', normalize=True)
plt.savefig('D:\\Top Secret\\Università\\progetto_ricerca\\script\\report\\random_total.png')
plt.close()
write_figure(f, 'random_total.png', caption='random forest')









































































'''

'''
lsvm = svm.LinearSVC(C=0.2)

#Linear SVM

f.write('\\section{Linear Support Vector Machine results:} \n')
f.write('Confusion matrix with number of sample and with normalization:')

lsvm.fit(X_train, y_train)
y_pred_lsvm = lsvm.predict(X_test)
confusion_lsvm = confusion_matrix(y_test, y_pred_lsvm)

plt.figure()
plot_confusion_matrix(confusion_lsvm, complete_class_name, title='SVM confusion matrix', normalize=True)
plt.savefig('D:\\Top Secret\\Università\\progetto_ricerca\\script\\report\\lsvm_initial_double_complete1.png')
plt.close()

confusion_lsvm= np.column_stack((np.array(complete_class_name), confusion_lsvm))
confusion_lsvm = np.append(complete_class_name_short, confusion_lsvm, axis=0)
write_matrix(f, confusion_lsvm)

write_figure(f, 'lsvm_initial_double_complete1.png', caption='linear SVM, last app classified')
f.write('\n\nResult of the KFold validation with 10 bins:')


result_lsvm = model_selection.cross_val_score(lsvm, X_train, y_train, cv=kfold, scoring=scoring)
write_row(f, result_lsvm.transpose())
f.write('\nThe mean is : %f' %result_lsvm.mean())


lsvm = svm.LinearSVC(C=0.01)

#Linear SVM

f.write('\\section{Linear Support Vector Machine results:} \n')
f.write('Confusion matrix with number of sample and with normalization:')

lsvm.fit(X_train, y_train)
y_pred_lsvm = lsvm.predict(X_test)
confusion_lsvm = confusion_matrix(y_test, y_pred_lsvm)

plt.figure()
plot_confusion_matrix(confusion_lsvm, complete_class_name, title='SVM confusion matrix', normalize=True)
plt.savefig('D:\\Top Secret\\Università\\progetto_ricerca\\script\\report\\lsvm_initial_double_complete2.png')
plt.close()

confusion_lsvm= np.column_stack((np.array(complete_class_name), confusion_lsvm))
confusion_lsvm = np.append(complete_class_name_short, confusion_lsvm, axis=0)
write_matrix(f, confusion_lsvm)

write_figure(f, 'lsvm_initial_double_complete2.png', caption='linear SVM, last app classified')
f.write('\n\nResult of the KFold validation with 10 bins:')


result_lsvm = model_selection.cross_val_score(lsvm, X_train, y_train, cv=kfold, scoring=scoring)
write_row(f, result_lsvm.transpose())
f.write('\nThe mean is : %f' %result_lsvm.mean())

lsvm = svm.LinearSVC(C=0.001)


#Linear SVM

f.write('\\section{Linear Support Vector Machine results:} \n')
f.write('Confusion matrix with number of sample and with normalization:')

lsvm.fit(X_train, y_train)
y_pred_lsvm = lsvm.predict(X_test)
confusion_lsvm = confusion_matrix(y_test, y_pred_lsvm)

plt.figure()
plot_confusion_matrix(confusion_lsvm, complete_class_name, title='SVM confusion matrix', normalize=True)
plt.savefig('D:\\Top Secret\\Università\\progetto_ricerca\\script\\report\\lsvm_initial_double_complete3.png')
plt.close()

confusion_lsvm= np.column_stack((np.array(complete_class_name), confusion_lsvm))
confusion_lsvm = np.append(complete_class_name_short, confusion_lsvm, axis=0)
write_matrix(f, confusion_lsvm)

write_figure(f, 'lsvm_initial_double_complete3.png', caption='linear SVM, last app classified')
f.write('\n\nResult of the KFold validation with 10 bins:')


result_lsvm = model_selection.cross_val_score(lsvm, X_train, y_train, cv=kfold, scoring=scoring)
write_row(f, result_lsvm.transpose())
f.write('\nThe mean is : %f' %result_lsvm.mean())


lsvm = svm.LinearSVC(C=0.01, penalty='l1', dual=False,)


#Linear SVM

f.write('\\section{Linear Support Vector Machine results:} \n')
f.write('Confusion matrix with number of sample and with normalization:')

lsvm.fit(X_train, y_train)
y_pred_lsvm = lsvm.predict(X_test)
confusion_lsvm = confusion_matrix(y_test, y_pred_lsvm)

plt.figure()
plot_confusion_matrix(confusion_lsvm, complete_class_name, title='SVM confusion matrix', normalize=True)
plt.savefig('D:\\Top Secret\\Università\\progetto_ricerca\\script\\report\\lsvm_initial_double_complete4.png')
plt.close()

confusion_lsvm= np.column_stack((np.array(complete_class_name), confusion_lsvm))
confusion_lsvm = np.append(complete_class_name_short, confusion_lsvm, axis=0)
write_matrix(f, confusion_lsvm)

write_figure(f, 'lsvm_initial_double_complete4.png', caption='linear SVM, last app classified')
f.write('\n\nResult of the KFold validation with 10 bins:')


result_lsvm = model_selection.cross_val_score(lsvm, X_train, y_train, cv=kfold, scoring=scoring)
write_row(f, result_lsvm.transpose())
f.write('\nThe mean is : %f' %result_lsvm.mean())
'''
