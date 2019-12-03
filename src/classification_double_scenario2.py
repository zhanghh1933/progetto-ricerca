import getopt
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spy
import sys

from src.plot_confusion_matrix import plot_confusion_matrix, write_matrix, write_row, write_figure, write_log_table

from sklearn import datasets, linear_model, svm, model_selection, preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, classification_report
from sklearn.externals import joblib
import src.settings as sett

# getopt.getopt()


def double_scenario_classification(argv, new_method):
    file_add = 'addi_features_all_double.mat'
    file_dct_Y = 'deq_dct_coef_all_double.mat'

    file_add_single = 'addi_features_all.mat'
    file_dct_single_Y = 'deq_dct_coef_all.mat'

    folder_report = 'D:\\Top Secret\\Università\\progetto_ricerca\\scriptNew\\reportTest\\'

    parte1 = True
    parte2 = True
    parte3 = True
    parte4 = True

    #################################

    lr = linear_model.LogisticRegression()
    lsvm = svm.LinearSVC()
    rf = RandomForestClassifier(n_estimators=100)

    np.set_printoptions(precision=2)

    # load all data in a time
    a_add = spy.loadmat('D:\\Top Secret\\Università\\progetto_ricerca\\scriptNew\\outputNew\\' + file_add)
    a_dct = spy.loadmat('D:\\Top Secret\\Università\\progetto_ricerca\\scriptNew\\outputNew\\' + file_dct_Y)
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

    if parte1:
        file_name = 'report_double_parte1_.tex'

        f = open(folder_report + file_name, 'w')

        f.write('\\chapter{Double Scenario Classification of the last shared app, KFold Validation}')

        # X = dataset[:, :]
        X = np.column_stack((Y, X))

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
        real_classes = X_test[:, :]
        X_train = X_train[:, 1:]
        X_test = X_test[:, 1:]
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
        plt.savefig(folder_report + 'lr_initial_double_simple.png')
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
        plt.savefig(folder_report + 'lsvm_initial_double_simple.png')
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
        plt.savefig(folder_report + 'rf_initial_double_simple.png')
        plt.close()

        confusion_rf = np.column_stack((np.array(simple_class_name), confusion_rf))
        confusion_rf = np.append(simple_class_name_up, confusion_rf, axis=0)
        write_matrix(f, confusion_rf)

        write_figure(f, 'rf_initial_double_simple.png', caption='random forest, last app classified')
        f.write('\n\nResult of the KFold validation with 10 bins:')

        result_rf = model_selection.cross_val_score(rf, X_train, y_train, cv=kfold, scoring=scoring)
        write_row(f, result_rf.transpose())
        f.write('\nThe mean is : %f' % result_rf.mean())
        test_31 = real_classes[real_classes[:, 0] == 31]
        test_31 = test_31[:, 1:]
        test_32 = real_classes[real_classes[:, 0] == 32]
        test_32 = test_32[:, 1:]
        test_33 = real_classes[real_classes[:, 0] == 33]
        test_33 = test_33[:, 1:]
        p_31 = lr.predict(test_31)
        p_32 = lr.predict(test_32)
        p_33 = lr.predict(test_33)
        print('test 31, in order 1 2 3 4 classes')
        print(len(test_31))
        print(list(p_31).count(1))
        print(list(p_31).count(2))
        print(list(p_31).count(3))
        print(list(p_31).count(4))
        print('test 32, in order 1 2 3 4 classes')
        print(len(test_32))
        print(list(p_32).count(1))
        print(list(p_32).count(2))
        print(list(p_32).count(3))
        print(list(p_32).count(4))
        print('test 33, in order 1 2 3 4 classes')
        print(len(test_33))
        print(list(p_33).count(1))
        print(list(p_33).count(2))
        print(list(p_33).count(3))
        print(list(p_33).count(4))
        ##############################################
        # here start telegram
        test_21 = real_classes[real_classes[:, 0] == 21]
        test_21 = test_21[:, 1:]
        test_22 = real_classes[real_classes[:, 0] == 22]
        test_22 = test_22[:, 1:]
        test_23 = real_classes[real_classes[:, 0] == 23]
        test_23 = test_23[:, 1:]
        p_21 = lr.predict(test_21)
        p_22 = lr.predict(test_22)
        p_23 = lr.predict(test_23)
        print('test 21, in order 1 2 3 4 classes')
        print(len(test_21))
        print(list(p_21).count(1))
        print(list(p_21).count(2))
        print(list(p_21).count(3))
        print(list(p_21).count(4))
        print('test 22, in order 1 2 3 4 classes')
        print(len(test_22))
        print(list(p_22).count(1))
        print(list(p_22).count(2))
        print(list(p_22).count(3))
        print(list(p_22).count(4))
        print('test 23, in order 1 2 3 4 classes')
        print(len(test_23))
        print(list(p_23).count(1))
        print(list(p_23).count(2))
        print(list(p_23).count(3))
        print(list(p_23).count(4))
        f.close()

    ##################################################################################################################################################################

    if parte2:
        # classification combined with first and last app
        file_name2 = 'report_double_parte2_.tex'
        f = open(folder_report + file_name2, 'w')

        Y = dataset[:, :1]
        Y = np.array(Y.ravel()).astype(int)

        f.write('\n\n\\chapter{Double Scenario Classification of the first and last shared app, KFold Validation}\n\n')

        complete_class = ['11', '12', '13', '21', '22', '23', '31', '32', '33', '4']
        complete_class_num = [11, 12, 13, 21, 22, 23, 31, 32, 33, 4]
        complete_class_name = ['mess\_mess', 'tele\_mess', 'what\_mess',
                               'mess\_tele', 'tele\_tele', 'what\_tele',
                               'mess\_what', 'tele\_what', 'what\_what',
                               'original']
        complete_class_name_noLatex = ['mess_mess', 'tele_mess', 'what_mess',
                                       'mess_tele', 'tele_tele', 'what_tele',
                                       'mess_what', 'tele_what', 'what_what',
                                       'original']

        complete_class_name_up = ['', 'messenger_mess', 'messenger_tele', 'messenger_what',
                                  'telegram_mess', 'telegram_tele', 'telegram_what',
                                  'whatsapp_mess', 'whatsapp_tele', 'whatsapp_what',
                                  'original']
        complete_class_name_short = ['', 'm\_m', 'm\_t', 'm\_w',
                                     't\_m', 't\_t', 't\_w',
                                     'w\_m', 'w\_t', 'w\_w',
                                     'original']
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
            ['mess\_mess', list(y_train).count(11), list(y_test).count(11)],
            ['tele\_mess', list(y_train).count(12), list(y_test).count(12)],
            ['what\_mess', list(y_train).count(13), list(y_test).count(13)],
            ['mess\_tele', list(y_train).count(21), list(y_test).count(21)],
            ['tele\_tele', list(y_train).count(22), list(y_test).count(22)],
            ['what\_tele', list(y_train).count(23), list(y_test).count(23)],
            ['mess\_what', list(y_train).count(31), list(y_test).count(31)],
            ['tele\_what', list(y_train).count(32), list(y_test).count(32)],
            ['what\_what', list(y_train).count(33), list(y_test).count(33)],
            ['original', list(y_train).count(4), list(y_test).count(4)]]

        write_matrix(f, np.array(stat))

        # logistic regression

        f.write('\\section{Logistic regression results:} \n')
        f.write('Confusion matrix with number of sample and with normalization:')

        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)
        confusion_lr = confusion_matrix(y_test, y_pred_lr, labels=complete_class_num)

        plt.figure()
        plot_confusion_matrix(confusion_lr, complete_class_name_noLatex, title='logistic confusion matrix, double app',
                              normalize=True)
        plt.savefig(folder_report + 'lr_initial_double_complete.png')
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
        plot_confusion_matrix(confusion_lsvm, complete_class_name_noLatex, title='SVM confusion matrix', normalize=True)
        plt.savefig(folder_report + 'lsvm_initial_double_complete.png')
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
        plot_confusion_matrix(confusion_rf, complete_class_name_noLatex, title='Random Forest confusion matrix',
                              normalize=True)
        plt.savefig(folder_report + 'rf_initial_double_complete.png')
        plt.close()

        confusion_rf = np.column_stack((np.array(complete_class_name), confusion_rf))
        confusion_rf = np.append(complete_class_name_short, confusion_rf, axis=0)
        write_matrix(f, confusion_rf)

        write_figure(f, 'rf_initial_double_complete.png', caption='random forest, last app classified')
        f.write('\n\nResult of the KFold validation with 10 bins:')

        result_rf = model_selection.cross_val_score(rf, X_train, y_train, cv=kfold, scoring=scoring)
        write_row(f, result_rf.transpose())
        f.write('\nThe mean is : %f' % result_rf.mean())
        f.close()

    ##################################################################################################################################################################
    if parte3:
        file_name3 = 'report_double_parte3_.tex'
        f = open(folder_report + file_name3, 'w')
        f.write('\n\n\\chapter{Single and double scenario, KFold Validation}\n\n')
        a_add_single = spy.loadmat('D:\\Top Secret\\Università\\progetto_ricerca\\scriptNew\\outputNew\\' + file_add_single)
        a_dct_single = spy.loadmat(
            'D:\\Top Secret\\Università\\progetto_ricerca\\scriptNew\\outputNew\\' + file_dct_single_Y)
        a_add_x_single = a_add_single['features_all']
        a_dct_x_single = a_dct_single['features_all']
        a_single = np.column_stack((a_add_x_single, a_dct_x_single))
        target_single = a_add_single['target']
        target_single = np.array(target_single.ravel()).astype(int)
        dataset_single = np.column_stack((target_single, a_single))
        dataset_single = np.append(dataset_single[0:350, :], dataset_single[700:, :], axis=0)
        dataset_final = np.append(dataset, dataset_single, axis=0)
        dataset_final[1400:1750, :] = 2
        dataset_final[2450:2800, :] = 2
        dataset_final[2800:3150, :] = 3
        Y = dataset_final[:, :1]
        Y = np.array(Y.ravel()).astype(int)
        X = dataset_final[:, 1:]
        X = preprocessing.scale(X)

        complete_class_num = [1, 2, 3, 11, 12, 13, 21, 23, 31, 4]
        # complete_class_name = ['1', '2', '3', '11', '12', '13', '21', '23', '31', '4']

        complete_class_name = ['mess', 'tele', 'what',
                               'mess\_mess', 'tele\_mess', 'what\_mess',
                               'mess\_tele', 'what\_tele',
                               'mess\_what',
                               'original']
        complete_class_name_noLatex = ['mess', 'tele', 'what',
                                       'mess_mess', 'tele_mess', 'what_mess',
                                       'mess_tele', 'what_tele',
                                       'mess_what',
                                       'original']
        complete_class_name_short = ['', 'm', 't', 'w',
                                     'm\_m', 'm\_t', 'm\_w',
                                     't\_m', 't\_w',
                                     'w\_m',
                                     'original']
        complete_class_name_short = np.array(complete_class_name_short)
        complete_class_name_short = np.reshape(complete_class_name_short, [1, 11])

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.7, random_state=0)
        scoring = 'accuracy'
        kfold = model_selection.KFold(n_splits=10, random_state=7)
        f.write(
            'Starting with fitting randomly the classifiers, there are some statistics of the data used for the first test: \\\\')

        stat = [
            ['', 'count train', 'count test'],
            ['mess', list(y_train).count(1), list(y_test).count(1)],
            ['tele', list(y_train).count(2), list(y_test).count(2)],
            ['what', list(y_train).count(3), list(y_test).count(3)],
            ['mess\_mess', list(y_train).count(11), list(y_test).count(11)],
            ['tele\_mess', list(y_train).count(12), list(y_test).count(12)],
            ['what\_mess', list(y_train).count(13), list(y_test).count(13)],
            ['mess\_tele', list(y_train).count(21), list(y_test).count(21)],
            ['what\_tele', list(y_train).count(23), list(y_test).count(23)],
            ['mess\_what', list(y_train).count(31), list(y_test).count(31)],
            ['original', list(y_train).count(4), list(y_test).count(4)]]

        write_matrix(f, np.array(stat))

        # logistic regression
        f.write('\\section{Logistic regression results:} \n')
        f.write('Confusion matrix with number of sample and with normalization:')
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)
        confusion_lr = confusion_matrix(y_test, y_pred_lr, labels=complete_class_num)

        plt.figure()
        plot_confusion_matrix(confusion_lr, complete_class_name_noLatex, title='logistic confusion matrix', normalize=True)
        plt.savefig(folder_report + 'lr_initial_single_double_complete.png')
        plt.close()

        confusion_lr = np.column_stack((np.array(complete_class_name), confusion_lr))
        confusion_lr = np.append(complete_class_name_short, confusion_lr, axis=0)
        write_matrix(f, confusion_lr)

        write_figure(f, 'lr_initial_single_double_complete.png', caption='logistic regression, last app classified')
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
        plot_confusion_matrix(confusion_lsvm, complete_class_name_noLatex, title='SVM confusion matrix', normalize=True)
        plt.savefig(folder_report + 'lsvm_initial_single_double_complete.png')
        plt.close()

        confusion_lsvm = np.column_stack((np.array(complete_class_name), confusion_lsvm))
        confusion_lsvm = np.append(complete_class_name_short, confusion_lsvm, axis=0)
        write_matrix(f, confusion_lsvm)

        write_figure(f, 'lsvm_initial_single_double_complete.png', caption='linear SVM, last app classified')
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
        plot_confusion_matrix(confusion_rf, complete_class_name_noLatex, title='Random Forest confusion matrix',
                              normalize=True)
        plt.savefig(folder_report + 'rf_initial_single_double_complete.png')
        plt.close()
        confusion_rf = np.column_stack((np.array(complete_class_name), confusion_rf))
        confusion_rf = np.append(complete_class_name_short, confusion_rf, axis=0)
        write_matrix(f, confusion_rf)

        write_figure(f, 'rf_initial_single_double_complete.png', caption='random forest, last app classified')
        f.write('\n\nResult of the KFold validation with 10 bins:')

        result_rf = model_selection.cross_val_score(rf, X_train, y_train, cv=kfold, scoring=scoring)
        write_row(f, result_rf.transpose())
        f.write('\nThe mean is : %f' % result_rf.mean())
        '''
        #SVM
        sv = svm.SVC(C=0.0001, kernel='poly')
        sv.fit(X_train, y_train)
        y_pred_lsvm = sv.predict(X_test)
        confusion_lsvm = confusion_matrix(y_test, y_pred_lsvm, labels=complete_class_num)
        plt.figure()
        plot_confusion_matrix(confusion_lsvm, complete_class_name, title='SVM confusion matrix', normalize=True)
        #plt.savefig(folder_report + 'lsvm_initial_double_complete.png')
        plt.close()
        '''
        f.close()
        print('ciao')

    ##################################################################################################################################################################
    if parte4:
        file_name3 = 'report_double_parte4_.tex'
        f = open(folder_report + file_name3, 'w')
        f.write('\n\n\\chapter{Single and double scenario, KFold Validation}\n\n')
        a_add_single = spy.loadmat('D:\\Top Secret\\Università\\progetto_ricerca\\scriptNew\\outputNew\\' + file_add_single)
        a_dct_single = spy.loadmat(
            'D:\\Top Secret\\Università\\progetto_ricerca\\scriptNew\\outputNew\\' + file_dct_single_Y)
        a_add_x_single = a_add_single['features_all']
        a_dct_x_single = a_dct_single['features_all']
        a_single = np.column_stack((a_add_x_single, a_dct_x_single))
        target_single = a_add_single['target']
        target_single = np.array(target_single.ravel()).astype(int)
        dataset_single = np.column_stack((target_single, a_single))
        dataset_single = np.append(dataset_single[0:350, :], dataset_single[700:, :], axis=0)
        dataset_final = np.append(dataset, dataset_single, axis=0)
        dataset_final[1400:1750, :] = 2
        dataset_final[1050:1400, :] = 2
        dataset_final[2100:2450, :] = 3
        dataset_final[2450:2800, :] = 2
        dataset_final[2800:3150, :] = 3
        Y = dataset_final[:, :1]
        Y = np.array(Y.ravel()).astype(int)
        X = dataset_final[:, 1:]
        X = preprocessing.scale(X)

        complete_class_num = [1, 2, 3, 11, 12, 13, 23, 4]
        # complete_class_name = ['1', '2', '3', '11', '12', '13', '23', '4']

        complete_class_name = ['mess', 'tele', 'what',
                               'mess\_mess', 'tele\_mess', 'what\_mess',
                               'what\_tele',

                               'original']
        complete_class_name_noLatex = ['mess', 'tele', 'what',
                                       'mess_mess', 'tele_mess', 'what_mess',
                                       'what_tele',

                                       'original']
        complete_class_name_short = ['', 'm', 't', 'w',
                                     'm\_m', 'm\_t', 'm\_w',
                                     't\_w',

                                     'original']
        complete_class_name_short = np.array(complete_class_name_short)
        complete_class_name_short = np.reshape(complete_class_name_short, [1, 9])

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.7, random_state=0)
        scoring = 'accuracy'
        kfold = model_selection.KFold(n_splits=10, random_state=7)
        f.write(
            'Starting with fitting randomly the classifiers, there are some statistics of the data used for the first test: \\\\')

        stat = [
            ['', 'count train', 'count test'],
            ['mess', list(y_train).count(1), list(y_test).count(1)],
            ['tele', list(y_train).count(2), list(y_test).count(2)],
            ['what', list(y_train).count(3), list(y_test).count(3)],
            ['mess\_mess', list(y_train).count(11), list(y_test).count(11)],
            ['tele\_mess', list(y_train).count(12), list(y_test).count(12)],
            ['what\_mess', list(y_train).count(13), list(y_test).count(13)],
            ['mess\_tele', list(y_train).count(21), list(y_test).count(21)],
            ['what\_tele', list(y_train).count(23), list(y_test).count(23)],
            ['mess\_what', list(y_train).count(31), list(y_test).count(31)],
            ['original', list(y_train).count(4), list(y_test).count(4)]]

        write_matrix(f, np.array(stat))

        # logistic regression
        f.write('\\section{Logistic regression results:} \n')
        f.write('Confusion matrix with number of sample and with normalization:')
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)
        confusion_lr = confusion_matrix(y_test, y_pred_lr, labels=complete_class_num)

        plt.figure()
        plot_confusion_matrix(confusion_lr, complete_class_name_noLatex, title='logistic confusion matrix', normalize=True)
        plt.savefig(folder_report + 'lr_initial_single_double_complete_second_test.png')
        plt.close()

        confusion_lr = np.column_stack((np.array(complete_class_name), confusion_lr))
        confusion_lr = np.append(complete_class_name_short, confusion_lr, axis=0)
        write_matrix(f, confusion_lr)

        write_figure(f, 'lr_initial_single_double_complete_second_test.png',
                     caption='logistic regression, last app classified')
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
        plot_confusion_matrix(confusion_lsvm, complete_class_name_noLatex, title='SVM confusion matrix', normalize=True)
        plt.savefig(folder_report + 'lsvm_initial_single_double_complete_second_test.png')
        plt.close()

        confusion_lsvm = np.column_stack((np.array(complete_class_name), confusion_lsvm))
        confusion_lsvm = np.append(complete_class_name_short, confusion_lsvm, axis=0)
        write_matrix(f, confusion_lsvm)

        write_figure(f, 'lsvm_initial_single_double_complete_second_test.png', caption='linear SVM, last app classified')
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
        plot_confusion_matrix(confusion_rf, complete_class_name_noLatex, title='Random Forest confusion matrix',
                              normalize=True)
        plt.savefig(folder_report + 'rf_initial_single_double_complete_second_test.png')
        plt.close()
        confusion_rf = np.column_stack((np.array(complete_class_name), confusion_rf))
        confusion_rf = np.append(complete_class_name_short, confusion_rf, axis=0)
        write_matrix(f, confusion_rf)

        write_figure(f, 'rf_initial_single_double_complete_second_test.png', caption='random forest, last app classified')
        f.write('\n\nResult of the KFold validation with 10 bins:')

        result_rf = model_selection.cross_val_score(rf, X_train, y_train, cv=kfold, scoring=scoring)
        write_row(f, result_rf.transpose())
        f.write('\nThe mean is : %f' % result_rf.mean())

        f.close()
        print('ciao')



# punto corretto di inizio del programma per uno scrip python
if __name__ == "__main__":
   double_scenario_classification(sys.argv[1:])