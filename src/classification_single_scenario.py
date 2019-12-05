import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spy
import sys
import getopt

from plot_confusion_matrix import plot_confusion_matrix, write_matrix, write_row, write_figure, write_log_table
from settings import SETTINGS

from sklearn import linear_model, svm, preprocessing, model_selection
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score
import joblib


def single_scenario_computation(argv, new_method=False):
    lr = linear_model.LogisticRegression() # solver='liblinear', multi_class='ovr')
    lsvm = svm.LinearSVC(max_iter=SETTINGS['max_iter'])
    rf = RandomForestClassifier(n_estimators=SETTINGS['n_estimators'])

    if new_method:
        f = open(SETTINGS['report_single_new_method'], 'w')
    else:
        f = open(SETTINGS['report_single'], 'w')
    report_folder = SETTINGS['report_folder']
    report_image_folder = SETTINGS['report_image_folder']
    if new_method:
        report_image_folder += 'new_met_'

    output_class_folder = SETTINGS['output_class_folder']

    # load all data in a time
    a_add = spy.loadmat(SETTINGS['single_addi'])

    if new_method:
        a_dct = spy.loadmat(SETTINGS['single_dct_new'])
    else:
        a_dct = spy.loadmat(SETTINGS['single_dct'])

    a_add_x = a_add['features_all']
    a_dct_x = a_dct['features_all']

    a = np.column_stack((a_add_x, a_dct_x))
    a = preprocessing.scale(a)
    target = a_add['target']
    target = np.array(target.ravel()).astype(int)
    dataset = np.column_stack((target, a))

    X_m = dataset[:350, :]
    Y_m = X_m[:, :1]
    X_m = X_m[:, 1:]

    X_o = dataset[350:700, :]
    Y_o = X_o[:, :1]
    X_o = X_o[:, 1:]

    X_t = dataset[700:1050, :]
    Y_t = X_t[:, :1]
    X_t = X_t[:, 1:]

    X_w = dataset[1050:, :]
    Y_w = X_w[:, :1]
    X_w = X_w[:, 1:]

    dataset_shuflled = dataset
    dataset_shuflled = np.delete(dataset_shuflled, 8, axis=0)
    X_train, X_test, y_train, y_test = train_test_split(preprocessing.scale(dataset_shuflled[:, 1:]),
                                                        np.array(dataset_shuflled[:, :1].ravel()).astype(int),
                                                        test_size=0.3, random_state=0)
    scoring = 'accuracy'
    kfold = model_selection.KFold(n_splits=10, random_state=7)

    f.write('\\chapter{Single Scenario Classification, KFold Validation}')
    f.write(
        'Starting with fitting randomly the classifiers, there are some statistics of the data used for the first test: \\\\')

    classes = ['messenger', 'telegram', 'whatsapp', 'original']
    classes_up = ['', 'messenger', 'telegram', 'whatsapp', 'original']
    classes_up = np.array(classes_up)
    classes_up = np.reshape(classes_up, [1, 5])

    stat = [
        ['', 'count train', 'count test'],
        ['messenger', list(y_train).count(1), list(y_test).count(1)],
        ['telegram', list(y_train).count(2), list(y_test).count(2)],
        ['whatsapp', list(y_train).count(3), list(y_test).count(3)],
        ['original', list(y_train).count(4), list(y_test).count(4)]]

    write_matrix(f, np.array(stat))

    # logistic regression simple test
    print("logistic regression simple test")
    f.write('\\section{Logistic regression results:} \n')
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    confusion_lr = confusion_matrix(y_test, y_pred_lr)
    f.write('Confusion matrix with number of sample and with normalization:')

    plt.figure()
    plot_confusion_matrix(confusion_lr, classes, title='Logistic regression confusion matrix', normalize=True)
    plt.savefig(report_image_folder + 'lr_initial.png')
    plt.close()

    confusion_lr = np.column_stack((np.array(classes), confusion_lr))
    confusion_lr = np.append(classes_up, confusion_lr, axis=0)
    write_matrix(f, confusion_lr)

    write_figure(f, 'lr_initial.png', caption='logistic regression')
    f.write('\n\nResult of the KFold validation with 10 bins:')
    result_lr = model_selection.cross_val_score(lr, X_train, y_train, cv=kfold, scoring=scoring)
    write_row(f, result_lr.transpose())
    f.write('\nThe mean is : %f' % result_lr.mean())

    # lsvm
    print("lsvm regression simple test")
    f.write('\\section{Linear Support Vector Machine results:}')
    lsvm.fit(X_train, y_train)
    y_pred_lsvm = lsvm.predict(X_test)
    confusion_lsvm = confusion_matrix(y_test, y_pred_lsvm)
    f.write('Confusion matrix with number of sample and with normalization:')

    plt.figure()
    plot_confusion_matrix(confusion_lsvm, classes, title='Linear suppurt vector machine confusion matrix', normalize=True)
    plt.savefig(report_image_folder + 'lsvm_initial.png')
    plt.close()

    confusion_lsvm = np.column_stack((np.array(classes), confusion_lsvm))
    confusion_lsvm = np.append(classes_up, confusion_lsvm, axis=0)
    write_matrix(f, confusion_lsvm)

    write_figure(f, 'lsvm_initial.png', caption='linear SVM')

    result_lsvm = model_selection.cross_val_score(lsvm, X_train, y_train, cv=kfold, scoring=scoring)

    f.write('\n\nResult of the KFold validation with 10 bins:')
    write_row(f, result_lsvm.transpose())
    f.write('\nThe mean is : %f' % result_lsvm.mean())

    # random forest
    print("random forest regression simple test")
    f.write('\\section{Random forest results:}')
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    confusion_rf = confusion_matrix(y_test, y_pred_rf)

    f.write('Confusion matrix with number of sample and with normalization:')

    plt.figure()
    plot_confusion_matrix(confusion_rf, classes, title='Random forest confusion matrix', normalize=True)
    plt.savefig(report_image_folder + 'rf_initial.png')
    plt.close()

    confusion_rf = np.column_stack((np.array(classes), confusion_rf))
    confusion_rf = np.append(classes_up, confusion_rf, axis=0)
    write_matrix(f, confusion_rf)

    write_figure(f, 'rf_initial.png', caption='random forest')

    result_rf = model_selection.cross_val_score(rf, X_train, y_train, cv=kfold, scoring=scoring)

    f.write('\n\nResult of the KFold validation with 10 bins:')
    write_row(f, result_rf.transpose())
    f.write('\nThe mean is : %f' % result_rf.mean())

    f.write('\n')
    f.write('\n\\chapter{Single Scenario Classification, Circularly Validation}\n\n')
    f.write(
        'Here was used the same dataset as before but the training used a 0.3 of the dataset, and it is shifted circulary to cover all the dataset.')
    f.write('Here is the table of all steps calculated \\\\')

    length = len(X_m)
    size = int(length * 0.3)
    shift = 5
    index = int(length / shift)

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

    # deleteting figure 9 because is fake
    Y_m = np.delete(Y_m, 8, axis=0)
    X_m = np.delete(X_m, 8, axis=0)

    for x in range(0, index):
        print('make '+ str(x) +' cicle')
        M_index = (M_index + shift) % length
        T_index = (T_index + shift) % length
        W_index = (W_index + shift) % length
        O_index = (O_index + shift) % length

        if M_index + size < len(X_m):
            M_x_train = X_m[M_index:M_index + size, :]
            M_x_test = np.concatenate((X_m[:M_index, :], X_m[M_index + size:, :]))
            M_y_train = Y_m[:105, :]
            M_y_test = Y_m[105:, :]
        else:
            left = (M_index + size) - len(X_m)
            M_x_train = np.concatenate((X_m[M_index:, :], X_m[:left, :]))
            M_x_test = X_m[left:M_index, :]
            M_y_train = Y_m[:105, :]
            M_y_test = Y_m[105:, :]

        if T_index + size < length:
            T_x_train = X_t[T_index:T_index + size, :]
            T_x_test = np.concatenate((X_t[:T_index, :], X_t[T_index + size:, :]))
            T_y_train = Y_t[:105, :]
            T_y_test = Y_t[105:, :]
        else:
            left = (T_index + size) - length
            T_x_train = np.concatenate((X_t[T_index:, :], X_t[:left, :]))
            T_x_test = X_t[left:T_index, :]
            T_y_train = Y_t[:105, :]
            T_y_test = Y_t[105:, :]

        if W_index + size < length:
            W_x_train = X_w[W_index:W_index + size, :]
            W_x_test = np.concatenate((X_w[:W_index, :], X_w[W_index + size:, :]))
            W_y_train = Y_w[:105, :]
            W_y_test = Y_w[105:, :]
        else:
            left = (W_index + size) - length
            W_x_train = np.concatenate((X_w[W_index:, :], X_w[:left, :]))
            W_x_test = X_w[left:W_index, :]
            W_y_train = Y_w[:105, :]
            W_y_test = Y_w[105:, :]

        if O_index + size < length:
            O_x_train = X_o[O_index:O_index + size, :]
            O_x_test = np.concatenate((X_o[:O_index, :], X_o[O_index + size:, :]))
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

        print("-----log regression ")
        lr.fit(X_train, Y_train)
        print("-----lsvm ")
        lsvm.fit(X_train, Y_train)
        print("-----random for ")
        rf.fit(X_train, Y_train)

        pr_lr = lr.predict(X_test)
        cf_lr = confusion_matrix(Y_test, pr_lr)
        if x == 0:
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

    aver = [['logistic r.', 'linear SVM', 'random f.'],
            [lr_precision, lsvm_precision, rf_precision]]

    f.write('Average of all steps: \n')
    write_matrix(f, np.array(aver))

    f.write('Confusion matrix estimated on overall tests: \n')

    # total confusion matrix plot
    plt.figure()
    plot_confusion_matrix(total_cf_lr, ['messenger', 'telegram', 'whatsapp', 'original'],
                          title='logistic regression confusion matrix', normalize=True)

    plt.savefig(report_image_folder + 'logistic_total.png')
    plt.close()
    write_figure(f, 'logistic_total.png', caption='logistic regression')

    plt.figure()
    plot_confusion_matrix(total_cf_lsvm, ['messenger', 'telegram', 'whatsapp', 'original'],
                          title='linear suppurt vector machine confusion matrix', normalize=True)
    plt.savefig(report_image_folder + 'lsvm_total.png')
    plt.close()
    write_figure(f, 'lsvm_total.png', caption='linear SVM')

    plt.figure()
    plot_confusion_matrix(total_cf_rf, ['messenger', 'telegram', 'whatsapp', 'original'],
                          title='random forest confusion matrix', normalize=True)
    plt.savefig(report_image_folder + 'random_total.png')
    plt.close()
    write_figure(f, 'random_total.png', caption='random forest')

    f.close()

    prefix = ''
    if new_method:
        prefix = 'new_'

    joblib.dump(rf, output_class_folder + prefix + 'random_forest_single_scenario.pkl')
    joblib.dump(lsvm, output_class_folder + prefix + 'linear_svm_single_scenario.pkl')
    joblib.dump(lr, output_class_folder + prefix + 'logistic_single_scenario.pkl')


# punto corretto di inizio del programma per uno scrip python
if __name__ == "__main__":
    method = False
    try:
        opts, args = getopt.getopt(sys.argv[1:], '', ['new',])
    except getopt.GetoptError:
        print('test.py -i <inputfile> -o <outputfile>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '--new':
            method = True
            print(opt, ' : ', arg)

    # passo tutti i argv anche se vuoti per magari un futuro utilizzo
    single_scenario_computation(sys.argv[1:], method)
