import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def write_matrix(f, matrix):
    f.write('\n {\\def\\arraystretch{1.3} \n \\begin{table}[H] \n')
    f.write('\\centering \n')
    strin = 'l|' * len(matrix[0])
    f.write('\\begin{tabular}{|%s} \n' % strin)
    f.write('\\hline \n')
    for x in range (0, len(matrix)):
        for y in range (0, len(matrix[0])):
            f.write('%s' % str(matrix[x, y]))
            if y == (matrix.shape[1]-1):
                f.write('  \\\\ \\hline')
            else:
                f.write('  &')
        f.write('\n')
    f.write('\\end{tabular} \n')
    f.write('\\end{table} }\n')

def write_row(f, matrix):
    f.write('\n {\\def\\arraystretch{1.3} \n \\begin{table}[H] \n')
    f.write('\\centering \n')
    strin = 'l |' * len(matrix)
    f.write('\\begin{tabular}{|%s}  \n' % strin)
    f.write('\\hline \n')
    for x in range (0, len(matrix)):
        f.write('%.4f' % matrix[x])
        if x == (len(matrix)-1):
            f.write('\\\\ \\hline  \n')
        else:
            f.write('&')
        f.write('\n')
    f.write('\\end{tabular} \n')
    f.write('\\end{table} }\n')


def write_log_table(f, matrix):
    f.write('\n\\begin{longtable} \n')
    strin = 'l |' * len(matrix[0])
    f.write('{|%s} \n' % strin)
    f.write('\\hline \n')
    for x in range (0, len(matrix)):
        for y in range (0, len(matrix[0])):
            f.write('%s' % str(matrix[x, y]))
            if y == (matrix.shape[1]-1):
                f.write('  \\\\ \\hline')
            else:
                f.write('  &')
        f.write('\n')
    f.write('\\end{longtable} \n')

def write_figure(f, path, caption=' '):
    f.write('\n \\begin{figure}[H] \n')
    f.write('\\centering \n')
    f.write('\\includegraphics[scale=.6]{images/%s} \n' %path)
    f.write('\\caption{%s} \n' %caption)
    f.write('\\end{figure} \n')

    

