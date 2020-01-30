import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as colors
import itertools


def print_matrix(z, classes,
                 title='Title',
                 folder_output='../report/images/difference_matrix/',
                 file_name='result.png'):
    colors_ = [(1, 0, 0), (1, 1, 1), (0, 1, 0)]
    cm = LinearSegmentedColormap.from_list('my_map', colors_, N=100)
    divnorm = colors.DivergingNorm(vmin=-50, vcenter=0, vmax=50)
    plt.figure()
    plt.imshow(z, interpolation='nearest', cmap=cm, norm=divnorm)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = 'd'
    thresh = z.max() / 2.
    for i, j in itertools.product(range(z.shape[0]), range(z.shape[1])):
        plt.text(j, i, format(z[i, j], fmt),
                 horizontalalignment="center",
                 color="black")
    plt.tight_layout()
    # folder = '../report/images/difference_matrix/'
    plt.savefig(folder_output + file_name)
    plt.close()


########################################################################################################################
# attention here the function above is a general purpose but it's usage below is made over
# the results given by mine computation
# use the function to plot your results
########################################################################################################################
# prima parte
logistic_double_last_new = np.array([
    [746, 0, 2, 0],
    [0, 618, 118, 0],
    [0, 216, 494, 0],
    [0, 0, 8, 248]
])

logistic_double_last_old = np.array([
    [741, 0, 7, 0],
    [0, 676, 60, 0],
    [1, 226, 480, 3],
    [0, 0, 8, 248],
])

double_last_class = ['messenger', 'telegram', 'whatsapp', 'original']

res = logistic_double_last_new - logistic_double_last_old

for index, x in np.ndenumerate(res):
    if index[0] == index[1]:
        # i numeri sono gia apposto
        pass
    else:
        res[index] = -res[index]

print_matrix(res, double_last_class, title='Differences of confusion matrix', file_name='logistic_double_last_diff.png')

########################################################################################################################
# seconda parte

logistic_double_double_new = np.array([
    [248, 5, 1, 0, 0, 0, 0, 0, 0, 0],
    [2, 233, 14, 0, 0, 0, 2, 0, 0, 0],
    [13, 26, 202, 0, 0, 2, 0, 0, 0, 0],
    [0, 0, 0, 65, 116, 3, 0, 68, 0, 0],
    [0, 0, 0, 66, 57, 2, 0, 114, 0, 0],
    [0, 0, 0, 3, 4, 235, 1, 2, 0, 0],
    [0, 0, 0, 1, 0, 1, 80, 0, 152, 0],
    [0, 0, 0, 65, 129, 3, 0, 50, 0, 0],
    [0, 0, 0, 0, 0, 0, 104, 0, 123, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 5, 251],
])

logistic_double_double_old = np.array([
    [249, 4, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 241, 9, 0, 1, 0, 0, 0, 0, 0],
    [21, 9, 213, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 43, 107, 2, 0, 99, 0, 0],
    [0, 0, 0, 52, 79, 0, 0, 108, 0, 0],
    [0, 0, 0, 0, 1, 243, 0, 1, 0, 0],
    [0, 2, 0, 0, 0, 0, 83, 2, 147, 0],
    [0, 0, 0, 57, 112, 0, 0, 78, 0, 0],
    [0, 0, 0, 0, 0, 0, 102, 0, 124, 3],
    [1, 0, 0, 0, 0, 0, 0, 0, 4, 251],
])

double_double_class = ['mess_mess', 'tele_mess', 'what_mess',
                       'mess_tele', 'tele_tele', 'what_tele',
                       'mess_what', 'tele_what', 'what_what',
                       'original']

res = logistic_double_double_new - logistic_double_double_old

for index, x in np.ndenumerate(res):
    if index[0] == index[1]:
        # i numeri sono gia apposto
        pass
    else:
        res[index] = -res[index]

print_matrix(res, double_double_class, title='Differences of confusion matrix', file_name='logistic_double_double_diff.png')

########################################################################################################################
# terza parte

logistic_double_fist_last_new = np.array([
    [184, 0, 0, 46, 4, 19, 0, 0, 0, 0],
    [0, 710, 0, 0, 0, 0, 0, 5, 0, 0],
    [0, 0, 419, 0, 0, 0, 0, 0, 58, 4],
    [21, 0, 0, 220, 6, 4, 0, 0, 0, 0],
    [0, 0, 0, 0, 215, 22, 0, 0, 0, 0],
    [0, 0, 0, 0, 88, 169, 0, 0, 0, 0],
    [0, 256, 0, 0, 0, 0, 0, 5, 0, 0],
    [0, 9, 0, 0, 0, 0, 2, 236, 0, 0],
    [0, 0, 191, 0, 0, 0, 0, 1, 52, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 239],
])

logistic_double_fist_last_old = np.array([
    [181, 0, 0, 45, 4, 23, 0, 0, 0, 0],
    [0, 700, 0, 0, 0, 0, 9, 6, 0, 0],
    [0, 0, 414, 0, 0, 0, 0, 0, 63, 4],
    [22, 0, 0, 217, 6, 6, 0, 0, 0, 0],
    [0, 0, 0, 0, 225, 12, 0, 0, 0, 0],
    [0, 0, 0, 0, 77, 180, 0, 0, 0, 0],
    [0, 248, 0, 0, 0, 0, 6, 7, 0, 0],
    [0, 7, 0, 0, 0, 0, 0, 240, 0, 0],
    [0, 1, 180, 0, 0, 0, 0, 0, 63, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 239],
])

double_fist_last_class = ['mess', 'tele', 'what',
                          'mess_mess', 'tele_mess', 'what_mess',
                          'mess_tele', 'what_tele',
                          'mess_what',
                          'original']

res = logistic_double_fist_last_new - logistic_double_fist_last_old

for index, x in np.ndenumerate(res):
    if index[0] == index[1]:
        # i numeri sono gia apposto
        pass
    else:
        res[index] = -res[index]

print_matrix(res, double_fist_last_class, title='Differences of confusion matrix', file_name='logistic_fist_last_diff.png')
