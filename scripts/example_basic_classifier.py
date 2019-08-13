from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

sys.path.append('..')

from libs.data.mass import Mass
from libs.data.utils import power_spectrum


def plot_confusion_matrix(y_true, y_pred,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = unique_labels(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


if __name__ == '__main__':
    dataset = Mass(load_checkpoint=True)

    subject_id = 1
    original_fs = dataset.fs
    page_duration = dataset.page_duration
    signal_names = dataset.get_signal_names()
    stage2int_dict = {
        '?': 2,
        'W': 1,
        'R': 0,
        'N1': -1,
        'N2': -2,
        'N3': -3
    }

    # Get database
    output_fs = 100
    x_train, y_train = dataset.get_subset_data(
        subject_id_list=dataset.get_train_ids(),
        output_fs=output_fs, border_duration=0, ignore_unknown=True)
    x_test, y_test = dataset.get_subset_data(
        subject_id_list=dataset.get_test_ids(),
        output_fs=output_fs, border_duration=0, ignore_unknown=True)
    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    x_test = np.concatenate(x_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    print('%d segments in train set. %d segments in test set'
          % (y_train.size, y_test.size))

    # Check class frequency
    print('')
    print('Training set')
    values, counts = np.unique(y_train, return_counts=True)
    for value, count in zip(values, counts):
        print('%s: %d segments (%1.2f %% of total)'
              % (value, count, 100*count/y_train.size))
    print('')
    print('Test set')
    values, counts = np.unique(y_test, return_counts=True)
    for value, count in zip(values, counts):
        print('%s: %d segments (%1.2f %% of total)'
              % (value, count, 100 * count / y_test.size))
    print('')

    # Compute simple features using FFT
    print('Computing Training set features')
    x_train_features = []
    for i in range(x_train.shape[0]):
        example_feats = []
        for chn in range(x_train.shape[2]):
            single_channel = x_train[i, :, chn]
            power, freq = power_spectrum(single_channel, output_fs)
            delta_idx = np.where((freq >= 0) & (freq <= 4))[0]
            theta_idx = np.where((freq >= 4) & (freq <= 7.5))[0]
            alpha_idx = np.where((freq >= 7.5) & (freq <= 15.5))[0]
            beta_idx = np.where((freq >= 15.5) & (freq <= 31))[0]
            gamma_idx = np.where(freq >= 31)[0]

            example_feats.append([
                power[delta_idx].sum(),
                power[theta_idx].sum(),
                power[alpha_idx].sum(),
                power[beta_idx].sum(),
                power[gamma_idx].sum()
            ])

        example_feats = np.concatenate(example_feats).flatten()
        x_train_features.append(example_feats)
    x_train_features = np.stack(x_train_features, axis=0)

    print('Computing Test set features')
    x_test_features = []
    for i in range(x_test.shape[0]):
        example_feats = []
        for chn in range(x_test.shape[2]):
            single_channel = x_test[i, :, chn]
            power, freq = power_spectrum(single_channel, output_fs)
            delta_idx = np.where((freq >= 0) & (freq <= 4))[0]
            theta_idx = np.where((freq >= 4) & (freq <= 7.5))[0]
            alpha_idx = np.where((freq >= 7.5) & (freq <= 15.5))[0]
            beta_idx = np.where((freq >= 15.5) & (freq <= 31))[0]
            gamma_idx = np.where(freq >= 31)[0]

            example_feats.append([
                power[delta_idx].sum(),
                power[theta_idx].sum(),
                power[alpha_idx].sum(),
                power[beta_idx].sum(),
                power[gamma_idx].sum()
            ])

        example_feats = np.concatenate(example_feats).flatten()
        x_test_features.append(example_feats)
    x_test_features = np.stack(x_test_features, axis=0)

    # Train a simple classifier to solve the task
    x_train_features, y_train = shuffle(
        x_train_features, y_train, random_state=0)
    clf = RandomForestClassifier(
        n_estimators=50, max_depth=4, class_weight="balanced", random_state=0)
    clf = clf.fit(x_train_features, y_train)

    # Predict on test data
    y_hat_test = clf.predict(x_test_features)

    # Evaluate performance
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plot_confusion_matrix(y_test, y_hat_test,
                          title='Confusion matrix, without normalization')
    plt.show()

    # Plot normalized confusion matrix
    plot_confusion_matrix(y_test, y_hat_test, normalize=True,
                          title='Normalized confusion matrix')
    plt.show()

