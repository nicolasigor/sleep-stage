from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append('..')

from libs.data.mass import Mass


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

    # Visualization of entire night
    show_wholenight = False

    if show_wholenight:
        signal = dataset.get_subject_signal(subject_id)
        y = dataset.get_subject_hypnogram(subject_id)
        fig, ax = plt.subplots(2, 1, figsize=(10, 6), dpi=120, sharex=True)

        ax[0].set_title('S%02d Whole-night' % subject_id)
        sep = 500
        time_axis = np.arange(signal.shape[0]) / (original_fs * 3600)
        for i, name in enumerate(signal_names):
            ax[0].plot(
                time_axis, signal[:, i] - sep * i, label=name, linewidth=1)
        ax[0].set_yticks([])
        ax[0].legend(loc='lower right')

        hypno_values = 2 * np.ones(time_axis.shape)
        for i, stage in enumerate(y):
            page_size = original_fs * page_duration
            start_sample = i * page_size
            end_sample = (i+1) * page_size
            hypno_values[start_sample:end_sample] = stage2int_dict[stage]
        ax[1].plot(time_axis, hypno_values)
        ax[1].set_yticks([2, 1, 0, -1, -2, -3])
        ax[1].set_yticklabels(['?', 'W', 'R', 'N1', 'N2', 'N3'])
        ax[1].set_xlabel('Time [h]')

        plt.show()

    # Visualization of single segment
    output_fs = 100
    x, y = dataset.get_subject_data(
        subject_id, output_fs=output_fs, border_duration=0, ignore_unknown=True)
    # Show an example of each class
    for stage_name in ['W', 'R', 'N1', 'N2', 'N3']:
        useful_idx = np.where(y == stage_name)[0]
        single_idx = np.random.choice(useful_idx)

        single_stage = y[single_idx]
        single_segment = x[single_idx, :, :]

        fig, ax = plt.subplots(6, 1, figsize=(12, 6), dpi=120, sharex=True)

        ax[0].set_title('S%02d Stage %s' % (subject_id, single_stage))
        sep = 500
        time_axis = np.arange(single_segment.shape[0]) / output_fs
        for i, name in enumerate(signal_names):
            max_value = np.max(np.abs(single_segment[:, i]))
            if name in ['frontal', 'central', 'occipital']:
                color = '#1b2631'
            elif name in ['eog_left', 'eog_right']:
                color = '#0277bd'
            else:
                color = '#c62828'
            ax[i].plot(
                time_axis, single_segment[:, i], label=name,
                linewidth=1, color=color)
            ax[i].legend(loc='upper right')
            ax[i].set_ylim([-max_value, max_value])
            ax[i].tick_params(labelsize=8)
        ax[-1].set_xlabel('Time [s]')

        plt.show()
