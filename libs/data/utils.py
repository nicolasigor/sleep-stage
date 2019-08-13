"""utils.py: Module for general data eeg data operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import resample_poly, butter, filtfilt, firwin, lfilter

PATH_THIS_DIR = os.path.dirname(__file__)
PATH_DATA = os.path.join(PATH_THIS_DIR, '..', '..', 'datasets')

from libs.common import checks


def broad_filter(signal, fs, lowcut=0.1, highcut=35):
    """Returns filtered signal sampled at fs Hz, with a [lowcut, highcut] Hz
    bandpass."""
    # Generate butter bandpass of order 3.
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(3, [low, high], btype='band')
    # Apply filter to the signal with zero-phase.
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal


def filter_fir(kernel, signal):
    filtered_signal = lfilter(kernel, 1.0, signal)
    n_shift = (kernel.size - 1) // 2
    aligned = np.zeros(filtered_signal.shape)
    aligned[:-n_shift] = filtered_signal[n_shift:]
    return aligned


def filter_windowed_sinusoidal(
        signal, fs, central_freq, ntaps,
        sinusoidal_fn=np.cos, window_fn=np.hanning):
    # Kernel design
    kernel = get_kernel(
        ntaps, central_freq, fs, window_fn, sinusoidal_fn)
    # Apply kernel
    filtered_signal = filter_fir(kernel, signal)
    return filtered_signal


def get_kernel(ntaps, central_freq, fs=1, window_fn=np.hanning, sinusoidal_fn=np.cos):
    # Kernel design
    time_array = np.arange(ntaps) - ntaps // 2
    time_array = time_array / fs
    b_base = sinusoidal_fn(2 * np.pi * central_freq * time_array)
    cos_base = np.cos(2 * np.pi * central_freq * time_array)
    window = window_fn(b_base.size)
    norm_factor = np.sum(window * (cos_base ** 2))
    kernel = b_base * window / norm_factor
    return kernel


def resample_signal(signal, fs_old, fs_new):
    """Returns resampled signal, from fs_old Hz to fs_new Hz."""
    gcd_freqs = math.gcd(fs_new, fs_old)
    up = int(fs_new / gcd_freqs)
    down = int(fs_old / gcd_freqs)
    signal = resample_poly(signal, up, down)
    signal = np.array(signal, dtype=np.float32)
    return signal


def resample_signal_linear(signal, fs_old, fs_new):
    """Returns resampled signal, from fs_old Hz to fs_new Hz.

    This implementation uses simple linear interpolation to achieve this.
    """
    t = np.cumsum(np.ones(len(signal)) / fs_old)
    t_new = np.arange(t[0], t[-1], 1 / fs_new)
    signal = interp1d(t, signal)(t_new)
    return signal


def norm_clip_signal(signal, computed_std, clip_value=10):
    norm_signal = signal / computed_std
    # Now clip to clip_value (only if clip is not None)
    if clip_value:
        norm_signal = np.clip(norm_signal, -clip_value, clip_value)
    return norm_signal


def power_spectrum(signal, fs):
    """Returns the single-sided power spectrum of the signal using FFT"""
    n = signal.size
    y = np.fft.fft(signal)
    y = np.abs(y) / n
    power = y[:n // 2]
    power[1:-1] = 2 * power[1:-1]
    freq = np.fft.fftfreq(n, d=1 / fs)
    freq = freq[:n // 2]
    return power, freq


def pages2seq(pages_data, pages_indices):
    if pages_data.shape[0] != pages_indices.shape[0]:
        raise ValueError('Shape mismatch. Inputs need the same number of rows.')

    page_size = pages_data.shape[1]
    max_page = np.max(pages_indices)
    max_size = (max_page + 1) * page_size
    global_sequence = np.zeros(max_size, dtype=pages_data.dtype)
    for i, page in enumerate(pages_indices):
        sample_start = page * page_size
        sample_end = (page + 1) * page_size
        global_sequence[sample_start:sample_end] = pages_data[i, :]
    return global_sequence


def extract_pages(sequence, pages_indices, page_size, border_size=0):
    """Extracts and returns the given set of pages from the sequence.

    Args:
        sequence: (1-D Array) sequence from where to extract data.
        pages_indices: (1-D Array) array of indices of pages to be extracted.
        page_size: (int) number in samples of each page.
        border_size: (Optional, int,, defaults to 0) number of samples to be
            added at each border.

    Returns:
        pages_data: (2-D Array) array of shape [n_pages,page_size+2*border_size]
            that contains the extracted data.
    """
    pages_list = []
    for page in pages_indices:
        sample_start = page * page_size - border_size
        sample_end = (page + 1) * page_size + border_size
        page_signal = sequence[sample_start:sample_end]
        pages_list.append(page_signal)
    pages_data = np.stack(pages_list, axis=0)
    return pages_data


def simple_split_with_list(x, y, train_fraction=0.8, seed=None):
    """Splits data stored in a list.

    The data x and y are list of arrays with shape [batch, ...].
    These are split in two sets randomly using train_fraction over the number of
    element of the list. Then these sets are returned with
    the arrays concatenated along the first dimension
    """
    n_subjects = len(x)
    n_train = int(n_subjects * train_fraction)
    print('Split: Total %d -- Training %d' % (n_subjects, n_train))
    random_idx = np.random.RandomState(seed=seed).permutation(n_subjects)
    train_idx = random_idx[:n_train]
    test_idx = random_idx[n_train:]
    x_train = np.concatenate([x[i] for i in train_idx], axis=0)
    y_train = np.concatenate([y[i] for i in train_idx], axis=0)
    x_test = np.concatenate([x[i] for i in test_idx], axis=0)
    y_test = np.concatenate([y[i] for i in test_idx], axis=0)
    return x_train, y_train, x_test, y_test


def split_ids_list_v2(subject_ids, split_id, train_fraction=0.75, verbose=False):
    n_subjects = len(subject_ids)
    n_train = int(n_subjects * train_fraction)
    if verbose:
        print('Split IDs: Total %d -- Training %d' % (n_subjects, n_train))
    n_val = n_subjects - n_train
    start_idx = split_id * n_val
    epoch = int(start_idx / n_subjects)
    random_idx_1 = np.random.RandomState(seed=epoch).permutation(n_subjects)
    random_idx_2 = np.random.RandomState(seed=epoch+1).permutation(n_subjects)
    random_idx = np.concatenate([random_idx_1, random_idx_2])
    start_idx_relative = start_idx % n_subjects
    val_idx = random_idx[start_idx_relative:(start_idx_relative + n_val)]
    val_ids = [subject_ids[i] for i in val_idx]
    train_ids = [sub_id for sub_id in subject_ids if sub_id not in val_ids]
    val_ids.sort()
    train_ids.sort()
    return train_ids, val_ids


def shuffle_data(x, y, seed=None):
    """Shuffles data assuming that they are numpy arrays."""
    n_examples = x.shape[0]
    random_idx = np.random.RandomState(seed=seed).permutation(n_examples)
    x = x[random_idx]
    y = y[random_idx]
    return x, y


def shuffle_data_with_ids(x, y, sub_ids, seed=None):
    """Shuffles data assuming that they are numpy arrays."""
    n_examples = x.shape[0]
    random_idx = np.random.RandomState(seed=seed).permutation(n_examples)
    x = x[random_idx]
    y = y[random_idx]
    sub_ids = sub_ids[random_idx]
    return x, y, sub_ids
