"""Class definition to manipulate data spindle EEG datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import time

import numpy as np
import pyedflib

from . import utils
from libs.common import checks

KEY_EEG_FRONTAL = 'frontal'
KEY_EEG_CENTRAL = 'central'
KEY_EEG_OCCIPITAL = 'occipital'
KEY_EMG = 'emg'
KEY_EOG_LEFT = 'eog_left'
KEY_EOG_RIGHT = 'eog_right'
KEY_HYPNOGRAM = 'hypnogram'

PATH_MASS_RELATIVE = 'mass'
PATH_REC = 'register'
PATH_STATES = 'state'

KEY_FILE_EEG = 'file_eeg'
KEY_FILE_STATES = 'file_states'

IDS_TEST = [2, 6, 12, 13]


class Mass(object):
    """This is a class to manipulate the MASS dataset.

    Expected directory tree inside DATA folder (see utils.py):

    PATH_MASS_RELATIVE
    |__ PATH_REC
        |__ 01-02-0001 PSG.edf
        |__ 01-02-0002 PSG.edf
        |__ ...
    |__ PATH_STATES
        |__ 01-02-0001 Base.edf
        |__ 01-02-0002 Base.edf
        |__ ...
    """

    def __init__(
            self,
            load_checkpoint,
            verbose=True
    ):
        """Constructor.

        Args:
            load_checkpoint: (Boolean). Whether to load from a checkpoint or to
               load from scratch using the original files of the dataset.
        """
        self.dataset_name = 'mass'
        valid_ids = [i for i in range(1, 20)]
        self.test_ids = IDS_TEST
        self.train_ids = [i for i in valid_ids if i not in self.test_ids]
        self.all_ids = self.train_ids + self.test_ids
        self.all_ids.sort()
        if verbose:
            print('Dataset %s with %d patients.'
                  % (self.dataset_name, len(self.all_ids)))
            print('Train size: %d. Test size: %d'
                  % (len(self.train_ids), len(self.test_ids)))
            print('Train subjects: \n', self.train_ids)
            print('Test subjects: \n', self.test_ids)

        self.fs = 256  # Original sampling frequency [Hz]
        self.page_duration = 20  # Time of window page [s]

        # Save attributes
        dataset_dir = os.path.join(utils.PATH_DATA, PATH_MASS_RELATIVE)
        if os.path.isabs(dataset_dir):
            self.dataset_dir = dataset_dir
        else:
            self.dataset_dir = os.path.abspath(
                os.path.join(utils.PATH_DATA, dataset_dir))
        # We verify that the directory exists
        if not load_checkpoint:
            checks.check_directory(self.dataset_dir)

        self.load_checkpoint = load_checkpoint
        self.ckpt_dir = os.path.abspath(os.path.join(
            self.dataset_dir, '..', 'ckpt_%s' % self.dataset_name))
        self.ckpt_file = os.path.join(
            self.ckpt_dir, '%s.pickle' % self.dataset_name)

        # Data loading
        self.data = self._load_data(verbose=verbose)

    def save_checkpoint(self):
        """Saves a pickle file containing the loaded data."""
        os.makedirs(self.ckpt_dir, exist_ok=True)
        with open(self.ckpt_file, 'wb') as handle:
            pickle.dump(
                self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Checkpoint saved at %s' % self.ckpt_file)

    def _load_data(self, verbose):
        """Loads data either from a checkpoint or from scratch."""
        if self.load_checkpoint and self._exists_checkpoint():
            if verbose:
                print('Loading from checkpoint... ', flush=True, end='')
            data = self._load_from_checkpoint()
        else:
            if verbose:
                if self.load_checkpoint:
                    print("A checkpoint doesn't exist at %s."
                          " Loading from source instead." % self.ckpt_file)
                else:
                    print('Loading from source.')
            data = self._load_from_source()
        if verbose:
            print('Loaded')
        return data

    def _load_from_checkpoint(self):
        """Loads the pickle file containing the loaded data."""
        with open(self.ckpt_file, 'rb') as handle:
            data = pickle.load(handle)
        return data

    def _exists_checkpoint(self):
        """Checks whether the pickle file with the checkpoint exists."""
        return os.path.isfile(self.ckpt_file)

    def _load_from_source(self):
        """Loads the data from files and transforms it appropriately."""
        data_paths = self._get_file_paths()
        data = {}
        n_data = len(data_paths)
        start = time.time()
        for i, subject_id in enumerate(data_paths.keys()):
            print('\nLoading ID %d' % subject_id)
            path_dict = data_paths[subject_id]

            # Read data
            signal_eeg_frontal = self._read_eeg(
                path_dict[KEY_FILE_EEG], KEY_EEG_FRONTAL)
            signal_eeg_central = self._read_eeg(
                path_dict[KEY_FILE_EEG], KEY_EEG_CENTRAL)
            signal_eeg_occipital = self._read_eeg(
                path_dict[KEY_FILE_EEG], KEY_EEG_OCCIPITAL)

            signal_eog_left = self._read_eeg(
                path_dict[KEY_FILE_EEG], KEY_EOG_LEFT)
            signal_eog_right = self._read_eeg(
                path_dict[KEY_FILE_EEG], KEY_EOG_RIGHT)

            signal_emg = self._read_eeg(
                path_dict[KEY_FILE_EEG], KEY_EMG)

            signal_len = signal_eeg_frontal.shape[0]
            hypnogram = self._read_states(
                path_dict[KEY_FILE_STATES], signal_len)
            print('Hypnogram pages: %d' % hypnogram.shape[0])

            # Save data
            ind_dict = {
                KEY_EEG_FRONTAL: signal_eeg_frontal,
                KEY_EEG_CENTRAL: signal_eeg_central,
                KEY_EEG_OCCIPITAL: signal_eeg_occipital,
                KEY_EOG_LEFT: signal_eog_left,
                KEY_EOG_RIGHT: signal_eog_right,
                KEY_EMG: signal_emg,
                KEY_HYPNOGRAM: hypnogram
            }
            data[subject_id] = ind_dict
            print('Loaded ID %d (%02d/%02d ready). Time elapsed: %1.4f [s]'
                  % (subject_id, i+1, n_data, time.time()-start))
        print('%d records have been read.' % len(data))
        return data

    def _get_file_paths(self):
        """Returns a list of dicts containing paths to load the database."""
        # Build list of paths
        data_paths = {}
        for subject_id in self.all_ids:
            path_eeg_file = os.path.join(
                self.dataset_dir, PATH_REC,
                '01-02-%04d PSG.edf' % subject_id)
            path_states_file = os.path.join(
                self.dataset_dir, PATH_STATES,
                '01-02-%04d Base.edf' % subject_id)
            # Save paths
            ind_dict = {
                KEY_FILE_EEG: path_eeg_file,
                KEY_FILE_STATES: path_states_file
            }
            # Check paths
            for key in ind_dict:
                if not os.path.isfile(ind_dict[key]):
                    print(
                        'File not found: %s' % ind_dict[key])
            data_paths[subject_id] = ind_dict
        print('%d records in %s dataset.' % (len(data_paths), self.dataset_name))
        print('Subject IDs: %s' % self.all_ids)
        return data_paths

    def _read_eeg(self, path_eeg_file, channel_name):
        """Loads signal from 'path_eeg_file', does filtering and resampling."""
        proper_name_dict = {
            KEY_EEG_FRONTAL: 'EEG F3-CLE',  # Frontal
            KEY_EEG_CENTRAL: 'EEG C3-CLE',  # Central
            KEY_EEG_OCCIPITAL: 'EEG O1-CLE',  # Occipital
            KEY_EMG: 'EMG Chin',  # EMG
            KEY_EOG_LEFT: 'EOG Left Horiz',  # Left eye
            KEY_EOG_RIGHT: 'EOG Right Horiz',  # Right eye
        }

        checks.check_valid_value(
            channel_name, 'channel_name', list(proper_name_dict.keys()))

        with pyedflib.EdfReader(path_eeg_file) as file:
            channel_names = file.getSignalLabels()
            channel_to_extract = channel_names.index(
                proper_name_dict[channel_name])
            signal = file.readSignal(channel_to_extract)
            fs_old = file.samplefrequency(channel_to_extract)
            # Check
            print('Channel extracted: %s' % file.getLabel(channel_to_extract))
        fs_old_round = int(np.round(fs_old))
        # Transform the original fs frequency with decimals to rounded version
        signal = utils.resample_signal_linear(
            signal, fs_old=fs_old, fs_new=fs_old_round)
        signal = signal.astype(np.float32)
        return signal

    def _read_states(self, path_states_file, signal_length):
        """Loads hypnogram from 'path_states_file'. Only n2 pages are returned.
        First, last and second to last pages of the hypnogram are ignored, since
        there is no enough context."""
        # Total pages not necessarily equal to total_annots
        page_size = self.fs * self.page_duration
        total_pages = int(np.ceil(signal_length / page_size)) - 1

        with pyedflib.EdfReader(path_states_file) as file:
            annotations = file.readAnnotations()

        onsets = np.array(annotations[0])
        durations = np.round(np.array(annotations[1]))
        stages_str = annotations[2]
        # keep only 20s durations
        valid_idx = (durations == self.page_duration)
        onsets = onsets[valid_idx]
        onsets_pages = np.round(onsets / self.page_duration).astype(np.int32)
        stages_str = stages_str[valid_idx]
        stages_char = [single_annot[-1] for single_annot in stages_str]

        # Build complete hypnogram
        total_annots = len(stages_char)

        state_ids = np.array(['1', '2', '3', '4', 'R', 'W', '?'])
        correct_id_dict = {
            '1': 'N1',
            '2': 'N2',
            '3': 'N3',
            '4': 'N3',
            'R': 'R',
            'W': 'W',
            '?': '?'
        }
        unknown_id = '?'  # Character for unknown state in hypnogram

        not_unknown_ids = [
            state_id for state_id in state_ids
            if state_id != unknown_id]
        not_unknown_state_dict = {}
        for state_id in not_unknown_ids:
            state_idx = np.where(
                [stages_char[i] == state_id for i in range(total_annots)])[0]
            not_unknown_state_dict[state_id] = onsets_pages[state_idx]

        hypnogram = []
        for page in range(total_pages):
            state_not_found = True
            for state_id in not_unknown_ids:
                if page in not_unknown_state_dict[state_id] and state_not_found:
                    hypnogram.append(correct_id_dict[state_id])
                    state_not_found = False
            if state_not_found:
                hypnogram.append(unknown_id)
        hypnogram = np.asarray(hypnogram)

        return hypnogram

    def get_ids(self):
        return self.all_ids

    def get_train_ids(self):
        return self.train_ids

    def get_test_ids(self):
        return self.test_ids

    def get_signal_names(self):
        names = [
            KEY_EEG_FRONTAL,
            KEY_EEG_CENTRAL,
            KEY_EEG_OCCIPITAL,
            KEY_EOG_LEFT,
            KEY_EOG_RIGHT,
            KEY_EMG
        ]
        return names

    def get_subject_signal(self, subject_id, verbose=False):
        checks.check_valid_value(subject_id, 'ID', self.all_ids)
        ind_dict = self.data[subject_id]
        signal = np.stack(
            [
                ind_dict[KEY_EEG_FRONTAL],
                ind_dict[KEY_EEG_CENTRAL],
                ind_dict[KEY_EEG_OCCIPITAL],
                ind_dict[KEY_EOG_LEFT],
                ind_dict[KEY_EOG_RIGHT],
                ind_dict[KEY_EMG]
            ],
            axis=1)
        if verbose:
            print('Getting signal of ID %s' % subject_id)
        return signal

    def get_subset_signals(self, subject_id_list, verbose=False):
        subset_signals = []
        for subject_id in subject_id_list:
            signal = self.get_subject_signal(
                subject_id, verbose=verbose)
            subset_signals.append(signal)
        return subset_signals

    def get_signals(self, verbose=False):
        subset_signals = self.get_subset_signals(
            self.all_ids, verbose=verbose)
        return subset_signals

    def get_subject_hypnogram(self, subject_id, verbose=False):
        """Returns the hypogram of this subject."""
        checks.check_valid_value(subject_id, 'ID', self.all_ids)

        ind_dict = self.data[subject_id]
        hypno = ind_dict[KEY_HYPNOGRAM]
        if verbose:
            print('Getting Hypnogram of ID %s' % subject_id)
        return hypno

    def get_subset_hypnograms(self, subject_id_list, verbose=False):
        """Returns the list of hypograms from a list of subjects."""
        subset_hypnos = []
        for subject_id in subject_id_list:
            hypno = self.get_subject_hypnogram(
                subject_id,
                verbose=verbose)
            subset_hypnos.append(hypno)
        return subset_hypnos

    def get_hypnograms(self, verbose=False):
        """Returns the list of hypograms from all subjects."""
        subset_hypnos = self.get_subset_hypnograms(
            self.all_ids,
            verbose=verbose)
        return subset_hypnos

    def get_subject_data(
            self,
            subject_id,
            output_fs=100,
            border_duration=0,
            ignore_unknown=True,
            verbose=False):
        """
        Returns segments of the signals for the given id and their
        corresponding sleep stage.
        """
        checks.check_valid_value(subject_id, 'ID', self.all_ids)
        if output_fs > self.fs:
            raise ValueError('output fs cannot be greater than original fs')

        signal = self.get_subject_signal(subject_id)
        hypnogram = self.get_subject_hypnogram(subject_id)

        # Resample if needed
        if output_fs < self.fs:
            if verbose:
                print('Resampling from %d Hz to %d Hz' % (self.fs, output_fs))
            resampled_list = []
            for chn in range(signal.shape[1]):
                this_signal = signal[:, chn]
                this_signal = utils.resample_signal(
                    this_signal, self.fs, output_fs)
                resampled_list.append(this_signal)
            signal = np.stack(resampled_list, axis=1)

        # Extract segments
        segments_list = []
        stages_list = []
        page_size = int(self.page_duration * output_fs)
        border_size = int(border_duration * output_fs)
        for i, stage in enumerate(hypnogram):
            if i == 0:
                if verbose:
                    "Skipping first segment."
            elif stage == "?" and ignore_unknown:
                if verbose:
                    "Dropped unknown stage '?'."
            else:
                sample_start = i * page_size - border_size
                sample_end = (i + 1) * page_size + border_size
                this_segment = signal[sample_start:sample_end, :]
                segments_list.append(this_segment)
                stages_list.append(stage)

        x = np.stack(segments_list, axis=0)
        y = np.stack(stages_list, axis=0)

        if verbose:
            print('S%02d with %d segments' % (subject_id, x.shape[0]))

        return x, y

    def get_subset_data(
            self,
            subject_id_list,
            output_fs=100,
            border_duration=0,
            ignore_unknown=True,
            verbose=False
    ):
        """Returns the list of signals and marks from a list of subjects.
        """
        subset_signals = []
        subset_stages = []
        for subject_id in subject_id_list:
            signal, stages = self.get_subject_data(
                subject_id,
                output_fs=output_fs,
                border_duration=border_duration,
                ignore_unknown=ignore_unknown,
                verbose=verbose)
            subset_signals.append(signal)
            subset_stages.append(stages)
        return subset_signals, subset_stages

    def get_data(
            self,
            output_fs=100,
            border_duration=0,
            ignore_unknown=True,
            verbose=False
    ):
        """Returns the list of signals and marks from all subjects.
        """
        subset_signals, subset_stages = self.get_subset_data(
            self.all_ids,
            output_fs=output_fs,
            border_duration=border_duration,
            ignore_unknown=ignore_unknown,
            verbose=verbose)
        return subset_signals, subset_stages

