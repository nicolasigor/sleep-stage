# TODO: Adaptarlo para leer los canales de interes y el hipnograma y los metodos deseados

"""Class definition to manipulate data spindle EEG datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle

import numpy as np

from . import utils

KEY_EEG_FRONTAL = 'frontal'
KEY_EEG_CENTRAL = 'central'
KEY_EEG_OCCIPITAL = 'occipital'
KEY_EMG = 'emg'
KEY_EOG_LEFT = 'eog_left'
KEY_EOG_RIGHT = 'eog_right'
KEY_HYPNOGRAM = 'hypnogram'


PATH_MASS_RELATIVE = 'mass'
PATH_REC = 'register'
PATH_MARKS = os.path.join('label', 'spindle')
PATH_STATES = os.path.join('label', 'state')

KEY_FILE_EEG = 'file_eeg'
KEY_FILE_STATES = 'file_states'
KEY_FILE_MARKS = 'file_marks'

IDS_INVALID = [4, 8, 15, 16]
IDS_TEST = [2, 6, 12, 13]
# IDS_INVALID = []
# IDS_TEST = [2, 6, 12, 13, 4, 8, 15, 16]



class Mass(object):
    """This is a class to manipulate the MASS data EEG dataset.

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
    |__ PATH_MARKS
        |__ 01-02-0001 SpindleE1.edf
        |__ 01-02-0002 SpindleE1.edf
        |__ ...
        |__ 01-02-0001 SpindleE2.edf
        |__ 01-02-0002 SpindleE2.edf
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
        # MASS parameters
        self.channel = 'EEG C3-CLE'  # Channel for SS marks
        # In MASS, we need to index by name since not all the lists are
        # sorted equally

        # Hypnogram parameters
        self.state_ids = np.array(['1', '2', '3', '4', 'R', 'W', '?'])
        self.unknown_id = '?'  # Character for unknown state in hypnogram
        self.n2_id = '2'  # Character for N2 identification in hypnogram

        # Sleep spindles characteristics
        self.min_ss_duration = 0.3  # Minimum duration of SS in seconds
        self.max_ss_duration = 3  # Maximum duration of SS in seconds

        valid_ids = [i for i in range(1, 20) if i not in IDS_INVALID]
        self.test_ids = IDS_TEST
        self.train_ids = [i for i in valid_ids if i not in self.test_ids]

        print('Train size: %d. Test size: %d'
              % (len(self.train_ids), len(self.test_ids)))
        print('Train subjects: \n', self.train_ids)
        print('Test subjects: \n', self.test_ids)


        # Save attributes
        if os.path.isabs(dataset_dir):
            self.dataset_dir = dataset_dir
        else:
            self.dataset_dir = os.path.abspath(
                os.path.join(utils.PATH_DATA, dataset_dir))
        # We verify that the directory exists
        checks.check_directory(self.dataset_dir)

        self.load_checkpoint = load_checkpoint
        self.dataset_name = dataset_name
        self.event_name = event_name
        self.n_experts = n_experts
        self.ckpt_dir = os.path.abspath(os.path.join(
            self.dataset_dir, '..', 'ckpt_%s' % self.dataset_name))
        self.ckpt_file = os.path.join(
            self.ckpt_dir, '%s.pickle' % self.dataset_name)
        self.all_ids = all_ids
        self.all_ids.sort()
        if verbose:
            print('Dataset %s with %d patients.'
                  % (self.dataset_name, len(self.all_ids)))

        # events and data EEG related parameters
        self.params = pkeys.default_params.copy()
        if params is not None:
            self.params.update(params)  # Overwrite defaults

        # Sampling frequency [Hz] to be used (not the original)
        self.fs = self.params[pkeys.FS]
        # Time of window page [s]
        self.page_duration = self.params[pkeys.PAGE_DURATION]
        self.page_size = int(self.page_duration * self.fs)

        # Data loading
        self.data = self._load_data(verbose=verbose)
        self.global_std = 1.0


    def get_subject_signal(
            self,
            subject_id,
            normalize_clip=True,
            normalization_mode=constants.WN_RECORD,
            which_expert=1,
            verbose=False
    ):
        checks.check_valid_value(subject_id, 'ID', self.all_ids)
        valid_experts = [(i + 1) for i in range(self.n_experts)]
        checks.check_valid_value(which_expert, 'which_expert', valid_experts)
        checks.check_valid_value(
            normalization_mode, 'normalization_mode',
            [constants.N2_RECORD, constants.WN_RECORD])

        ind_dict = self.data[subject_id]

        # Unpack data
        signal = ind_dict[KEY_EEG]

        if normalize_clip:
            if normalization_mode == constants.WN_RECORD:
                if verbose:
                    print('Normalization with stats from '
                          'pages containing true events.')
                # Normalize using stats from pages with true events.
                marks = ind_dict['%s_%d' % (KEY_MARKS, which_expert)]
                # Transform stamps into sequence
                marks = utils.stamp2seq(marks, 0, signal.shape[0] - 1)
                tmp_pages = ind_dict[KEY_ALL_PAGES]
                activity = utils.extract_pages(
                    marks, tmp_pages,
                    self.page_size, border_size=0)
                activity = activity.sum(axis=1)
                activity = np.where(activity > 0)[0]
                tmp_pages = tmp_pages[activity]
                signal, _ = utils.norm_clip_signal(
                    signal, tmp_pages, self.page_size,
                    norm_computation=self.params[pkeys.NORM_COMPUTATION_MODE],
                    computed_std=self.global_std)
            else:
                if verbose:
                    print('Normalization with stats from '
                          'N2 pages.')
                n2_pages = ind_dict[KEY_N2_PAGES]
                signal, _ = utils.norm_clip_signal(
                    signal, n2_pages, self.page_size,
                    norm_computation=self.params[pkeys.NORM_COMPUTATION_MODE],
                    computed_std=self.global_std)
        return signal

    def get_subset_signals(
            self,
            subject_id_list,
            normalize_clip=True,
            normalization_mode=constants.WN_RECORD,
            which_expert=1,
            verbose=False
    ):
        subset_signals = []
        for subject_id in subject_id_list:
            signal = self.get_subject_signal(
                subject_id,
                normalize_clip=normalize_clip,
                normalization_mode=normalization_mode,
                which_expert=which_expert,
                verbose=verbose)
            subset_signals.append(signal)
        return subset_signals

    def get_signals(
            self,
            normalize_clip=True,
            normalization_mode=constants.WN_RECORD,
            which_expert=1,
            verbose=False
    ):
        subset_signals = self.get_subset_signals(
            self.all_ids,
            normalize_clip=normalize_clip,
            normalization_mode=normalization_mode,
            which_expert=which_expert,
            verbose=verbose)
        return subset_signals

    def get_ids(self):
        return self.all_ids

    def get_subject_pages(
            self,
            subject_id,
            pages_subset=constants.WN_RECORD,
            verbose=False
    ):
        """Returns the indices of the pages of this subject."""
        checks.check_valid_value(subject_id, 'ID', self.all_ids)
        checks.check_valid_value(
            pages_subset, 'pages_subset',
            [constants.N2_RECORD, constants.WN_RECORD])

        ind_dict = self.data[subject_id]

        if pages_subset == constants.WN_RECORD:
            pages = ind_dict[KEY_ALL_PAGES]
        else:
            pages = ind_dict[KEY_N2_PAGES]

        if verbose:
            print('Getting ID %s, %d %s pages'
                  % (subject_id, pages.size, pages_subset))
        return pages

    def get_subset_pages(
            self,
            subject_id_list,
            pages_subset=constants.WN_RECORD,
            verbose=False
    ):
        """Returns the list of pages from a list of subjects."""
        subset_pages = []
        for subject_id in subject_id_list:
            pages = self.get_subject_pages(
                subject_id,
                pages_subset=pages_subset,
                verbose=verbose)
            subset_pages.append(pages)
        return subset_pages

    def get_pages(
            self,
            pages_subset=constants.WN_RECORD,
            verbose=False
    ):
        """Returns the list of pages from all subjects."""
        subset_pages = self.get_subset_pages(
            self.all_ids,
            pages_subset=pages_subset,
            verbose=verbose
        )
        return subset_pages

    def get_subject_stamps(
            self,
            subject_id,
            which_expert=1,
            pages_subset=constants.WN_RECORD,
            verbose=False
    ):
        """Returns the sample-stamps of marks of this subject."""
        checks.check_valid_value(subject_id, 'ID', self.all_ids)
        valid_experts = [(i + 1) for i in range(self.n_experts)]
        checks.check_valid_value(which_expert, 'which_expert', valid_experts)
        checks.check_valid_value(
            pages_subset, 'pages_subset',
            [constants.N2_RECORD, constants.WN_RECORD])

        ind_dict = self.data[subject_id]

        marks = ind_dict['%s_%d' % (KEY_MARKS, which_expert)]

        if pages_subset == constants.WN_RECORD:
            pages = ind_dict[KEY_ALL_PAGES]
        else:
            pages = ind_dict[KEY_N2_PAGES]

        # Get stamps that are inside selected pages
        marks = utils.extract_pages_for_stamps(
            marks, pages, self.page_size)

        if verbose:
            print('Getting ID %s, %s pages, %d stamps'
                  % (subject_id, pages_subset, marks.shape[0]))
        return marks

    def get_subset_stamps(
            self,
            subject_id_list,
            which_expert=1,
            pages_subset=constants.WN_RECORD,
            verbose=False
    ):
        """Returns the list of stamps from a list of subjects."""
        subset_marks = []
        for subject_id in subject_id_list:
            marks = self.get_subject_stamps(
                subject_id,
                which_expert=which_expert,
                pages_subset=pages_subset,
                verbose=verbose)
            subset_marks.append(marks)
        return subset_marks

    def get_stamps(
            self,
            which_expert=1,
            pages_subset=constants.WN_RECORD,
            verbose=False
    ):
        """Returns the list of stamps from all subjects."""
        subset_marks = self.get_subset_stamps(
            self.all_ids,
            which_expert=which_expert,
            pages_subset=pages_subset,
            verbose=verbose
        )
        return subset_marks

    def get_subject_hypnogram(
            self,
            subject_id,
            verbose=False
    ):
        """Returns the hypogram of this subject."""
        checks.check_valid_value(subject_id, 'ID', self.all_ids)

        ind_dict = self.data[subject_id]

        hypno = ind_dict[KEY_HYPNOGRAM]

        if verbose:
            print('Getting Hypnogram of ID %s' % subject_id)
        return hypno

    def get_subset_hypnograms(
            self,
            subject_id_list,
            verbose=False
    ):
        """Returns the list of hypograms from a list of subjects."""
        subset_hypnos = []
        for subject_id in subject_id_list:
            hypno = self.get_subject_hypnogram(
                subject_id,
                verbose=verbose)
            subset_hypnos.append(hypno)
        return subset_hypnos

    def get_hypnograms(
            self,
            verbose=False
    ):
        """Returns the list of hypograms from all subjects."""
        subset_hypnos = self.get_subset_hypnograms(
            self.all_ids,
            verbose=verbose
        )
        return subset_hypnos

    def get_subject_data(
            self,
            subject_id,
            augmented_page=False,
            border_size=0,
            which_expert=1,
            pages_subset=constants.WN_RECORD,
            normalize_clip=True,
            normalization_mode=constants.WN_RECORD,
            verbose=False,
    ):
        """Returns segments of signal and marks from pages for the given id.

        Args:
            subject_id: (int) id of the subject of interest.
            augmented_page: (Optional, boolean, defaults to False) whether to
                augment the page with half page at each side.
            border_size: (Optional, int, defaults to 0) number of samples to be
                added at each border of the segments.
            which_expert: (Optional, int, defaults to 1) Which expert
                annotations should be returned. It has to be consistent with
                the given n_experts, in a one-based counting.
            pages_subset: (Optional, string, [WN_RECORD, N2_RECORD]) If
                WN_RECORD (default), pages from the whole record. If N2_RECORD,
                only N2 pages are returned.
            normalize_clip: (Optional, boolean, defaults to True) If true,
                the signal is normalized and clipped from pages statistics.
            normalization_mode: (Optional, string, [WN_RECORD, N2_RECORD]) If
                WN_RECORD (default), statistics for normalization are
                computed from pages containing true events. If N2_RECORD,
                statistics are computed from N2 pages.
            verbose: (Optional, boolean, defaults to False) Whether to print
                what is being read.

        Returns:
            signal: (2D array) each row is an (augmented) page of the signal
            marks: (2D array) each row is an (augmented) page of the marks
        """
        checks.check_valid_value(subject_id, 'ID', self.all_ids)
        valid_experts = [(i+1) for i in range(self.n_experts)]
        checks.check_valid_value(which_expert, 'which_expert', valid_experts)
        checks.check_valid_value(
            pages_subset, 'pages_subset',
            [constants.N2_RECORD, constants.WN_RECORD])
        checks.check_valid_value(
            normalization_mode, 'normalization_mode',
            [constants.N2_RECORD, constants.WN_RECORD])

        ind_dict = self.data[subject_id]

        # Unpack data
        signal = ind_dict[KEY_EEG]
        marks = ind_dict['%s_%d' % (KEY_MARKS, which_expert)]
        if pages_subset == constants.WN_RECORD:
            pages = ind_dict[KEY_ALL_PAGES]
        else:
            pages = ind_dict[KEY_N2_PAGES]

        # Transform stamps into sequence
        marks = utils.stamp2seq(marks, 0, signal.shape[0] - 1)

        # Compute border to be added
        if augmented_page:
            total_border = self.page_size // 2 + border_size
        else:
            total_border = border_size

        if normalize_clip:
            if normalization_mode == constants.WN_RECORD:
                if verbose:
                    print('Normalization with stats from '
                          'pages containing true events.')
                # Normalize using stats from pages with true events.
                tmp_pages = ind_dict[KEY_ALL_PAGES]
                activity = utils.extract_pages(
                    marks, tmp_pages,
                    self.page_size, border_size=0)
                activity = activity.sum(axis=1)
                activity = np.where(activity > 0)[0]
                tmp_pages = tmp_pages[activity]
                signal, _ = utils.norm_clip_signal(
                    signal, tmp_pages, self.page_size,
                    norm_computation=self.params[pkeys.NORM_COMPUTATION_MODE],
                    computed_std=self.global_std)
            else:
                if verbose:
                    print('Normalization with stats from '
                          'N2 pages.')
                n2_pages = ind_dict[KEY_N2_PAGES]
                signal, _ = utils.norm_clip_signal(
                    signal, n2_pages, self.page_size,
                    norm_computation=self.params[pkeys.NORM_COMPUTATION_MODE],
                    computed_std=self.global_std)

        # Extract segments
        signal = utils.extract_pages(
            signal, pages, self.page_size, border_size=total_border)
        marks = utils.extract_pages(
            marks, pages, self.page_size, border_size=total_border)

        if verbose:
            print('Getting ID %s, %d %s pages, Expert %d'
                  % (subject_id, pages.size, pages_subset, which_expert))
        return signal, marks

    def get_subset_data(
            self,
            subject_id_list,
            augmented_page=False,
            border_size=0,
            which_expert=1,
            pages_subset=constants.WN_RECORD,
            normalize_clip=True,
            normalization_mode=constants.WN_RECORD,
            verbose=False,
    ):
        """Returns the list of signals and marks from a list of subjects.
        """
        subset_signals = []
        subset_marks = []
        for subject_id in subject_id_list:
            signal, marks = self.get_subject_data(
                subject_id,
                augmented_page=augmented_page,
                border_size=border_size,
                which_expert=which_expert,
                pages_subset=pages_subset,
                normalize_clip=normalize_clip,
                normalization_mode=normalization_mode,
                verbose=verbose,
            )
            subset_signals.append(signal)
            subset_marks.append(marks)
        return subset_signals, subset_marks

    def get_data(
            self,
            augmented_page=False,
            border_size=0,
            which_expert=1,
            pages_subset=constants.WN_RECORD,
            normalize_clip=True,
            normalization_mode=constants.WN_RECORD,
            verbose=False
    ):
        """Returns the list of signals and marks from all subjects.
        """
        subset_signals, subset_marks = self.get_subset_data(
            self.all_ids,
            augmented_page=augmented_page,
            border_size=border_size,
            which_expert=which_expert,
            pages_subset=pages_subset,
            normalize_clip=normalize_clip,
            normalization_mode=normalization_mode,
            verbose=verbose
        )
        return subset_signals, subset_marks

    def get_sub_dataset(self, subject_id_list):
        """Data structure of a subset of subjects"""
        data_subset = {}
        for pat_id in subject_id_list:
            data_subset[pat_id] = self.data[pat_id].copy()
        return data_subset

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
            signal = self._read_eeg(
                path_dict[KEY_FILE_EEG])
            signal_len = signal.shape[0]

            n2_pages, hypnogram = self._read_states(
                path_dict[KEY_FILE_STATES], signal_len)
            total_pages = int(np.ceil(signal_len / self.page_size))
            all_pages = np.arange(1, total_pages - 2, dtype=np.int16)
            print('N2 pages: %d' % n2_pages.shape[0])
            print('Whole-night pages: %d' % all_pages.shape[0])
            print('Hypnogram pages: %d' % hypnogram.shape[0])

            marks_1 = self._read_marks(
                path_dict['%s_1' % KEY_FILE_MARKS])
            marks_2 = self._read_marks(
                path_dict['%s_2' % KEY_FILE_MARKS])
            print('Marks SS from E1: %d, Marks SS from E2: %d'
                  % (marks_1.shape[0], marks_2.shape[0]))

            # Save data
            ind_dict = {
                KEY_EEG: signal,
                KEY_N2_PAGES: n2_pages,
                KEY_ALL_PAGES: all_pages,
                '%s_1' % KEY_MARKS: marks_1,
                '%s_2' % KEY_MARKS: marks_2,
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
            path_marks_1_file = os.path.join(
                self.dataset_dir, PATH_MARKS,
                '01-02-%04d SpindleE1.edf' % subject_id)
            path_marks_2_file = os.path.join(
                self.dataset_dir, PATH_MARKS,
                '01-02-%04d SpindleE2.edf' % subject_id)
            # Save paths
            ind_dict = {
                KEY_FILE_EEG: path_eeg_file,
                KEY_FILE_STATES: path_states_file,
                '%s_1' % KEY_FILE_MARKS: path_marks_1_file,
                '%s_2' % KEY_FILE_MARKS: path_marks_2_file
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

    def _read_eeg(self, path_eeg_file):
        """Loads signal from 'path_eeg_file', does filtering and resampling."""
        with pyedflib.EdfReader(path_eeg_file) as file:
            channel_names = file.getSignalLabels()
            channel_to_extract = channel_names.index(self.channel)
            signal = file.readSignal(channel_to_extract)
            fs_old = file.samplefrequency(channel_to_extract)
            # Check
            print('Channel extracted: %s' % file.getLabel(channel_to_extract))

        fs_old_round = int(np.round(fs_old))
        # Transform the original fs frequency with decimals to rounded version
        signal = utils.resample_signal_linear(
            signal, fs_old=fs_old, fs_new=fs_old_round)
        # Broand bandpass filter to signal
        signal = utils.broad_filter(signal, fs_old)
        # Now resample to the required frequency
        signal = utils.resample_signal(
            signal, fs_old=fs_old_round, fs_new=self.fs)
        signal = signal.astype(np.float32)
        return signal

    def _read_marks(self, path_marks_file):
        """Loads data spindle annotations from 'path_marks_file'.
        Marks with a duration outside feasible boundaries are removed.
        Returns the sample-stamps of each mark."""
        with pyedflib.EdfReader(path_marks_file) as file:
            annotations = file.readAnnotations()
        onsets = np.array(annotations[0])
        durations = np.array(annotations[1])
        offsets = onsets + durations
        marks_time = np.stack((onsets, offsets), axis=1)  # time-stamps
        # Transforms to sample-stamps
        marks = np.round(marks_time * self.fs).astype(np.int32)
        # Combine marks that are too close according to standards
        marks = stamp_correction.combine_close_stamps(
            marks, self.fs, self.min_ss_duration)
        # Fix durations that are outside standards
        marks = stamp_correction.filter_duration_stamps(
            marks, self.fs, self.min_ss_duration, self.max_ss_duration)
        return marks

    def _read_states(self, path_states_file, signal_length):
        """Loads hypnogram from 'path_states_file'. Only n2 pages are returned.
        First, last and second to last pages of the hypnogram are ignored, since
        there is no enough context."""
        # Total pages not necessarily equal to total_annots
        total_pages = int(np.ceil(signal_length / self.page_size))

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

        not_unkown_ids = [
            state_id for state_id in self.state_ids
            if state_id != self.unknown_id]
        not_unkown_state_dict = {}
        for state_id in not_unkown_ids:
            state_idx = np.where(
                [stages_char[i] == state_id for i in range(total_annots)])[0]
            not_unkown_state_dict[state_id] = onsets_pages[state_idx]
        hypnogram = []
        for page in range(total_pages):
            state_not_found = True
            for state_id in not_unkown_ids:
                if page in not_unkown_state_dict[state_id] and state_not_found:
                    hypnogram.append(state_id)
                    state_not_found = False
            if state_not_found:
                hypnogram.append(self.unknown_id)
        hypnogram = np.asarray(hypnogram)

        # Extract N2 pages
        n2_pages = np.where(hypnogram == self.n2_id)[0]
        # Drop first, last and second to last page of the whole registers
        # if they where selected.
        last_page = total_pages - 1
        n2_pages = n2_pages[
            (n2_pages != 0)
            & (n2_pages != last_page)
            & (n2_pages != last_page - 1)]
        n2_pages = n2_pages.astype(np.int16)

        return n2_pages, hypnogram
