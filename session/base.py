from pathlib import Path
import pandas as pd
import numpy as np
from temporaldata import ArrayDict, RegularTimeSeries, Data
from mne_bids import read_raw_bids
import warnings
import h5py
import os

class SessionBase():
    """
    This class is used to load the iEEG neural data for a given session from the OpenNeuro BIDS dataset file format as used in OpenNeuro. The dataset is assumed to be stored in the root_dir directory.
    """
    dataset_identifier = None # NOTE: every subclass must define this variable

    def __init__(self, subject_identifier, session_identifier, root_dir=None, allow_corrupted=False):
        # Check if the root_dir is set in the environment variables
        if root_dir is None: root_dir = self.find_root_dir()

        self.root_dir = Path(root_dir)
        self.subject_identifier = subject_identifier
        self.session_identifier = session_identifier
        self.allow_corrupted = allow_corrupted
        
        self.data_dict = {
            "subject": self.subject_identifier,
            "session": self.session_identifier,
            "brainset": self.dataset_identifier,
            "allow_corrupted": self.allow_corrupted,
        }

        # Discover subjects and ensure the subject identifier exists in the dataset
        self.all_subjects = self.__class__.discover_subjects(self.root_dir)
        assert self.subject_identifier in self.all_subjects, f"Subject {self.subject_identifier} not found in dataset. List of subjects: {self.all_subjects}"
        self.subject_dir = self.root_dir / self.subject_identifier

        # Discover sessions and ensure the session identifier exists in the dataset
        self.all_sessions = self.__class__.discover_sessions(self.subject_identifier, root_dir=self.root_dir)
        all_session_identifiers = [session['session_identifier'] for session in self.all_sessions]
        assert self.session_identifier in all_session_identifiers, f"Session {self.session_identifier} not found in {self.all_sessions}"
        self.session = self.all_sessions[all_session_identifiers.index(self.session_identifier)]
    
    @classmethod
    def find_root_dir(cls):
        """
        Find the root directory of the dataset in the environment variables.
        """
        try:
            return os.environ.get("ROOT_DIR_" + cls.dataset_identifier.upper())
        except KeyError:
            raise ValueError(f"When loading dataset {cls.dataset_identifier}, ROOT_DIR_{cls.dataset_identifier.upper()} not set in environment variables. " \
                             f"Please either set the ROOT_DIR_{cls.dataset_identifier.upper()} environment variable or pass the root_dir argument to the constructor.")

    @classmethod
    def discover_subjects(cls, root_dir=None):
        """
        Discover all subjects in the dataset.

        Args:
            root_dir: The root directory of the dataset. If not provided, the root directory will be found in the environment variables.

        Returns:
            list: List of subject identifiers.
        """ 
        raise NotImplementedError("Not implemented")

    @classmethod
    def discover_sessions(cls, subject_identifier, root_dir=None):
        """
        Discover available sessions for the subject subject_identifier in the root_dir. This is a static method that can be used to discover sessions for any subject in the dataset.

        Args:
            subject_identifier: The identifier of the subject to discover sessions for.
            root_dir: The root directory of the dataset. If not provided, the root directory will be found in the environment variables.

        Returns:
            list: List of dictionaries containing:
                - session_identifier: the session identifier (e.g., "031411")
                - events_file: path to the file containing the events
                - ieeg_file: path to the file containing the iEEG data. Must be a BIDSPath object.
                - ieeg_electrodes_file: path to the file containing the electrodes. This file will contain the coordinates of each electrode.
                - ieeg_channels_file: path to the file containing the channels. This file will contain the labels of each channel and session-specific metadata (good vs bad channels, etc.)
        """
        raise NotImplementedError("Not implemented")
    
    def get_data(self):
        """
        Get the data for the session in the temporaldata format.

        Returns:
            data: The data for the session in the format of a temporaldata.Data object.
        """
        return Data(**self.data_dict, domain="auto")
    
    def save_data(self, save_root_dir):
        """
        Save the data for the session in the temporaldata format.

        Args:
            save_root_dir: The root directory to save the data to.

        Returns:
            path: The path to the saved data.
            data: The data for the session in the format of a temporaldata.Data object.
        """
        path = Path(save_root_dir) / self.dataset_identifier / self.subject_identifier / self.session_identifier
        path.mkdir(parents=True, exist_ok=True)

        data = self.get_data()
        
        # Save to HDF5
        with h5py.File(path / "data.h5", "w") as f:
            data.to_hdf5(f)
        return path, data

    def _load_ieeg_electrodes(self, electrodes_file, channels_file):
        """
        This is an optional function to implement (only if the session contains ieeg data). Load the electrodes from the electrodes file.

        Returns:
            electrodes: The electrodes in the format of a temporaldata.ArrayDict. The labels are the channel names, the coordinates are the x, y, z coordinates of the electrodes, and the types are the types of the channels.
        """
        raise NotImplementedError("Not implemented")

    def _load_ieeg_data(self, ieeg_file, suppress_warnings=True):
        """
        This is an optional function to implement (only if the session contains ieeg data). Load the iEEG data from the ieeg file.

        Returns:
            ieeg_data: The iEEG data in the format of a temporaldata.RegularTimeSeries. The data is a numpy array of shape (n_channels, n_samples), the sampling rate is the sampling rate of the data (int, in Hz), the domain start is 0, and the domain is automatically determined based on the data.
        """
        raise NotImplementedError("Not implemented")

    def _load_electrical_stimulation(self):
        """
        This is an optional function to implement (only if the session contains electrical stimulation data). Load the electrical stimulation data from the electrical stimulation file.

        Returns:
            electrical_stimulation: The electrical stimulation data in the format of a temporaldata.IrregularTimeSeries. The timestamps are the timestamps of the electrical stimulation, the stimulation_site are the sites of the electrical stimulation, the duration are the duration of the electrical stimulation, the waveform_type are the types of the electrical stimulation, the current are the currents of the electrical stimulation, the frequency are the frequencies of the electrical stimulation, the pulse_width are the pulse widths of the electrical stimulation, and the domain is automatically determined based on the data.
        """
        raise NotImplementedError("Not implemented")


class BIDSSession(SessionBase):
    """ 
    This class is used to load the iEEG neural data for a given session from the OpenNeuro BIDS dataset file format as used in OpenNeuro. The dataset is assumed to be stored in the root_dir directory.
    """
    def __init__(self, subject_identifier, session_identifier, root_dir=None, allow_corrupted=False):
        super().__init__(subject_identifier, session_identifier, root_dir=root_dir, allow_corrupted=allow_corrupted)

        self.data_dict['channels'] = self._load_ieeg_electrodes(self.session['ieeg_electrodes_file'], self.session['ieeg_channels_file'])
        self.data_dict['ieeg'] = self._load_ieeg_data(self.session['ieeg_file'])

    @classmethod
    def discover_subjects(cls, root_dir=None):
        if root_dir is None: root_dir = cls.find_root_dir()

        participants_file = Path(root_dir) / "participants.tsv"
        assert participants_file.exists(), f"participants.tsv not found in {root_dir} (looking for path: {participants_file})"
        participants_df = pd.read_csv(participants_file, sep='\t')
        assert 'participant_id' in participants_df.columns, "participants.tsv found but no 'participant_id' column present"
        return participants_df['participant_id'].values
    
    def _load_ieeg_electrodes(self, electrodes_file, channels_file):
        electrodes_df = pd.read_csv(electrodes_file, sep='\t')
        channels_df = pd.read_csv(channels_file, sep='\t')

        # Remove any rows that contain NaN values (usually meaning non-iEEG channels)
        electrodes_df = electrodes_df.dropna()

        # Filter channels to only include ECOG or SEEG types and good channels if not allowing corrupted data
        if 'type' in channels_df.columns:
            channels_df = channels_df[channels_df['type'].str.upper().isin(['ECOG', 'SEEG'])]
        if ('status' in channels_df.columns) and (not self.allow_corrupted):
            channels_df = channels_df[channels_df['status'].str.upper().isin(['GOOD'])]

        # Merge electrode coordinates into channels dataframe
        # For each channel, find the corresponding electrode and copy x, y, z coordinates
        channels_df = channels_df[['name', 'type']].merge(
            electrodes_df[['name', 'x', 'y', 'z']], 
            on='name', 
            how='left'
        )

        electrodes = ArrayDict(
            id=channels_df['name'].values.astype(str),
            x=channels_df['x'].values.astype(float),
            y=channels_df['y'].values.astype(float),
            z=channels_df['z'].values.astype(float),
            brain_area = np.array(["UNKNOWN"] * len(channels_df)), # TODO: add brain area
            type=channels_df['type'].values.astype(str),
        )
        return electrodes

    def _load_ieeg_data(self, ieeg_file, suppress_warnings=True):
        if suppress_warnings:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="No BIDS -> MNE mapping found")
                warnings.filterwarnings("ignore", message="Unable to map the following column")
                warnings.filterwarnings("ignore", message="Not setting positions")
                warnings.filterwarnings("ignore", message="DigMontage is only a subset of info.")
                warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne_bids")
                raw = read_raw_bids(ieeg_file, verbose=False)
        else:
            raw = read_raw_bids(ieeg_file, verbose=True)

        raw = raw.pick(self.data_dict['channels'].id)

        return RegularTimeSeries(
            data=raw.get_data().astype(np.float32).T * 1e6, # shape should be (n_samples, n_channels), and convert to microvolts
            sampling_rate=int(raw.info['sfreq']),
            domain_start = 0.0, # Start of the domain (in seconds)
            domain = "auto" # Automatically determine the domain based on the data
        )