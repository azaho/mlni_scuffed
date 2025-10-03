from .base import BIDSSession

from temporaldata import ArrayDict, RegularTimeSeries, IrregularTimeSeries
from pathlib import Path
import pandas as pd
import numpy as np
import h5py
import json
import os

class BrainTreebankSession(BIDSSession):
    """
    This class is used to load the iEEG neural data for a given session from the BrainTreebank dataset at https://braintreebank.dev/
    """
    dataset_identifier = "braintreebank"
    def __init__(self, subject_identifier, session_identifier, root_dir=None, allow_corrupted=False):
        super().__init__(subject_identifier, session_identifier, root_dir=root_dir, allow_corrupted=allow_corrupted)

    @classmethod
    def discover_subjects(cls, root_dir=None):
        return list("sub_" + str(i) for i in range(1, 11)) # from 1 to 10. sub_1, sub_2, ..., sub_10

    @classmethod
    def discover_sessions(cls, subject_identifier, root_dir=None):
        if root_dir is None: root_dir = cls.find_root_dir()

        all_subject_trials = [
            (1, 0), (1, 1), (1, 2), 
            (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6),
            (3, 0), (3, 1), (3, 2), 
            (4, 0), (4, 1), (4, 2), 
            (5, 0), 
            (6, 0), (6, 1), (6, 4),
            (7, 0), (7, 1), 
            (8, 0), 
            (9, 0), 
            (10, 0), (10, 1)
        ]
        this_subject_trial_ids = [trial_id for subject_id, trial_id in all_subject_trials if subject_id == int(subject_identifier[4:])] # e.g. sub_1 -> [0, 1, 2] - session IDs

        return [
            {
                "session_identifier": f"trial{trial_id:03}",
                "events_file": None, # TODO: add the events later
                "ieeg_file": os.path.join(root_dir, f'{subject_identifier}_trial{trial_id:03}.h5'),
                "ieeg_electrodes_file": os.path.join(root_dir, f'electrode_labels/{subject_identifier}/electrode_labels.json'),
                "ieeg_channels_file": os.path.join(root_dir, f'corrupted_elec.json'),
            }
            for trial_id in this_subject_trial_ids
        ]

    def __clean_electrode_label(self, electrode_label):
        return electrode_label.replace('*', '').replace('#', '')
    def __filter_electrode_labels(self, electrode_labels, channels_file):
        """
            Filter the electrode labels to remove corrupted electrodes and electrodes that don't have brain signal
        """
        filtered_electrode_labels = electrode_labels
        # Step 1. Remove corrupted electrodes
        if not self.allow_corrupted:
            with open(channels_file, 'r') as f:
                corrupted_electrodes = json.load(f)[self.subject_identifier]
                corrupted_electrodes = [self.__clean_electrode_label(e) for e in corrupted_electrodes]
            filtered_electrode_labels = [e for e in filtered_electrode_labels if e not in corrupted_electrodes]
        # Step 2. Remove trigger electrodes
        trigger_electrodes = [e for e in electrode_labels if (e.upper().startswith("DC") or e.upper().startswith("TRIG"))]
        filtered_electrode_labels = [e for e in filtered_electrode_labels if e not in trigger_electrodes]
        return filtered_electrode_labels
    def _load_ieeg_electrodes(self, electrodes_file, channels_file):
        # Load electrode labels
        with open(electrodes_file, 'r') as f:
            electrode_labels = json.load(f)
        electrode_labels = [self.__clean_electrode_label(e) for e in electrode_labels]
        self._non_filtered_electrode_labels = electrode_labels # used in load_ieeg_data for accessing the original electrode indices in h5 file

        electrode_labels = self.__filter_electrode_labels(electrode_labels, channels_file)
        
        # Load localization data
        loc_file = os.path.join(self.root_dir, f'localization/{self.subject_identifier}/depth-wm.csv') # TODO: make this a parameter in the session constructor
        df = pd.read_csv(loc_file)
        df['Electrode'] = df['Electrode'].apply(self.__clean_electrode_label)
        coordinates = np.zeros((len(electrode_labels), 3), dtype=np.float32)
        for label_idx, label in enumerate(electrode_labels):
            row = df[df['Electrode'] == label].iloc[0]
            # Convert coordinates from subject (LPI) to MNI (RAS) space. NOTE: this is not the same as the MNI space used in the BIDS specification. Awaiting proper MNI coordinates from braintreebank.
            # L = Left (+), P = Posterior (+), I = Inferior (+)
            # MNI (RAS): X = Right (+), Y = Anterior (+), Z = Superior (+)
            # So: 
            #   X_MNI = -L (flip sign)
            #   Y_MNI = -P (flip sign)
            #   Z_MNI = -I (flip sign)
            x_mni = -row['L']
            y_mni = -row['P']
            z_mni = -row['I']
            coordinates[label_idx] = np.array([x_mni, y_mni, z_mni], dtype=np.float32)

        return ArrayDict(
            id=np.array(electrode_labels),
            x=coordinates[:, 0],
            y=coordinates[:, 1],
            z=coordinates[:, 2],
            type=np.array(["SEEG"] * len(electrode_labels)),
            brain_area=np.array(["UNKNOWN"] * len(electrode_labels))
        )

    def _load_ieeg_data(self, ieeg_file, suppress_warnings=True):
        """
        This is an optional function to implement (only if the session contains ieeg data). Load the iEEG data from the ieeg file.

        Returns:
            ieeg_data: The iEEG data in the format of a temporaldata.RegularTimeSeries. The data is a numpy array of shape (n_channels, n_samples), the sampling rate is the sampling rate of the data (int, in Hz), the domain start is 0, and the domain is automatically determined based on the data.
        """
    
        with h5py.File(ieeg_file, 'r', locking=False) as f:
            h5_neural_data_keys = {electrode_label: f"electrode_{electrode_i}" for electrode_i, electrode_label in enumerate(self._non_filtered_electrode_labels)}
            # Get data length first
            electrode_data_length = f['data'][h5_neural_data_keys[self._non_filtered_electrode_labels[0]]].shape[0]

            electrode_labels = self.data_dict['channels'].id            
            # Pre-allocate tensor with specific dtype
            neural_data_cache = np.zeros((len(electrode_labels), electrode_data_length), dtype=np.float32)
            # Load data
            for electrode_id, electrode_label in enumerate(electrode_labels):
                neural_data_key = h5_neural_data_keys[electrode_label]
                neural_data_cache[electrode_id] = f['data'][neural_data_key]
        
        return RegularTimeSeries(
            data=neural_data_cache.T,
            sampling_rate=2048,
            domain_start=0.0,
            domain="auto"
        )


if __name__ == "__main__":
    root_dir = "/orcd/data/fiete/001/zaho/braintreebank/"
    import dotenv
    dotenv.load_dotenv()
    save_root_dir = os.getenv("DATA_ROOT_DIR")
    for subject_identifier in BrainTreebankSession.discover_subjects(root_dir=root_dir):
        for session in BrainTreebankSession.discover_sessions(subject_identifier=subject_identifier, root_dir=root_dir):
            session_identifier = session['session_identifier']
            session = BrainTreebankSession(subject_identifier=subject_identifier, session_identifier=session_identifier, root_dir=root_dir, allow_corrupted=False)
            path, data = session.save_data(save_root_dir=save_root_dir)
            
            print(f"Saved data for subject {subject_identifier} and session {session_identifier} to {path}")

            session_length = data.ieeg.data.shape[0] / data.ieeg.sampling_rate
            n_electrodes = data.ieeg.data.shape[1]
            print(f"\t\tSession length: {session_length:.2f} seconds\t\t{n_electrodes} electrodes")
