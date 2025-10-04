import logging
import os
import random
from fnmatch import fnmatch
from functools import partial
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from omegaconf import DictConfig
from temporaldata import Data
from torch.utils.data import ConcatDataset, DataLoader, Dataset

logger = logging.getLogger(__name__)


class SingleSessionDataset(Dataset):
    def __init__(self, session_string: str, context_length: float):
        """
        Initialize the SingleSessionDataset with session details and context length.

        Args:
            session_string (str): A string in the format "brainset/subject/session[from:to]". If [from:to] is not provided, the entire session will be used.
            context_length (float): The length of the context window in seconds.

        """
        self.context_length = context_length
        self.session_string = session_string.split("[")[0]  # "brainset/subject/session"

        self.brainset, self.subject, self.session = self.session_string.split("/")
        self.data_file = os.path.join(
            os.environ["DATA_ROOT_DIR"],
            self.brainset,
            self.subject,
            self.session,
            "data.h5",
        )
        self.start_times = []  # list of start times of the context windows

        with h5py.File(self.data_file, "r") as f:
            data = Data.from_hdf5(f)

            if "[" in session_string:
                self.time_from, self.time_to = session_string.split("[")[1][:-1].split(":")
                self.time_from, self.time_to = (
                    float(self.time_from),
                    float(self.time_to),
                )
            else:
                self.time_from, self.time_to = data.domain.start[0], data.domain.end[0]

            # TODO: Handle artifacts here. Goal: go over every start_time, if it contians some artifact channels, remove those channels from that time.
            # If as result, too few channels remain, remove the start_time.
            self.start_times = np.arange(self.time_from, self.time_to - self.context_length, self.context_length)

    def __len__(self):
        return len(self.start_times)

    def __getitem__(self, idx: int) -> dict:
        """
        Return a dictionary containing the iEEG data, channels, and metadata for the given index.

        Args:
            idx (int): The index of the item to return.

        Returns:
            dict: A dictionary containing the iEEG data, channels, and metadata for the given index. Keys: "ieeg": {data: torch.Tensor[n_channels, n_samples], sampling_rate: int}, "channels": {id: np.array}, "metadata": {brainset: str, subject: str, session: str}.
        """
        start_time = self.start_times[idx]
        end_time = start_time + self.context_length
        with h5py.File(self.data_file, "r") as f:
            data = Data.from_hdf5(f)
            data = data.slice(start_time, end_time).materialize()

        return {
            "ieeg": {
                "data": torch.from_numpy(data.ieeg.data.T),  # shape: (n_channels, n_samples)
                "sampling_rate": int(data.ieeg.sampling_rate),
            },
            "channels": {"id": data.channels.id},
            "metadata": {
                "brainset": data.brainset,
                "subject": data.subject,
                "session": data.session,
            },
        }


class MultiSessionDataset(ConcatDataset):
    def __init__(self, session_strings: list, context_length: float):
        """
        Initialize MultiSessionDataset with support for glob pattern expansion.

        Args:
            session_strings (list): List of session strings, potentially containing glob patterns (*, ?, [seq], [!seq])
            context_length (float): Context length in seconds
        """
        super().__init__([SingleSessionDataset(session_string, context_length) for session_string in self._expand_session_wildcards(session_strings)])

    # Code that can be used to discover directories in the dataset
    def _discover_dirs(self, path, require_data_h5=False):
        """Helper to discover directories at a path, optionally requiring data.h5."""
        path = Path(path)
        if not path.exists():
            return []
        dirs = [d.name for d in path.iterdir() if d.is_dir() and not d.name.startswith(".") and (not require_data_h5 or (d / "data.h5").exists())]
        return sorted(dirs)

    def _expand_session_pattern(self, data_root, brainset_pattern, subject_pattern, session_pattern):
        """Recursively expand glob patterns in brainset/subject/session pattern."""
        data_root = Path(data_root)

        # Get brainsets - filter using glob pattern
        all_brainsets = self._discover_dirs(data_root)
        brainsets = [b for b in all_brainsets if fnmatch(b, brainset_pattern)]

        # Recursively expand subjects and sessions
        expanded = []
        for brainset in brainsets:
            all_subjects = self._discover_dirs(data_root / brainset)
            subjects = [s for s in all_subjects if fnmatch(s, subject_pattern)]
            for subject in subjects:
                all_sessions = self._discover_dirs(data_root / brainset / subject, require_data_h5=True)
                sessions = [s for s in all_sessions if fnmatch(s, session_pattern)]
                expanded.extend(f"{brainset}/{subject}/{session}" for session in sessions)

        return expanded

    def _expand_session_wildcards(self, session_strings: list) -> list:
        """
        Expand glob patterns in session strings (format: "brainset/subject/session[time_from:time_to]").
        Glob patterns (e.g., *, ?, [seq], [!seq]) can be used in any position. Time ranges are preserved.

        Examples:
            - "*/sub-01/*" - all brainsets, subject sub-01, all sessions
            - "dataset1/sub-*/ses-1" - dataset1, all subjects starting with 'sub-', session ses-1
            - "*/*/ses-*task-SPESclin*" - all brainsets/subjects, sessions matching the pattern
        """
        data_root = Path(os.environ.get("DATA_ROOT_DIR", ""))
        if not data_root.exists():
            raise ValueError("DATA_ROOT_DIR environment variable must be set")

        expanded = []
        for session_string in session_strings:
            time_range = ("[" + session_string.split("[")[1]) if "[" in session_string else ""

            # Parse and expand pattern
            brainset_pattern, subject_pattern, session_pattern = session_string.split("[")[0].split("/")
            matches = self._expand_session_pattern(data_root, brainset_pattern, subject_pattern, session_pattern)
            expanded.extend(f"{m}{time_range}" for m in matches)

        return sorted(expanded)


class SessionBatchSampler(torch.utils.data.Sampler):
    """
    Batch sampler that ensures each batch only contains samples from a single session.
    This is critical when different sessions have different numbers of channels or sampling rates.

    Args:
        dataset_sizes (list): List of dataset sizes for each session
        batch_size (int): Number of samples per batch
        shuffle (bool): Whether to shuffle indices within sessions and batch order
        drop_last (bool): Whether to drop incomplete batches
    """

    def __init__(
        self,
        dataset_sizes: list,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        self.dataset_sizes = dataset_sizes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        # Create batches for each session
        all_batches = []
        start_idx = 0

        for size in self.dataset_sizes:
            # Create indices for this session
            session_indices = list(range(start_idx, start_idx + size))
            if self.shuffle:
                random.shuffle(session_indices)

            # Create batches for this session
            session_batches = [session_indices[i : i + self.batch_size] for i in range(0, len(session_indices), self.batch_size) if not self.drop_last or i + self.batch_size <= len(session_indices)]
            all_batches.extend(session_batches)
            start_idx += size

        # Shuffle the order of batches across sessions if needed
        if self.shuffle:
            random.shuffle(all_batches)

        return iter(all_batches)

    def __len__(self):
        if self.drop_last:
            return sum(size // self.batch_size for size in self.dataset_sizes)
        return sum((size + self.batch_size - 1) // self.batch_size for size in self.dataset_sizes)


from preprocess.electrode_subset import electrode_subset_batch
from preprocess.laplacian_rereferencing import laplacian_rereference_batch


def ieeg_collate_fn(batch: list, cfg: DictConfig | None = None) -> dict:
    """
    Custom collate function to handle mixed data types in iEEG dataset.
    Optionally applies preprocessing after batching if cfg is provided.

    Handles:
    - Tensors: stacked normally
    - Integers: converted to tensors
    - Strings: kept as lists
    - Numpy string arrays: converted to lists of numpy arrays (one per batch item)

    Args:
        batch: List of samples from dataset
        cfg: Optional configuration object for preprocessing
    """
    if not batch:
        return {}

    # Get the structure from the first item
    first_item = batch[0]

    collated = {}

    # Handle "ieeg" data
    collated["ieeg"] = {
        "data": torch.stack([item["ieeg"]["data"] for item in batch]),
        "sampling_rate": first_item["ieeg"]["sampling_rate"],
    }

    # Handle "channels" - keep as list of arrays (can't stack string arrays)
    collated["channels"] = {"id": first_item["channels"]["id"]}

    # Handle "metadata" - keep strings as lists
    collated["metadata"] = {
        "brainset": first_item["metadata"]["brainset"],
        "subject": first_item["metadata"]["subject"],
        "session": first_item["metadata"]["session"],
    }

    # Apply preprocessing after batching if cfg is provided
    if cfg is not None:
        if cfg.model.signal_preprocessing.laplacian_rereference:
            collated = laplacian_rereference_batch(collated, inplace=True)
        collated = electrode_subset_batch(collated, cfg.training.max_n_electrodes, inplace=True)

    return collated


class iEEGDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for iEEG data.

    This module encapsulates all data loading logic including:
    - Loading training subject trials from config
    - Creating train/validation splits
    - Creating dataloaders with appropriate settings

    Args:
        cfg: Configuration object containing all settings
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_dataset = None
        self.val_dataset = None

    def prepare_data(self):
        """
        Download or prepare data. This is called only on 1 GPU/process.
        """
        # Check for DATA_ROOT_DIR environment variable
        if "DATA_ROOT_DIR" not in os.environ:
            raise ValueError("DATA_ROOT_DIR environment variable must be set")

    def setup(self, stage: Literal["fit", "validate", "test", "predict"] | None = None):
        """
        Load data and create train/validation splits.
        This is called on every GPU/process.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'
        """
        # Load training subject trials
        with open(self.cfg.training.train_subject_trials_file) as f:
            train_subject_trials = yaml.safe_load(f)

        logger.info(f"Loading {len(train_subject_trials)} training sessions...")
        self.full_dataset = MultiSessionDataset(train_subject_trials, self.cfg.model.context_length)

        # Extract individual session sizes from the ConcatDataset
        self.session_sizes = [len(dataset) for dataset in self.full_dataset.datasets]

        # Note: Session-aware batching requires using the full dataset
        # Train/val split would need to be done at the session level to maintain homogeneous batches
        # For now, we use all data for both training and validation with session-aware batching
        # TODO: Implement session-level train/val split if needed

        logger.info(f"Total samples across {len(self.session_sizes)} sessions: {len(self.full_dataset)}")
        logger.info(f"Session sizes: {self.session_sizes}")

    def train_dataloader(self):
        """Create and return the training dataloader with session-aware batching."""
        # Create session-aware batch sampler
        batch_sampler = SessionBatchSampler(
            dataset_sizes=self.session_sizes,
            batch_size=self.cfg.training.batch_size,
            shuffle=True,
            drop_last=True,
        )

        return DataLoader(
            self.full_dataset,  # Use full_dataset, not train_dataset, because sampler handles indices
            batch_sampler=batch_sampler,
            num_workers=self.cfg.cluster.num_workers_dataloaders,
            prefetch_factor=self.cfg.cluster.prefetch_factor,
            # persistent_workers=self.cfg.cluster.num_workers_dataloaders > 0,
            pin_memory=True,
            collate_fn=partial(ieeg_collate_fn, cfg=self.cfg),
        )

    def val_dataloader(self):
        """Create and return the validation dataloader with session-aware batching."""
        # Create session-aware batch sampler (no shuffling for validation)
        batch_sampler = SessionBatchSampler(
            dataset_sizes=self.session_sizes,
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
            drop_last=True,
        )

        return DataLoader(
            self.full_dataset,  # Use full_dataset for now - train/val split needs session-level splitting
            batch_sampler=batch_sampler,
            num_workers=self.cfg.cluster.num_workers_dataloaders,
            prefetch_factor=self.cfg.cluster.prefetch_factor,
            # persistent_workers=self.cfg.cluster.num_workers_dataloaders > 0,
            pin_memory=True,
            collate_fn=partial(ieeg_collate_fn, cfg=self.cfg),
        )


if __name__ == "__main__":
    dataset = SingleSessionDataset(
        "vanblooijs_hermes_developmental_2023/sub-ccepAgeUMCU01/ses-1_task-SPESclin_run-021448",
        context_length=1.0,
    )
    logger.info(dataset[0])
    logger.info(len(dataset))
