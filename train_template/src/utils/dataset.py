from __future__ import annotations  # allow compatibility for Python 3.9

import logging
import os
from functools import partial
from typing import Literal

import pytorch_lightning as pl
import torch
import yaml
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from mlni.dataset.multi_session import MultiSessionDataset
from mlni.dataset.preprocessing import electrode_subset_batch, laplacian_rereference_batch
from mlni.dataset.sampler import SessionBatchSampler

logger = logging.getLogger(__name__)


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
