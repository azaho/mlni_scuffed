import math

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig
from transformers import LlamaConfig, LlamaModel

from preprocess.spectrogram import SpectrogramPreprocessor


class iEEGTransformer(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()

        self.spectrogram_preprocessor = SpectrogramPreprocessor(cfg)
        self.signal_projection = nn.Linear(
            self.spectrogram_preprocessor.n_freqs, cfg.model.transformer.d_model
        )

        max_position_embeddings = math.ceil(
            cfg.model.context_length
            / cfg.model.signal_preprocessing.segment_length
            / (1 - cfg.model.signal_preprocessing.p_overlap)
        )
        rope_theta = max_position_embeddings * 2 * np.pi
        # LLaMA uses RoPE by default on the sequence dimension
        config = LlamaConfig(
            hidden_size=cfg.model.transformer.d_model,
            num_hidden_layers=cfg.model.transformer.n_layers,
            num_attention_heads=cfg.model.transformer.n_heads,
            intermediate_size=4 * cfg.model.transformer.d_model,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            hidden_dropout_prob=cfg.training.dropout,
            attention_dropout=cfg.training.dropout,
        )
        self.transformer = LlamaModel(config)

        # Task-specific head (e.g., classification, regression, etc.)
        self.output_head = nn.Linear(
            cfg.model.transformer.d_model, self.spectrogram_preprocessor.n_freqs
        )  # adjust as needed

    def forward(self, batch: dict):
        """
        Args:
            x: [batch_size, num_electrodes, num_timesamples]
        """
        batch_size, num_electrodes, num_timesamples = batch["ieeg"]["data"].shape

        inputs = self.spectrogram_preprocessor(
            batch
        )  # [batch_size, num_electrodes, num_timebins, n_freqs]
        num_timebins = inputs.shape[2]

        # Project signal values
        signal_emb = self.signal_projection(
            inputs
        )  # [batch_size, num_electrodes, num_timebins, d_model]
        position_ids = (
            torch.arange(num_timebins, device=signal_emb.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(batch_size, num_electrodes, num_timebins)
        )

        # Option 1: Flatten to [B, N*M, d_model] - attend across all electrode-time pairs
        signal_emb = signal_emb.reshape(
            batch_size, num_electrodes * num_timebins, -1
        )  # [batch_size, num_electrodes * num_timebins, d_model]
        position_ids = position_ids.reshape(batch_size, num_electrodes * num_timebins)
        outputs = self.transformer(
            inputs_embeds=signal_emb, position_ids=position_ids
        )  # [batch_size, num_electrodes * num_timebins, d_model]

        # Get final representation
        hidden_states = (
            outputs.last_hidden_state
        )  # [batch_size, num_electrodes * num_timebins, d_model]
        hidden_states = hidden_states.reshape(
            batch_size, num_electrodes, num_timebins, -1
        )  # [batch_size, num_electrodes, num_timebins, d_model]

        outputs = self.output_head(hidden_states)

        return inputs, hidden_states, outputs

    def calculate_pretraining_loss(self, batch: dict, batch_idx: int) -> torch.Tensor:
        inputs, hidden_states, outputs = self(batch)
        loss = nn.functional.mse_loss(
            outputs[:, :, :-1], inputs[:, :, 1:]
        )  # next timebin prediction loss
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss = self.calculate_pretraining_loss(batch, batch_idx)
        batch_size = batch["ieeg"]["data"].shape[0]
        self.log("train_loss", loss, batch_size=batch_size)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss = self.calculate_pretraining_loss(batch, batch_idx)
        batch_size = batch["ieeg"]["data"].shape[0]
        self.log("val_loss", loss, batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.cfg.training.learning_rate
        )
        return optimizer
