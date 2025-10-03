import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, DeviceStatsMonitor
from pytorch_lightning.loggers import WandbLogger, CSVLogger
import hydra
from omegaconf import DictConfig, OmegaConf
from model.mse_ar import iEEGTransformer
from dataset import iEEGDataModule
from logger import TrainLossLogger
import logging
import sys
from dotenv import load_dotenv

load_dotenv()

torch.set_float32_matmul_precision('medium')


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    # Configure logging to stdout (not stderr)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True  # Override any existing logging config
    )
    logger = logging.getLogger(__name__)

    # Set random seed for reproducibility
    if cfg.training.random_seed is None:
        cfg.training.random_seed = hash(cfg.training.random_string) % (2**32)
    pl.seed_everything(cfg.training.random_seed)
    
    # Print configuration
    logger.info("=" * 80 + "\nConfiguration:\n" + OmegaConf.to_yaml(cfg) + "=" * 80)
    
    # Initialize data module
    logger.info("Initializing data module...")
    datamodule = iEEGDataModule(cfg)
    
    # Initialize model
    logger.info("Initializing model...")
    model = iEEGTransformer(cfg)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {n_params:,}")
    logger.info(f"Trainable parameters: {n_trainable:,}")
    
    # Set up callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints",
            filename="{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            every_n_epochs=cfg.cluster.save_model_every_n_epochs,
        ),
        LearningRateMonitor(logging_interval="step"),
        # Monitor GPU/CPU stats
        DeviceStatsMonitor(cpu_stats=True),
        # Print train loss every batch
        TrainLossLogger(),
    ]
    
    # Set up loggers
    loggers = []
    
    # CSV Logger (always active) for local logging
    csv_logger = CSVLogger(save_dir="logs", name="training_logs")
    loggers.append(csv_logger)
    logger.info("CSV logging enabled: logs/training_logs")
    
    # WandB Logger (optional)
    if cfg.cluster.wandb_project and cfg.cluster.wandb_entity:
        wandb_logger = WandbLogger(
            project=cfg.cluster.wandb_project,
            entity=cfg.cluster.wandb_entity,
            name=cfg.training.setup_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        loggers.append(wandb_logger)
        logger.info(f"W&B logging enabled: {cfg.cluster.wandb_entity}/{cfg.cluster.wandb_project}")
    else:
        logger.info("W&B logging disabled (wandb_project and wandb_entity are empty)")
    
    # Set up trainer
    precision = "bf16-mixed" if cfg.cluster.use_mixed_precision and cfg.cluster.amp_dtype == "bfloat16" else "32-true"
    
    trainer = pl.Trainer(
        max_epochs=cfg.training.n_epochs,
        callbacks=callbacks,
        logger=loggers,  # Use list of loggers
        precision=precision,
        accelerator="auto",
        devices=1,
        log_every_n_steps=1,  # Log every batch
        gradient_clip_val=1.0,
        deterministic=True,
        enable_progress_bar=False,  # Disable messy progress bar
        enable_model_summary=True,
    )
    
    # Train
    logger.info("=" * 80 + "\nStarting training...\n" + "=" * 80)
    trainer.fit(model, datamodule=datamodule)
    logger.info("=" * 80 + "\nTraining complete!\n" + "=" * 80)


if __name__ == "__main__":
    main()

