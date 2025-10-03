import logging
import time
import torch
import psutil
from datetime import datetime
from pytorch_lightning.callbacks import Callback


class TrainLossLogger(Callback):
    """Callback to log train loss every batch using Lightning logging."""
    
    def __init__(self):
        super().__init__()
        self.epoch_start_time = None
        self.batch_start_time = None
        self.batch_end_time = None
        self.logger = logging.getLogger(__name__)
    
    def _get_memory_usage(self):
        """Get RAM and GPU memory usage, including peak GPU usage since last reset."""
        ram = psutil.virtual_memory()
        ram_used_gb, ram_total_gb, ram_percent = ram.used / (1024 ** 3), ram.total / (1024 ** 3), ram.percent
        
        if torch.cuda.is_available():
            gpu_allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            gpu_reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
            gpu_peak_allocated_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            gpu_peak_reserved_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
            gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            return {'ram_used_gb': ram_used_gb, 'ram_total_gb': ram_total_gb, 'ram_percent': ram_percent,
                    'gpu_allocated_gb': gpu_allocated_gb, 'gpu_reserved_gb': gpu_reserved_gb, 
                    'gpu_peak_allocated_gb': gpu_peak_allocated_gb, 'gpu_peak_reserved_gb': gpu_peak_reserved_gb,
                    'gpu_total_gb': gpu_total_gb, 'has_gpu': True}
        return {'ram_used_gb': ram_used_gb, 'ram_total_gb': ram_total_gb, 'ram_percent': ram_percent, 'has_gpu': False}
    
    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()
        self.batch_end_time = None  # Reset for new epoch
        # Reset peak memory stats at start of epoch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        mem = self._get_memory_usage()
        gpu_str = f" | GPU: {mem['gpu_allocated_gb']:.1f}GB allocated, {mem['gpu_reserved_gb']:.1f}GB reserved" if mem['has_gpu'] else ""
        self.logger.info("=" * 80 + f"\nEpoch {trainer.current_epoch}/{trainer.max_epochs - 1} started\n" +
                        f"RAM: {mem['ram_used_gb']:.1f}/{mem['ram_total_gb']:.1f}GB ({mem['ram_percent']:.1f}%){gpu_str}\n" + "=" * 80)
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.batch_start_time = time.time()
        # Calculate idle time (gap between previous batch end and current batch start)
        if self.batch_end_time is not None:
            self.idle_time = self.batch_start_time - self.batch_end_time
        else:
            self.idle_time = None  # First batch has no idle time
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if outputs is not None and 'loss' in outputs:
            loss, elapsed = outputs['loss'].item(), time.time() - self.batch_start_time
            batch_size, total_batches = batch['ieeg']['data'].shape[0], trainer.num_training_batches
            progress_pct, mem = (batch_idx + 1) / total_batches * 100, self._get_memory_usage()
            
            # Log to Lightning's loggers (CSV, WandB, etc.)
            log_kwargs = dict(on_step=True, on_epoch=False, prog_bar=False, logger=True, batch_size=batch_size)
            for k, v in {'train_loss_batch': loss, 'batch_time': elapsed, 'ram_used_gb': mem['ram_used_gb'], 'ram_percent': mem['ram_percent']}.items():
                pl_module.log(k, v, **log_kwargs)
            if self.idle_time is not None:
                pl_module.log('idle_time', self.idle_time, **log_kwargs)
            if mem['has_gpu']:
                for k, v in {'gpu_allocated_gb': mem['gpu_allocated_gb'], 'gpu_reserved_gb': mem['gpu_reserved_gb'],
                             'gpu_peak_allocated_gb': mem['gpu_peak_allocated_gb'], 'gpu_peak_reserved_gb': mem['gpu_peak_reserved_gb']}.items():
                    pl_module.log(k, v, **log_kwargs)
            
            # Console logging
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            gpu_str = f"GPU: {mem['gpu_peak_allocated_gb']:.2f}G" if mem['has_gpu'] else "GPU: N/A"
            header = f"[{timestamp} | {gpu_str} | RAM: {mem['ram_used_gb']:.1f}G]"
            idle_str = f" | idle: {self.idle_time:.3f}s" if self.idle_time is not None else ""
            print(f"{header} Epoch {trainer.current_epoch} | Batch {batch_idx+1:3d}/{total_batches} ({progress_pct:5.1f}%) | "
                  f"train_loss: {loss:.6f} | time: {elapsed:.3f}s{idle_str}")
            
            # Reset peak memory stats for next batch interval
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            # Save batch end time for next idle time calculation
            self.batch_end_time = time.time()
    
    def on_train_epoch_end(self, trainer, pl_module):
        elapsed, mem = time.time() - self.epoch_start_time, self._get_memory_usage()
        pl_module.log('epoch_time', elapsed, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        if mem['has_gpu']:
            gpu_str = f" | GPU: {mem['gpu_peak_allocated_gb']:.1f}GB allocated, {mem['gpu_peak_reserved_gb']:.1f}GB reserved"
        else:
            gpu_str = ""
        self.logger.info("=" * 80 + f"\nEpoch {trainer.current_epoch} completed in {elapsed:.2f}s\n" +
                        f"RAM: {mem['ram_used_gb']:.1f}/{mem['ram_total_gb']:.1f}GB ({mem['ram_percent']:.1f}%){gpu_str}\n" + "=" * 80)

