# Train Template

This is a template repository for training machine learning models for neural data using PyTorch Lightning. It provides a structured setup for data handling, model definition, and training processes.

## Quick Start

1. Install the required dependencies (we recommend using [uv](https://pypi.org/project/uv/) for managing virtual environments):

   ```bash
   uv sync  # if not using uv, use: pip install -e .[dev]
   ```

2. Modify the model, train logic, and configuration files as needed.

3. Run the training script:

   ```bash
   uv run -m src.train  # if not using uv, use: python -m src.train
   ```

## Repository Structure

- `configs/`: Contains configuration files for different training setups. We use [Hydra](https://hydra.cc/) for configuration management, allowing easy CLI overrides and organization.
- `src/`: Contains the source code for data modules, models, and training scripts.
  - `models/`: Model definitions using PyTorch Lightning.
  - `utils/`: Utility functions for data processing and logging.
  - `train.py`: The main training script that orchestrates the training process.
