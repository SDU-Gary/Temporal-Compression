"""Training script for multi-temporal lighting compression.

Usage:
    python scripts/train.py --config configs/baseline.yaml
    python scripts/train.py --config configs/baseline.yaml --resume checkpoints/checkpoint_step_5000.pt
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
from torch.utils.data import DataLoader

from models.full_model import MultiTimeCompressionModel
from data.dataset import MultiTimeLightingDataset
from training.trainer import Trainer
from utils.config import Config, validate_config
from utils.logger import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train multi-temporal lighting compression model')

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )

    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )

    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to train on (cuda/cpu). Overrides config if specified.'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory. Overrides config if specified.'
    )

    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility.

    Args:
        seed: Random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)
    import random
    random.seed(seed)

    # Make CUDA operations deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = Config.from_yaml(args.config)

    # Validate configuration
    try:
        validate_config(config)
    except ValueError as e:
        print(f"Configuration validation failed: {e}")
        sys.exit(1)

    # Override config with command line arguments
    if args.device is not None:
        config.experiment['device'] = args.device

    if args.output_dir is not None:
        config.experiment['output_dir'] = args.output_dir

    # Get experiment settings
    device = config.experiment.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(config.experiment['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration to output directory
    config.save(output_dir / 'config.yaml')

    # Setup logger
    logger = setup_logger(output_dir / 'logs', name='train')

    logger.info("=" * 70)
    logger.info("Multi-Temporal Lighting Compression Training")
    logger.info("=" * 70)
    logger.info(f"Experiment: {config.experiment['name']}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Random seed: {args.seed}")

    # Load dataset
    logger.info("\n" + "=" * 70)
    logger.info("Loading dataset")
    logger.info("=" * 70)

    dataset_path = Path(config.data['dataset_path'])
    logger.info(f"Dataset path: {dataset_path}")

    # Get split ratios from config
    train_ratio = config.data.get('train_ratio', 0.6)
    val_ratio = config.data.get('val_ratio', 0.2)

    # Create train and validation datasets
    # MultiTimeLightingDataset handles splitting internally
    train_dataset = MultiTimeLightingDataset(
        data_root=dataset_path,
        split='train',
        train_ratio=train_ratio,
        val_ratio=val_ratio
    )

    val_dataset = MultiTimeLightingDataset(
        data_root=dataset_path,
        split='val',
        train_ratio=train_ratio,
        val_ratio=val_ratio
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    logger.info(f"Number of time moments: {train_dataset.num_moments}")
    logger.info(f"Moments: {train_dataset.moments}")

    # Create data loaders
    batch_size = config.data.get('batch_size', 512)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device == 'cuda' else False
    )

    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Train batches per epoch: {len(train_loader)}")
    logger.info(f"Val batches per epoch: {len(val_loader)}")

    # Create model
    logger.info("\n" + "=" * 70)
    logger.info("Creating model")
    logger.info("=" * 70)

    model = MultiTimeCompressionModel(config.config)

    logger.info(f"Model architecture:")
    logger.info(f"  Gaussians: {config.model['num_gaussians']}")
    logger.info(f"  Base latent dim: {config.model['latent_dim_base']}")
    logger.info(f"  Time latent dim: {config.model['latent_dim_time']}")
    logger.info(f"  Temporal MLP: {config.model['temporal_mlp']}")
    logger.info(f"  Decoder MLP: {config.model['decoder_mlp']}")

    # Initialize Gaussians from probe positions
    logger.info("\nInitializing Gaussians from probe positions...")

    # Get all probe positions from dataset (need all positions, not just training split)
    # Combine train and val positions for initialization
    import numpy as np
    all_positions = np.vstack([train_dataset.positions, val_dataset.positions])

    model.initialize_gaussians(all_positions)
    logger.info(f"Initialized {model.gaussian_mixture.num_gaussians} Gaussians from {len(all_positions)} probe positions")

    # Create trainer
    logger.info("\n" + "=" * 70)
    logger.info("Creating trainer")
    logger.info("=" * 70)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        output_dir=output_dir,
        device=device
    )

    # Resume from checkpoint if specified
    if args.resume is not None:
        logger.info(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Start training
    logger.info("\n" + "=" * 70)
    logger.info("Starting training")
    logger.info("=" * 70)

    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("\n\nTraining interrupted by user")
        logger.info("Saving checkpoint...")
        trainer.save_checkpoint('checkpoint_interrupted.pt')
        logger.info("Checkpoint saved. Exiting.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        logger.info("Saving checkpoint...")
        trainer.save_checkpoint('checkpoint_error.pt')
        logger.info("Checkpoint saved. Exiting.")
        sys.exit(1)

    logger.info("\nTraining completed successfully!")


if __name__ == '__main__':
    main()
