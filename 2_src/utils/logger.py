"""Logging utilities with Rich formatting for colored terminal output."""

import logging
from pathlib import Path
from datetime import datetime
from rich.logging import RichHandler
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
from typing import Optional
import torch


def setup_logger(
    log_dir: Path,
    name: str = 'training',
    level: int = logging.INFO
) -> logging.Logger:
    """Setup logger with Rich formatting and file output.

    Args:
        log_dir: Directory to save log files
        name: Logger name
        level: Logging level

    Returns:
        logger: Configured logger instance
    """
    # Create log directory
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Rich handler for console (colored output)
    console_handler = RichHandler(
        rich_tracebacks=True,
        markup=True,
        show_time=True,
        show_path=False
    )
    console_handler.setLevel(level)

    # File handler for logging to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'{name}_{timestamp}.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)

    # Formatter for file handler
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)

    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info(f"Logger initialized. Logging to {log_file}")

    return logger


def log_metrics_table(metrics: dict, title: str = "Metrics") -> None:
    """Display metrics in a formatted table using Rich.

    Args:
        metrics: Dictionary of metric name -> value
        title: Table title
    """
    console = Console()

    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", width=20)
    table.add_column("Value", style="green", width=15, justify="right")

    for name, value in metrics.items():
        if isinstance(value, float):
            table.add_row(name, f"{value:.6f}")
        else:
            table.add_row(name, str(value))

    console.print(table)


def create_progress_bar() -> Progress:
    """Create a Rich progress bar for training.

    Returns:
        progress: Configured Progress instance
    """
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=Console()
    )
    return progress


class TrainingLogger:
    """Logger for training progress and metrics.

    Combines file logging with Rich console output.
    """

    def __init__(self, log_dir: Path, name: str = 'training'):
        """Initialize training logger.

        Args:
            log_dir: Directory for log files
            name: Logger name
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup logger
        self.logger = setup_logger(self.log_dir, name=name)

        # Console for Rich output
        self.console = Console()

        # Metrics history
        self.train_metrics = []
        self.val_metrics = []

    def log_config(self, config: dict):
        """Log configuration parameters.

        Args:
            config: Configuration dictionary
        """
        self.logger.info("=" * 70)
        self.logger.info("Configuration:")
        self.logger.info("=" * 70)

        def log_dict(d, indent=0):
            for key, value in d.items():
                if isinstance(value, dict):
                    self.logger.info("  " * indent + f"{key}:")
                    log_dict(value, indent + 1)
                else:
                    self.logger.info("  " * indent + f"{key}: {value}")

        log_dict(config)
        self.logger.info("=" * 70)

    def log_step(self, step: int, metrics: dict, prefix: str = "Train"):
        """Log metrics for a single step.

        Args:
            step: Current step number
            metrics: Dictionary of metrics
            prefix: Prefix for log message (e.g., 'Train', 'Val')
        """
        # Format metrics string
        metrics_str = " | ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
        self.logger.info(f"{prefix} Step {step:6d} | {metrics_str}")

    def log_epoch(self, epoch: int, train_metrics: dict, val_metrics: Optional[dict] = None):
        """Log metrics for an entire epoch.

        Args:
            epoch: Epoch number
            train_metrics: Training metrics
            val_metrics: Validation metrics (optional)
        """
        self.logger.info("=" * 70)
        self.logger.info(f"Epoch {epoch} Summary")
        self.logger.info("=" * 70)

        # Log training metrics
        self.logger.info("Training Metrics:")
        for name, value in train_metrics.items():
            self.logger.info(f"  {name}: {value:.6f}")

        # Log validation metrics if provided
        if val_metrics is not None:
            self.logger.info("\nValidation Metrics:")
            for name, value in val_metrics.items():
                self.logger.info(f"  {name}: {value:.6f}")

        self.logger.info("=" * 70)

        # Store metrics
        self.train_metrics.append(train_metrics)
        if val_metrics is not None:
            self.val_metrics.append(val_metrics)

    def log_model_info(self, model: torch.nn.Module, num_probes: int, num_moments: int):
        """Log model architecture and size information.

        Args:
            model: Model instance
            num_probes: Number of probes in dataset
            num_moments: Number of time moments
        """
        self.logger.info("=" * 70)
        self.logger.info("Model Information")
        self.logger.info("=" * 70)

        # Model size
        size_bytes, size_mb = model.get_model_size()
        self.logger.info(f"Model size: {size_mb:.2f} MB ({size_bytes:,} bytes)")

        # Compression ratio
        ratio = model.get_compression_ratio(num_probes, num_moments)
        self.logger.info(f"Compression ratio: {ratio:.2f}x")

        # Parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")

        self.logger.info("=" * 70)

    def info(self, message: str):
        """Log info message.

        Args:
            message: Message to log
        """
        self.logger.info(message)

    def warning(self, message: str):
        """Log warning message.

        Args:
            message: Message to log
        """
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message.

        Args:
            message: Message to log
        """
        self.logger.error(message)
