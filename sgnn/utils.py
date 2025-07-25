import torch
import numpy as np
import random
import warnings
import pandas as pd
import logging
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Dict, Tuple

warnings.filterwarnings("ignore")


warnings.filterwarnings("ignore")


# Set global seeds for reproducibility
def set_global_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


set_global_seed()


def plot_metrics(history: Dict[str, list], log_dir: Path):
    """Create comprehensive metric visualizations"""
    plt.figure(figsize=(12, 8))

    # Loss curves
    plt.subplot(2, 2, 1)
    plt.plot(history["epoch"], history["train_loss"], "b-", label="Train")
    plt.plot(history["epoch"], history["val_loss"], "r-", label="Validation")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # ROC-AUC
    plt.subplot(2, 2, 2)
    plt.plot(history["epoch"], history["val_roc_auc"], "g-")
    plt.title("Validation ROC-AUC")
    plt.xlabel("Epoch")
    plt.ylabel("ROC-AUC")

    # F1 Score
    plt.subplot(2, 2, 3)
    plt.plot(history["epoch"], history["val_f1"], "m-")
    plt.title("Validation F1 Score")
    plt.xlabel("Epoch")
    plt.ylabel("F1")

    # Accuracy
    plt.subplot(2, 2, 4)
    plt.plot(history["epoch"], history["val_acc"], "c-")
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.yscale("log")

    plt.tight_layout()
    plt.savefig(log_dir / "training_metrics.png")
    plt.close()

    # Save history to CSV
    history_df = pd.DataFrame(history)
    history_df.to_csv(log_dir / "training_history.csv", index=False)


def setup_logging(log_dir: Path) -> Tuple[logging.Logger, SummaryWriter]:
    """Comprehensive logging setup"""
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("gnn_experiment")
    logger.setLevel(logging.DEBUG)

    # File handler
    file_handler = logging.FileHandler(log_dir / "experiment.log")
    file_handler.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # TensorBoard writer
    tb_writer = SummaryWriter(log_dir=log_dir)

    return logger, tb_writer
