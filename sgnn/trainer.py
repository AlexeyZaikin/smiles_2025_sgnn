import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import torch.nn as nn
import logging
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    f1_score,
    precision_score,
    recall_score,
)
import time
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
import warnings
from omegaconf import DictConfig
from pathlib import Path
from typing import Dict, Tuple, Any

warnings.filterwarnings("ignore")


class GNNTrainer:
    """Optimized GNN trainer with automatic configuration"""

    def __init__(self, cfg: DictConfig, device: str = "cuda"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.metric_history = defaultdict(list)

    def train_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> Tuple[float, torch.Tensor, torch.Tensor]:
        """Efficient training loop with autocast"""
        model.train()
        total_loss = 0
        all_probs, all_labels = [], []

        for batch in loader:
            batch = batch.to(self.device)
            optimizer.zero_grad(set_to_none=True)

            # Mixed precision training
            with torch.amp.autocast(
                enabled=self.cfg.training.mixed_precision, device_type=str(self.device)
            ):
                out = model(batch)
                loss = criterion(out, batch.y.long())

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            probs = F.softmax(out, dim=1).detach().cpu()
            all_probs.append(probs)
            all_labels.append(batch.y.cpu())

        all_probs = torch.cat(all_probs)
        all_labels = torch.cat(all_labels)
        return total_loss / len(loader.dataset), all_probs, all_labels

    @torch.no_grad()
    def evaluate(
        self, model: nn.Module, loader: DataLoader, criterion: nn.Module
    ) -> Dict[str, float]:
        """Comprehensive model evaluation"""
        model.eval()
        total_loss = 0
        all_probs, all_preds, all_labels = [], [], []

        for batch in loader:
            batch = batch.to(self.device)
            out = model(batch)
            loss = criterion(out, batch.y.long())

            total_loss += loss.item() * batch.num_graphs
            probs = F.softmax(out, dim=1).cpu()
            preds = probs.argmax(dim=1)

            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(batch.y.cpu())

        all_probs = torch.cat(all_probs)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        metrics = self._compute_metrics(all_probs, all_preds, all_labels)
        metrics["loss"] = total_loss / len(loader.dataset)
        return metrics

    def _compute_metrics(
        self, probs: torch.Tensor, preds: torch.Tensor, labels: torch.Tensor
    ) -> Dict[str, Any]:
        """Calculate comprehensive classification metrics"""
        labels_np = labels.numpy()
        probs_np = probs.numpy()
        preds_np = preds.numpy()

        metrics = {
            "accuracy": (preds_np == labels_np).mean(),
            "precision": precision_score(
                labels_np, preds_np, average="binary", pos_label=1, zero_division=0
            ),
            "recall": recall_score(
                labels_np, preds_np, average="binary", pos_label=1, zero_division=0
            ),
            "f1": f1_score(
                labels_np, preds_np, average="binary", pos_label=1, zero_division=0
            ),
        }

        # ROC-AUC calculation
        try:
            metrics["roc_auc"] = roc_auc_score(labels_np, probs_np[:, 1])
        except ValueError:
            metrics["roc_auc"] = 0.5

        # Precision-Recall AUC
        precision, recall, _ = precision_recall_curve(labels_np, probs_np[:, 1])
        metrics["pr_auc"] = auc(recall, precision)

        return metrics

    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        logger: logging.Logger,
        tb_writer: SummaryWriter,
        log_dir: Path,
    ) -> Tuple[Dict[str, list], nn.Module]:
        """Training loop with early stopping and checkpointing"""
        best_val_f1 = 0
        early_stop_counter = 0
        history = defaultdict(list)
        epochs = self.cfg.training.max_epochs
        patience = self.cfg.training.patience

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            # Training phase
            train_loss, train_probs, train_labels = self.train_epoch(
                model, train_loader, optimizer, criterion
            )
            train_metrics = self._compute_metrics(
                train_probs, train_probs.argmax(dim=1), train_labels
            )

            # Validation phase
            val_metrics = self.evaluate(model, val_loader, criterion)

            # Update learning rate scheduler
            scheduler.step(val_metrics["f1"])

            # Track history
            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["train_roc_auc"].append(train_metrics["roc_auc"])
            history["train_f1"].append(train_metrics["f1"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_roc_auc"].append(val_metrics["roc_auc"])
            history["val_f1"].append(val_metrics["f1"])
            history["lr"].append(optimizer.param_groups[0]["lr"])

            # TensorBoard logging
            tb_writer.add_scalar("Loss/train", train_loss, epoch)
            tb_writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
            tb_writer.add_scalar("ROC-AUC/val", val_metrics["roc_auc"], epoch)
            tb_writer.add_scalar("F1/val", val_metrics["f1"], epoch)
            tb_writer.add_scalar("Learning Rate", history["lr"][-1], epoch)

            # Checkpointing
            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                early_stop_counter = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_f1": best_val_f1,
                    },
                    log_dir / "best_model.pth",
                )
                logger.info(f"New best model at epoch {epoch}: F1 = {best_val_f1:.4f}")
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            # Log progress
            if epoch % self.cfg.training.log_interval == 0:
                epoch_time = time.time() - start_time
                logger.info(
                    f"Epoch {epoch:03d} | Time: {epoch_time:.1f}s | "
                    f"LR: {history['lr'][-1]:.6f} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} | "
                    f"Val F1: {val_metrics['f1']:.4f}"
                )

        # Load best model
        checkpoint = torch.load(log_dir / "best_model.pth")
        model.load_state_dict(checkpoint["model_state_dict"])
        return history, model
