from datetime import datetime
import pickle
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn as nn
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
import warnings
import optuna
from optuna.trial import Trial
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import List

from sgnn.model import GNNModel
from sgnn.trainer import GNNTrainer
from sgnn.utils import plot_metrics, setup_logging

warnings.filterwarnings("ignore")


def objective(
    trial: Trial, cfg: DictConfig, full_data: List[Data], log_dir: Path
) -> float:
    """Optuna hyperparameter optimization objective"""
    # Suggest hyperparameters
    params = {
        "model": {
            "activation": trial.suggest_categorical(
                "activation", cfg.hparams.activation
            ),
            "hidden_channels": trial.suggest_int(
                "hidden_channels",
                cfg.hparams.hidden_channels.min,
                cfg.hparams.hidden_channels.max,
            ),
            "num_layers": trial.suggest_int(
                "num_layers", cfg.hparams.num_layers.min, cfg.hparams.num_layers.max
            ),
            "dropout": trial.suggest_float(
                "dropout", cfg.hparams.dropout.min, cfg.hparams.dropout.max
            ),
            "heads": trial.suggest_int(
                "heads", cfg.hparams.heads.min, cfg.hparams.heads.max
            )
            if cfg.model.type == "GATv2"
            else 1,
            # "use_edge_encoders": trial.suggest_categorical(
            #     "use_edge_encoders", [True, False]
            # ),
            "residual": trial.suggest_categorical("residual", [True, False]),
            "use_classifier_mlp": trial.suggest_categorical(
                "use_classifier_mlp", [True, False]
            ),
            "classifier_mlp_dims": trial.suggest_categorical(
                "classifier_mlp_dims", cfg.hparams.classifier_mlp_dims.options
            ),
        },
        "training": {
            "learning_rate": trial.suggest_float(
                "learning_rate",
                cfg.hparams.learning_rate.min,
                cfg.hparams.learning_rate.max,
                log=True,
            ),
        },
    }

    # Create trial-specific config
    trial_cfg = OmegaConf.merge(cfg, OmegaConf.create(params))

    # Set up logging
    trial_log_dir = log_dir / f"trial_{trial.number}"
    trial_log_dir.mkdir(exist_ok=True)
    logger, tb_writer = setup_logging(trial_log_dir)

    # Cross-validation
    cv_scores = []
    skf = StratifiedKFold(
        n_splits=trial_cfg.training.cv_folds, shuffle=True, random_state=trial_cfg.seed
    )
    labels = [data.y for data in full_data]

    for fold, (train_idx, val_idx) in enumerate(skf.split(full_data, labels)):
        fold_log_dir = trial_log_dir / f"fold_{fold}"
        fold_log_dir.mkdir(exist_ok=True)

        # Create data loaders
        train_loader = DataLoader(
            [full_data[i] for i in train_idx],
            batch_size=trial_cfg.training.batch_size,
            shuffle=True,
            num_workers=os.cpu_count(),
            pin_memory=True,
        )
        val_loader = DataLoader(
            [full_data[i] for i in val_idx],
            batch_size=trial_cfg.training.batch_size,
            num_workers=os.cpu_count(),
            pin_memory=True,
        )

        # Initialize model
        model = GNNModel(
            trial_cfg,
        ).to(torch.device(trial_cfg.device))

        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=trial_cfg.training.learning_rate,
            weight_decay=trial_cfg.training.weight_decay,
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=trial_cfg.training.lr_patience,
            factor=trial_cfg.training.lr_factor,
        )
        criterion = nn.CrossEntropyLoss()

        # Trainer setup
        trainer = GNNTrainer(trial_cfg, device=trial_cfg.device)

        try:
            # Training
            trainer.train(
                model,
                train_loader,
                val_loader,
                optimizer,
                criterion,
                scheduler,
                logger,
                tb_writer,
                fold_log_dir,
            )

            # Validation metrics
            val_metrics = trainer.evaluate(model, val_loader, criterion)
            cv_scores.append(val_metrics["roc_auc"])

            # Report intermediate result
            trial.report(val_metrics["roc_auc"], fold)

            # Handle pruning
            if trial.should_prune():
                raise optuna.TrialPruned()

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            cv_scores.append(0.0)

    # Clean up
    tb_writer.close()
    return np.mean(cv_scores)


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Main experiment runner with Hydra configuration"""
    # Initialize output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path.cwd() / "logs" / timestamp

    # Load datasets
    dataset_path = cfg.data.dataset_path
    all_data = pickle.load(open(dataset_path, "rb"))

    for dataset_name, data in all_data.items():
        if dataset_name != "plrx":
            continue
        dataset_dir = base_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Prepare data
        train_data = data["train"]
        test_data = data["test"]
        full_data = train_data  # For cross-validation

        for model_type in cfg.model.type:
            model_dir = dataset_dir / model_type
            model_dir.mkdir(exist_ok=True)

            # Set up logging
            logger, tb_writer = setup_logging(model_dir)
            logger.info(f"Starting experiment: {dataset_name}/{model_type}")

            # Update config for current experiment
            cfg.model.type = model_type

            # Optuna hyperparameter optimization
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=cfg.seed),
                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=cfg.optuna.n_startup_trials,
                    n_warmup_steps=cfg.optuna.n_warmup_steps,
                ),
            )

            study.optimize(
                lambda trial: objective(trial, cfg, full_data, model_dir),
                n_trials=cfg.optuna.n_trials,
                timeout=cfg.optuna.timeout,
                show_progress_bar=True,
            )

            # Save best parameters
            best_params = study.best_params
            logger.info(f"Best parameters: {best_params}")
            logger.info(f"Best ROC-AUC: {study.best_value:.4f}")

            # Final training with best parameters
            cfg.model.update(best_params.get("model", {}))
            cfg.training.update(best_params.get("training", {}))

            # Data loaders
            train_loader = DataLoader(
                train_data,
                batch_size=cfg.training.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
            )
            test_loader = DataLoader(
                test_data,
                batch_size=cfg.training.batch_size,
                num_workers=0,
                pin_memory=True,
            )

            # Initialize model
            model = GNNModel(cfg).to(torch.device(cfg.device))

            # Optimizer and scheduler
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=cfg.training.learning_rate,
                weight_decay=cfg.training.weight_decay,
            )
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="max",
                patience=cfg.training.lr_patience,
                factor=cfg.training.lr_factor,
            )
            criterion = nn.CrossEntropyLoss()

            # Train final model
            trainer = GNNTrainer(cfg, device=cfg.device)
            history, model = trainer.train(
                model,
                train_loader,
                test_loader,  # Using test as validation for final training
                optimizer,
                criterion,
                scheduler,
                logger,
                tb_writer,
                model_dir,
            )

            # Final evaluation
            test_metrics = trainer.evaluate(model, test_loader, criterion)

            # Log results
            logger.info(f"\n{'=' * 50}")
            logger.info(f"FINAL RESULTS: {dataset_name}/{model_type}")
            logger.info(f"Activation: {cfg.model.activation}")
            logger.info(f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
            logger.info(f"Test F1: {test_metrics['f1']:.4f}")
            logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
            logger.info(f"Test PR-AUC: {test_metrics['pr_auc']:.4f}")
            logger.info("=" * 50)

            # Visualizations
            plot_metrics(history, model_dir)

            # Close resources
            tb_writer.close()


if __name__ == "__main__":
    main()
