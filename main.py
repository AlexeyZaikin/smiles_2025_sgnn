from datetime import datetime
from itertools import product
import pickle
import torch
from tqdm.auto import tqdm
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
from sgnn.node_features_utils import add_node_features
from sgnn.sparsify_utils import get_sparsify_f_list
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
            "residual": trial.suggest_categorical("residual", [True, False]),
            "use_classifier_mlp": trial.suggest_categorical(
                "use_classifier_mlp", cfg.hparams.use_classifier_mlp.options
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

    if cfg.model.type == "GCN":
        params["model"]["use_edge_encoder"] = False
    else:
        params["model"]["use_edge_encoder"] = (
            trial.suggest_categorical(
                "use_edge_encoder", cfg.hparams.use_edge_encoder.options
            ),
        )

    if cfg.model.type in ["GATv2", "Transformer"]:
        params["model"]["heads"] = trial.suggest_int(
            "heads", cfg.hparams.heads.min, cfg.hparams.heads.max
        )
        params["model"]["concat"] = trial.suggest_categorical(
            "concat", cfg.hparams.concat.options
        )

    if params["model"]["use_edge_encoder"]:
        params["model"]["edge_encoder_channels"] = trial.suggest_int(
            "edge_encoder_channels",
            cfg.hparams.edge_encoder_channels.min,
            cfg.hparams.edge_encoder_channels.max,
        )
        params["model"]["edge_encoder_layers"] = trial.suggest_int(
            "edge_encoder_layers",
            cfg.hparams.edge_encoder_layers.min,
            cfg.hparams.edge_encoder_layers.max,
        )

    if params["model"]["use_classifier_mlp"]:
        params["model"]["classifier_mlp_channels"] = trial.suggest_int(
            "classifier_mlp_channels",
            cfg.hparams.classifier_mlp_channels.min,
            cfg.hparams.classifier_mlp_channels.max,
        )
        params["model"]["classifier_mlp_layers"] = trial.suggest_int(
            "classifier_mlp_layers",
            cfg.hparams.classifier_mlp_layers.min,
            cfg.hparams.classifier_mlp_layers.max,
        )

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

    for fold, (train_idx, val_idx) in tqdm(
        enumerate(skf.split(full_data, labels)), desc="CV Fold"
    ):
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
            in_channels=full_data[0].x.shape[-1],
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
    dataset_names = cfg.data.datasets
    all_data = pickle.load(open(dataset_path, "rb"))

    for dataset_name in dataset_names:
        data = all_data[dataset_name]
        dataset_dir = base_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # sparsify and add node features
        p_list = [0.8, 0.7, 0.3]
        sparsify_functions_list = get_sparsify_f_list(p_list)
        for sparsify_f, node_features in product(
            sparsify_functions_list, [True, False]
        ):
            data = sparsify_f(data)
            if node_features:
                data = add_node_features(data)

            cfg.data.sparsify = sparsify_f.__name__
            cfg.data.node_features = node_features

            # Prepare data
            train_data = data["train"]
            test_data = data["test"]
            full_data = train_data  # For cross-validation

            for model_type in cfg.hparams.model_type:
                model_dir = (
                    dataset_dir
                    / model_type
                    / cfg.data.sparsify
                    / f"node_features_{cfg.data.node_features}"
                )
                model_dir.mkdir(parents=True, exist_ok=True)

                # Set up logging
                logger, tb_writer = setup_logging(model_dir)
                logger.info(
                    f"Starting experiment: {dataset_name}/{model_type}/{cfg.data.sparsify}/node_features_{cfg.data.node_features}"
                )

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
                    test_loader,
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
