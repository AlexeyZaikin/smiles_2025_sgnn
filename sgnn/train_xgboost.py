# from datetime import datetime
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)
import warnings
import hydra
from omegaconf import DictConfig
from pathlib import Path
import time
import csv
import logging
import pickle
from tqdm.auto import tqdm
import glob

warnings.filterwarnings("ignore")


def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging for XGBoost experiments with unique logger per experiment"""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create unique logger name based on log directory path
    logger_name = f"xgboost_experiment_{hash(str(log_dir))}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    
    # File handler
    file_handler = logging.FileHandler(log_dir / "experiment.log")
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def train_xgboost_model(X_train, y_train, X_val, y_val, logger, **xgb_params):
    """Train XGBoost model"""

    # Default parameters
    default_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0,
        'early_stopping_rounds': 10
    }

    default_params.update(xgb_params)
    logger.info(f"XGBoost parameters: {default_params}")

    model = xgb.XGBClassifier(**default_params)

    eval_set = [(X_train, y_train), (X_val, y_val)]
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=False
    )
    return model



def grid_search_xgboost(X_train, y_train, logger, cfg=None):
    """Perform GridSearch to find best hyperparameters"""
    
    # Define default parameter grid for GridSearch
    param_grid = {
        'max_depth': [3, 4, 5, 6, 7],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [50, 100, 200],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2]
    }
    
    # Base parameters
    base_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0
    }
    
    logger.info("Starting GridSearch for hyperparameter tuning...")
    logger.info(f"Parameter grid: {param_grid}")
    
    # Create base model
    base_model = xgb.XGBClassifier(**base_params)
    
    # Get CV folds from config or use default
    cv_folds = cfg.xgboost.gridsearch.cv_folds
    
    # Perform GridSearch
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv_folds,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit GridSearch
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    search_time = (time.time() - start_time) / 60
    
    # Get best parameters and score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    logger.info(f"GridSearch completed in {search_time:.2f} minutes")
    logger.info(f"Best CV score: {best_score:.4f}")
    logger.info(f"Best parameters: {best_params}")
    
    # Create model with best parameters
    best_model = xgb.XGBClassifier(**base_params, **best_params)
    
    return best_model, best_params, best_score, search_time


def compute_metrics(y_true, y_pred, y_proba):
    """Calculate comprehensive classification metrics"""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    
    # ROC-AUC calculation
    metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    
    # Precision-Recall AUC
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    metrics["pr_auc"] = auc(recall, precision)
    
    return metrics


def main_loop(cfg: DictConfig, dataset_name: str, base_dir: Path, dataset_path: str | list):
    """Main loop for processing a single dataset"""
    
    # Check if GridSearch is enabled
    use_gridsearch = cfg.xgboost.gridsearch.enabled

    try:
        dataset_dir = base_dir / "xgboost"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        logger = setup_logging(dataset_dir)
        logger.info(f"Starting XGBoost experiment: {dataset_name}")
        logger.info(f"GridSearch enabled: {use_gridsearch}")

        # Load data
        if isinstance(dataset_path, str):
            data = pd.read_csv(dataset_path)
            data_split = pd.read_csv(dataset_path.replace('.node_features.csv', '.graph.csv'))

            train_data = data.loc[~data_split.iloc[-1, 2:].astype(bool).reset_index(drop=True), :]
            test_data = data.loc[data_split.iloc[-1, 2:].astype(bool).reset_index(drop=True), :]
            X_train = train_data.drop('target', axis=1)
            y_train = train_data['target'].astype(int)
            X_test = test_data.drop('target', axis=1)
            y_test = test_data['target'].astype(int)
        elif isinstance(dataset_path, list):
            X_train = []
            y_train = []
            X_test = []
            y_test = []
            for dataset_name in dataset_path:
                data = pd.read_csv(dataset_name)
                data_split = pd.read_csv(dataset_name.replace('.node_features.csv', '.graph.csv'))
                train_data = data.loc[~data_split.iloc[-1, 2:].astype(bool).reset_index(drop=True), :]
                test_data = data.loc[data_split.iloc[-1, 2:].astype(bool).reset_index(drop=True), :]
                X_train.append(train_data.drop('target', axis=1))
                y_train.append(train_data['target'].astype(int))
                X_test.append(test_data.drop('target', axis=1))
                y_test.append(test_data['target'].astype(int))
            X_train = pd.concat(X_train)
            y_train = pd.concat(y_train)
            X_test = pd.concat(X_test)
            y_test = pd.concat(y_test)
        
        data_size = len(X_train) + len(X_test)
        logger.info(f"Data size: {data_size}")
        logger.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        logger.info(f"Features: {X_train.shape[1]}")
        logger.info(f"Class distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")

        if use_gridsearch:
            # Use GridSearch to find best hyperparameters
            logger.info("Using GridSearch for hyperparameter optimization...")
                        
            model, best_params, best_cv_score, search_time = grid_search_xgboost(
                X_train, y_train, logger, cfg
            )
            
            # Train final model with best parameters on full training data
            logger.info("Training final model with best parameters on full training data...")
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = (time.time() - start_time) / 60 + search_time
            
            # Log best parameters
            logger.info(f"Best CV score: {best_cv_score:.4f}")
            logger.info(f"Best parameters: {best_params}")
            
        else:
            # Use default parameters from config
            xgb_params = {
                'max_depth': cfg.xgboost.max_depth,
                'learning_rate': cfg.xgboost.learning_rate,
                'n_estimators': cfg.xgboost.n_estimators,
                'subsample': cfg.xgboost.subsample,
                'colsample_bytree': cfg.xgboost.colsample_bytree
            }

            # Train model with default parameters
            X_train, y_train, X_val, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)
            start_time = time.time()
            model = train_xgboost_model(X_train, y_train, X_val, y_val, logger, **xgb_params)
            train_time = (time.time() - start_time) / 60
            best_params = xgb_params
            best_cv_score = None

        # Predictions on test set
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Compute metrics
        test_metrics = compute_metrics(y_test, y_pred, y_proba)

        # Log results
        logger.info(f"\n{'=' * 50}")
        logger.info(f"FINAL RESULTS: {dataset_name}/XGBoost")
        logger.info(f"Data size: {data_size}")
        if use_gridsearch:
            logger.info(f"Best CV score: {best_cv_score:.4f}")
            logger.info(f"Best parameters: {best_params}")
        else:
            logger.info(f"Max depth: {best_params['max_depth']}")
            logger.info(f"Learning rate: {best_params['learning_rate']}")
            logger.info(f"N estimators: {best_params['n_estimators']}")
        logger.info(f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
        logger.info(f"Test F1: {test_metrics['f1']:.4f}")
        logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"Test PR-AUC: {test_metrics['pr_auc']:.4f}")
        logger.info(f"Total time: {train_time:.2f} minutes")
        logger.info("=" * 50)

        # Save results to CSV file
        results_file = dataset_dir / f"results.csv"
        
        # Check if headers need to be created
        file_exists = results_file.exists()
        
        row = [
            dataset_name,
            "XGBoost_GridSearch" if use_gridsearch else "XGBoost",
            f"{test_metrics['roc_auc']:.4f}",
            f"{test_metrics['f1']:.4f}",
            f"{test_metrics['accuracy']:.4f}",
            f"{test_metrics['pr_auc']:.4f}",
            f"{train_time:.2f}",
            f"{data_size}"
        ]
        
        with open(results_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write headers if file is new
            if not file_exists:
                headers = [
                    "dataset_name", "model_type", "test_roc_auc", 
                    "test_f1", "test_accuracy", "test_pr_auc", 
                    "train_time_minutes", "data_size"
                ]
                writer.writerow(headers)
            writer.writerow(row)
        
        logger.info(f"Results appended to {results_file}")

        # Save model
        model_path = dataset_dir / "xgboost_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model saved to {model_path}")
        
        # Save best parameters if GridSearch was used
        if use_gridsearch:
            params_path = dataset_dir / "best_parameters.json"
            import json
            with open(params_path, 'w') as f:
                json.dump(best_params, f, indent=2)
            logger.info(f"Best parameters saved to {params_path}")

    except Exception as e:
        print(f"Error processing dataset {dataset_name}: {str(e)}")
        if 'logger' in locals():
            logger.error(f"Error processing dataset {dataset_name}: {str(e)}")


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """XGBoost training script for tabular data"""
    
    # Initialize output directory
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(cfg.save_path) / "logs" / str(cfg.data.dataset_size)

    # Load datasets
    dataset_paths = glob.glob(f"{cfg.data.dataset_path}/csv_{cfg.data.dataset_size}/*.node_features.csv")
    dataset_names = [path.split("/")[-1].split(".")[0] for path in dataset_paths]
    
    if cfg.per_dataset:
        # Process each dataset separately
        for i, dataset_name in enumerate(dataset_names):
            cur_dir = base_dir / dataset_name
            main_loop(cfg, dataset_name, cur_dir, dataset_paths[i])
    else:        
        cur_dir = base_dir / "combined_datasets"
        # Process combined data
        main_loop(cfg, "combined_datasets", cur_dir, dataset_paths)


if __name__ == "__main__":
    main()