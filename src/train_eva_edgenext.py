import os
import time
import argparse
from pathlib import Path
from datetime import datetime
import logging
import json

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torcheval.metrics.functional import binary_auroc
from sklearn.model_selection import StratifiedGroupKFold

from utils import set_seed, log_trainable_parameters
from models import setup_model, ISICModel, ISICModelEdgnet
from training import fetch_scheduler, valid_one_epoch, run_training
from datasets import prepare_loaders
from augmentations import get_augmentations

class MetricsLogger:
    """Logger for structured metrics tracking"""
    def __init__(self, log_file):
        self.log_file = log_file
        self.metrics = []
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    def log_epoch(self, fold, epoch, metrics_dict):
        """Log metrics in structured format"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'fold': fold,
            'epoch': epoch,
            **metrics_dict
        }
        self.metrics.append(entry)
        
        # Append to JSONL file for easy parsing
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    
    def log_fold_summary(self, fold, best_metric, total_epochs, training_time):
        """Log fold completion summary"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'event': 'fold_complete',
            'fold': fold,
            'best_metric': best_metric,
            'total_epochs': total_epochs,
            'training_time_seconds': training_time,
            'training_time_minutes': training_time / 60
        }
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')


def setup_basic_logging(model_name):
    """Set up basic logging configuration"""
    # Create logger
    logger = logging.getLogger('ISICTraining')
    logger.setLevel(logging.INFO)
    
    # Create logs directory
    os.makedirs('./logs', exist_ok=True)
    
    # File handler - saves all INFO logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(f'./logs/{model_name.lower()}_training_{timestamp}.log')
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, 
                       choices=['EVA', 'EDGENEXT'],
                       help='Model type to train (EVA or EDGENEXT)')
    args = parser.parse_args()

    logger = setup_basic_logging(args.model)
    metrics_logger = MetricsLogger(f"./logs/{args.model.lower()}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jsonl")

    logger.info(f"Starting ISIC 2024 Training Pipeline - Model: {args.model}")
    logger.info("=" * 60)
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")

    # Model configurations
    MODEL_CONFIGS = {
        'EVA': {
            'seed': 42,
            'img_size': 336,
            'model_name': "eva02_small_patch14_336.mim_in22k_ft_in1k",
            'model_maker': ISICModel
        },
        'EDGENEXT': {
            'seed': 1997,
            'img_size': 256,
            'model_name': "edgenext_base.in21k_ft_in1k",
            'model_maker': ISICModelEdgnet
        }
    }

    # Shared configuration
    CONFIG = {
        "epochs": 500,
        "train_batch_size": 32,
        "valid_batch_size": 64,
        "learning_rate": 1e-4,
        "scheduler": 'CosineAnnealingLR',
        "min_lr": 1e-6,
        "T_max": 2000,
        "weight_decay": 1e-6,
        "fold": 0,
        "n_fold": 5,
        "n_accumulate": 1,
        "group_col": 'patient_id',
        "device": device
    }

    # Merge model-specific config
    model_config = MODEL_CONFIGS[args.model]
    CONFIG.update({
        'seed': model_config['seed'],
        'img_size': model_config['img_size']
    })

    logger.info(f"Configuration: {json.dumps({k: str(v) for k, v in CONFIG.items()}, indent=2)}")

    # Path setup
    original_root = Path('/home/ubuntu/skinmap/feature-generation/generated_data')
    data_artifacts = "/data/10ktests"
    os.makedirs(data_artifacts, exist_ok=True)

    # Load data
    logger.info("Loading training data...")
    train_path = original_root / 'filtered_train_calculated_umaneo_metrics.csv'
    df_train = pd.read_csv(train_path)
    df_train["path"] = '/data/isic-data/isic-2024-challenge/train-image/image' + '/' + df_train['isic_id'] + ".jpg"

    positive_cases = df_train['target'].sum()
    total_cases = len(df_train)
    logger.info(f"Dataset loaded - Total samples: {total_cases}")
    logger.info(f"Positive cases: {positive_cases} ({positive_cases/total_cases*100:.2f}%)")
    logger.info(f"Negative cases: {total_cases - positive_cases} ({(total_cases-positive_cases)/total_cases*100:.2f}%)")
    logger.info(f"Imbalance ratio: {(total_cases - positive_cases) / positive_cases:.2f}:1")

    # Model setup
    model_name = model_config['model_name']
    model_maker = model_config['model_maker']

    # Get augmentations
    data_transforms = get_augmentations(CONFIG)

    # Loss function
    def criterion(outputs, targets):
        return nn.BCELoss()(outputs, targets)

    # Training function
    def train_model():
        logger.info("Initializing 5-fold cross-validation...")
        tsp = StratifiedGroupKFold(5, shuffle=True, random_state=CONFIG['seed'])
        results_list = []
        fold_df_valid_list = []
        overall_start_time = time.time()

        for fold_n, (train_index, val_index) in enumerate(tsp.split(df_train, y=df_train.target, groups=df_train[CONFIG["group_col"]])):
            fold_start_time = time.time()
            logger.info(f"\n{'='*60}")
            logger.info(f"Starting Fold {fold_n + 1}/5")
            logger.info(f"Train samples: {len(train_index)}, Validation samples: {len(val_index)}")

            fold_df_train = df_train.iloc[train_index].reset_index(drop=True)
            fold_df_valid = df_train.iloc[val_index].reset_index(drop=True)

            fold_pos_train = fold_df_train['target'].sum()
            fold_pos_val = fold_df_valid['target'].sum()
            logger.info(f"Fold {fold_n + 1} - Train positives: {fold_pos_train}, Val positives: {fold_pos_val}")

            set_seed(CONFIG['seed'])
            model = setup_model(model_name, drop_path_rate=0, drop_rate=0, model_maker=model_maker)
            logger.info(f"Model initialized: {model_name}")
            log_trainable_parameters(model, logger)

            train_loader, valid_loader = prepare_loaders(fold_df_train, fold_df_valid, CONFIG, data_transforms)

            optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], 
                               weight_decay=CONFIG['weight_decay'])
            scheduler = fetch_scheduler(optimizer, CONFIG)

            logger.info("Starting training...")
            model, history = run_training(
                train_loader, valid_loader,
                model, optimizer, scheduler,
                device=CONFIG['device'],
                num_epochs=CONFIG['epochs'],
                CONFIG=CONFIG, 
                tolerance_max=20,
                test_every_nth_step=lambda x: 5,
                seed=CONFIG['seed'],
                logger=logger)

            # Save model
            model_folder = f"/data/10kmodels/oof_{args.model.lower()}_base"
            os.makedirs(model_folder, exist_ok=True)
            model_path = os.path.join(model_folder, f"model__{fold_n}")
            torch.save(model.state_dict(), os.path.join(model_folder, model_path))
            results_list.append(np.max(history['Valid Kaggle metric']))
            logger.info(f"Model saved to: {model_path}")

            best_metric = np.max(history["Valid Kaggle metric"])
            best_epoch = np.argmax(history["Valid Kaggle metric"])
            results_list.append(best_metric)
            logger.info(f"Best validation metric: {best_metric} at epoch {best_epoch}")

            # Validation metrics
            logger.info("Running final validation")
            val_epoch_loss, val_epoch_auroc, val_epoch_custom_metric, tmp_predictions_all, tmp_targets_all = valid_one_epoch(
                model, 
                dataloader=valid_loader, 
                device=CONFIG['device'], 
                epoch=1, 
                optimizer=optimizer, 
                criterion=criterion, 
                use_custom_score=True,
                metric_function=binary_auroc, 
                num_classes=1,
                return_preds=True)

            logger.info(f"Final validation - Loss: {val_epoch_loss}, AUROC: {val_epoch_auroc}, Custom metric: {val_epoch_custom_metric}")

            # Save predictions
            fold_df_valid['tmp_targets_all'] = tmp_targets_all
            fold_df_valid['tmp_predictions_all'] = tmp_predictions_all
            fold_df_valid['fold_n'] = fold_n
            fold_df_valid_list.append(fold_df_valid)

            fold_time = time.time() - fold_start_time
            metrics_logger.log_fold_summary(fold_n, best_metric, len(history['Train Loss']), fold_time)
            logger.info(f"Fold {fold_n + 1} completed in {fold_time/60:.2f} minutes")

        # Combine and save all fold results
        logger.info("\nSaving out-of-fold predictions...")
        fold_df_valid_list = pd.concat(fold_df_valid_list).reset_index(drop=True)
        oof_path = f'/data/10ktests/oof_forecasts_{args.model.lower()}_base.parquet'
        fold_df_valid_list.to_parquet(oof_path)
        logger.info(f"OOF predictions saved to {oof_path}")

        total_time = time.time() - overall_start_time
        logger.info("\n" + "="*60)
        logger.info("TRAINING COMPLETE")
        logger.info("="*60)
        logger.info(f"Total training time: {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)")
        logger.info(f"Average validation metric: {np.mean(results_list):.4f} ± {np.std(results_list):.4f}")
        logger.info(f"Best fold: {np.argmax(results_list) + 1} (metric: {max(results_list):.4f})")
        logger.info(f"Fold metrics: {[f'{x:.4f}' for x in results_list]}")

    try:
        train_model()
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)

if __name__ == "__main__":
    main()
