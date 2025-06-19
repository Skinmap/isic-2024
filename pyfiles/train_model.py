import os
import sys
import time
import copy
import gc
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torcheval.metrics.functional import binary_auroc
from sklearn.model_selection import StratifiedGroupKFold

# Import local modules
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from utils import set_seed, print_trainable_parameters
from models import setup_model, ISICModel, ISICModelEdgnet
from training import fetch_scheduler, train_one_epoch, valid_one_epoch, run_training
from datasets import prepare_loaders
from augmentations import get_augmentations

def main():
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")

    # Configuration
    MODEL_NAME = "EDGENEXT"  # or "EVA"
    CONFIG = {
        "seed": 42 if MODEL_NAME == 'EVA' else 1997,
        "epochs": 500,
        "img_size": 336 if MODEL_NAME == 'EVA' else 256,
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

    # Path setup
    original_root = Path('/data/original')
    data_artifacts = "/data/artifacts"
    os.makedirs(data_artifacts, exist_ok=True)

    # Load data
    train_path = original_root / 'train-metadata.csv'
    df_train = pd.read_csv(train_path)
    df_train["path"] = str(original_root / 'train-image/image') + '/' + df_train['isic_id'] + ".jpg"

    # Model setup
    model_name = "eva02_small_patch14_336.mim_in22k_ft_in1k" if MODEL_NAME == 'EVA' else "edgenext_base.in21k_ft_in1k"
    model_maker = ISICModel if MODEL_NAME == 'EVA' else ISICModelEdgnet
    
    # Get augmentations
    data_transforms = get_augmentations(CONFIG)

    # Loss function
    def criterion(outputs, targets):
        return nn.BCELoss()(outputs, targets)

    # Training function
    def train_model():
        tsp = StratifiedGroupKFold(5, shuffle=True, random_state=CONFIG['seed'])
        results_list = []
        fold_df_valid_list = []

        for fold_n, (train_index, val_index) in enumerate(tsp.split(df_train, y=df_train.target, groups=df_train[CONFIG["group_col"]])):
            print(f"\nStarting fold {fold_n + 1}/5")
            
            fold_df_train = df_train.iloc[train_index].reset_index(drop=True)
            fold_df_valid = df_train.iloc[val_index].reset_index(drop=True)

            set_seed(CONFIG['seed'])
            model = setup_model(model_name, drop_path_rate=0, drop_rate=0, model_maker=model_maker)
            print_trainable_parameters(model)

            train_loader, valid_loader = prepare_loaders(fold_df_train, fold_df_valid, CONFIG, data_transforms)

            optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], 
                               weight_decay=CONFIG['weight_decay'])
            scheduler = fetch_scheduler(optimizer, CONFIG)

            model, history = run_training(
                train_loader, valid_loader,
                model, optimizer, scheduler,
                device=CONFIG['device'],
                num_epochs=CONFIG['epochs'],
                CONFIG=CONFIG, 
                tolerance_max=20,
                test_every_nth_step=lambda x: 5,
                seed=CONFIG['seed'])

            # Save model
            model_folder = f"./models/oof_{MODEL_NAME.lower()}_base"
            os.makedirs(model_folder, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(model_folder, f"model__{fold_n}"))
            results_list.append(np.max(history['Valid Kaggle metric']))

            # Validation metrics
            val_epoch_loss, val_epoch_auroc, val_epoch_custom_metric, tmp_predictions_all, tmp_targets_all = valid_one_epoch(
                model, 
                valid_loader, 
                device=CONFIG['device'], 
                epoch=1, 
                optimizer=optimizer, 
                criterion=criterion, 
                use_custom_score=True,
                metric_function=binary_auroc, 
                num_classes=1,
                return_preds=True)

            # Save predictions
            fold_df_valid['tmp_targets_all'] = tmp_targets_all
            fold_df_valid['tmp_predictions_all'] = tmp_predictions_all
            fold_df_valid['fold_n'] = fold_n
            fold_df_valid_list.append(fold_df_valid)

        # Combine and save all fold results
        fold_df_valid_list = pd.concat(fold_df_valid_list).reset_index(drop=True)
        fold_df_valid_list.to_parquet(f'/data/artifacts/oof_forecasts_{MODEL_NAME.lower()}_base.parquet')
        
        return results_list, fold_df_valid_list

    # Run training
    results, oof_forecasts = train_model()
    print(f"\nTraining complete. Average validation metric: {np.mean(results):.4f}")

if __name__ == "__main__":
    main()
