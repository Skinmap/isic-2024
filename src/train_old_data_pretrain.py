import os
import time
import copy
import gc
import argparse
import hashlib
from pathlib import Path
from collections import defaultdict

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim

import numpy as np
import pandas as pd
import multiprocessing as mp
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from PIL import Image
from torcheval.metrics.functional import multiclass_auroc

from utils import set_seed, print_trainable_parameters
from models import setup_model
from training import fetch_scheduler, train_one_epoch, valid_one_epoch
from datasets import ISICDatasetSamplerMulticlass, ISICDatasetSimple
from augmentations import get_augmentations


def get_hash(file_name):
    """Generate MD5 hash of image file."""
    image_tmp = Image.open(file_name)
    md5hash = hashlib.md5(image_tmp.tobytes()).hexdigest()
    return str(md5hash)


def get_hash_df(df):
    """Generate hash dataframe for image deduplication."""
    image_hash = []
    for _, row in df.iterrows():
        image_hash.append(get_hash(row.path))
    
    return pd.DataFrame({
        "path": df.path,
        "image_hash": image_hash
    })


def resize_image(image, resize=512):
    """Resize image while maintaining aspect ratio."""
    w, h = image.size

    if h < w:
        h_new = resize
        w_new = int(h_new / h * w // 8 * 8)
    else:
        w_new = resize
        h_new = int(w_new / w * h // 8 * 8)

    image = image.resize((w_new, h_new))
    return image


def resize_images(df, path, size_thr=512):
    """Resize images and save to specified path."""
    for _, row in df.iterrows():
        img = Image.open(row.path)
        w, h = img.size

        if min(w, h) > size_thr:
            img = resize_image(img, resize=size_thr)
        img.save(os.path.join(path, row.isic_id + ".png"))


def criterion_mc(outputs, targets):
    """Multiclass cross-entropy loss."""
    return nn.CrossEntropyLoss()(outputs, targets)


def get_nth_test_step(x):
    """Always test every epoch for simplicity."""
    return 1


def run_training_pretrain(
        train_loader, valid_loader, model, optimizer, scheduler, device, num_epochs, 
        model_folder=None, model_name="", seed=42, tolerance_max=15, criterion=criterion_mc):
    """Run training loop for pretraining on external data."""
    set_seed(seed)
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_score = -np.inf
    history = defaultdict(list)
    tolerance = 0

    for epoch in range(1, num_epochs + 1): 
        if tolerance > tolerance_max:
            print(f"Early stopping at epoch {epoch} due to no improvement for {tolerance_max} epochs")
            break

        gc.collect()
        train_epoch_loss, train_epoch_auroc = train_one_epoch(
            model, 
            optimizer, 
            scheduler, 
            dataloader=train_loader, 
            device=device,
            CONFIG={'device': device, 'n_accumulate': 1},
            epoch=epoch, 
            criterion=criterion,
            metric_function=multiclass_auroc, 
            num_classes=3)

        val_epoch_loss, val_epoch_auroc, val_epoch_custom_metric = valid_one_epoch(
            model, 
            valid_loader, 
            device=device, 
            epoch=epoch, 
            optimizer=optimizer, 
            criterion=criterion, 
            use_custom_score=False,
            metric_function=multiclass_auroc, 
            num_classes=3)

        history['Train Loss'].append(train_epoch_loss)
        history['Valid Loss'].append(val_epoch_loss)
        history['Train AUROC'].append(train_epoch_auroc)
        history['Valid AUROC'].append(val_epoch_auroc)
        history['Valid Kaggle metric'].append(val_epoch_custom_metric)
        history['lr'].append(scheduler.get_lr()[0])

        if best_epoch_score <= val_epoch_auroc:
            tolerance = 0
            print(f"Validation AUROC Improved ({best_epoch_score:.4f} ---> {val_epoch_auroc:.4f})")
            best_epoch_score = val_epoch_auroc
            best_model_wts = copy.deepcopy(model.state_dict())
            if model_folder is not None:
                torch.save(model.state_dict(), os.path.join(model_folder, model_name))
        else:
            tolerance += 1

        print()

    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best AUROC: {:.4f}".format(best_epoch_score))    
    model.load_state_dict(best_model_wts)
    return model, history


def main():
    parser = argparse.ArgumentParser(description='Train model on external/old data for pretraining')
    parser.add_argument('--data-dir', type=str, default='../data/original', 
                       help='Path to original data directory')
    parser.add_argument('--external-images-dir', type=str, default='../images',
                       help='Path to external images directory')
    parser.add_argument('--model-dir', type=str, default='../models/pretraining',
                       help='Directory to save trained models')
    parser.add_argument('--artifacts-dir', type=str, default='../data/artifacts',
                       help='Directory to save artifacts')
    parser.add_argument('--model-name', type=str, default='eva02_small_patch14_336.mim_in22k_ft_in1k',
                       help='Model architecture name')
    parser.add_argument('--epochs', type=int, default=500,
                       help='Number of training epochs')
    parser.add_argument('--img-size', type=int, default=336,
                       help='Input image size')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Set up device and random seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")

    set_seed(args.seed)

    # Create directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.artifacts_dir, exist_ok=True)

    # Configuration
    CONFIG = {
        "seed": args.seed,
        "epochs": args.epochs,
        "img_size": args.img_size,
        "train_batch_size": args.batch_size,
        "valid_batch_size": 64,
        "learning_rate": args.learning_rate,
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

    # Load ISIC training data
    original_root = Path(args.data_dir)
    train_path = original_root / 'train-metadata.csv'
    df_train = pd.read_csv(train_path)
    df_train["path"] = os.path.join(args.data_dir, 'train-image/image/') + df_train['isic_id'] + ".jpg"

    print(f"ISIC training data loaded: {len(df_train)} samples")

    # Load external metadata
    metadata_path = os.path.join(args.external_images_dir, 'metadata.csv')
    if not os.path.exists(metadata_path):
        print(f"Error: External metadata file not found at {metadata_path}")
        return

    metadata_df = pd.read_csv(metadata_path)
    
    # Process external metadata
    metadata_df['diagnosis_pr'] = metadata_df.diagnosis.map({
        'nevus': 'nevus',
        'melanoma': 'melanoma',
        'basal cell carcinoma': 'bkl',
        'seborrheic keratosis': 'bkl',
        'solar lentigo': 'bkl',
        'lentigo NOS': 'bkl'
    })
    
    mask = (metadata_df.benign_malignant == 'benign') & (metadata_df.diagnosis_pr != 'bkl')
    metadata_df.loc[mask, 'diagnosis_pr'] = 'nevus'
    metadata_df["path"] = os.path.join(args.external_images_dir, '') + metadata_df['isic_id'] + ".jpg"
    
    print(f"External metadata loaded: {len(metadata_df)} samples")
    
    # Hash-based deduplication
    print("Performing hash-based deduplication...")
    hash_df = Parallel(n_jobs=mp.cpu_count())(delayed(get_hash_df)(df)
        for df in np.array_split(metadata_df, mp.cpu_count()*2))
    hash_df = pd.concat(hash_df).reset_index(drop=True)

    metadata_df = metadata_df.merge(hash_df, how="left", on=["path"])
    metadata_df = metadata_df.groupby('image_hash').first().reset_index(drop=True)

    # Map to target labels
    metadata_df["diagnosis_pr_target"] = metadata_df.diagnosis_pr.map({
        "nevus": 0,
        "bkl": 1,
        "melanoma": 2
    })
    metadata_df = metadata_df[~metadata_df.diagnosis_pr.isna()].reset_index(drop=True)
    metadata_df = metadata_df.rename(columns={'diagnosis_pr_target': 'target'})
    
    print(f"After deduplication and filtering: {len(metadata_df)} samples")
    
    # Resize external images
    resized_path = "../external_images_resized"
    os.makedirs(resized_path, exist_ok=True)
    
    print("Resizing external images...")
    Parallel(n_jobs=mp.cpu_count())(delayed(resize_images)(df, resized_path)
        for df in np.array_split(metadata_df, mp.cpu_count()*2))
    
    # Update paths to resized images
    metadata_df['path'] = resized_path + '/' + metadata_df['isic_id'] + '.png'
    metadata_df = metadata_df[
        metadata_df['path'].apply(lambda x: os.path.exists(x))
    ].reset_index(drop=True)
    
    print(f"Resized images available: {len(metadata_df)} samples")
    
    # Train/validation split
    train_pretrain_df, val_pretrain_df = train_test_split(
        metadata_df, test_size=0.2, shuffle=True, stratify=metadata_df.target, 
        random_state=CONFIG['seed'])
    
    print(f"Training split: {len(train_pretrain_df)} samples")
    print(f"Validation split: {len(val_pretrain_df)} samples")
    
    # Get data transforms
    data_transforms = get_augmentations(CONFIG)
    
    # Create datasets and data loaders
    train_dataset = ISICDatasetSamplerMulticlass(
        train_pretrain_df, transforms=data_transforms["train"], process_target=True, n_classes=3)
    valid_dataset = ISICDatasetSimple(
        val_pretrain_df, transforms=data_transforms["valid"], process_target=True, n_classes=3)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['train_batch_size'], 
                              num_workers=10, shuffle=True, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CONFIG['valid_batch_size'], 
                              num_workers=10, shuffle=False, pin_memory=True)
    
    # Setup model
    model = setup_model(args.model_name, num_classes=3, device=device)
    print_trainable_parameters(model)
    
    # Setup optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], 
                           weight_decay=CONFIG['weight_decay'])
    scheduler = fetch_scheduler(optimizer, CONFIG)
    
    # Train model
    print("Starting training...")
    model_save_name = "ema_small_pretrained_medium"
    model, history = run_training_pretrain(
        train_loader, valid_loader, 
        model, optimizer, scheduler,
        device=CONFIG['device'],
        num_epochs=CONFIG['epochs'],
        model_folder=args.model_dir,
        model_name=model_save_name,
        criterion=criterion_mc,
        seed=CONFIG['seed'])
    
    # Save final model
    final_model_path = os.path.join(args.model_dir, model_save_name)
    torch.save(model.state_dict(), final_model_path)
    print(f"Model saved to: {final_model_path}")
    
    # Generate predictions on ISIC training data
    print("Generating predictions on ISIC training data...")
    df_train_dataset = ISICDatasetSimple(
        df_train, transforms=data_transforms["valid"], process_target=True, n_classes=3)
    df_train_loader = DataLoader(df_train_dataset, batch_size=CONFIG['valid_batch_size'], 
                                 num_workers=5, shuffle=False, pin_memory=True)
    
    # Binary criterion for evaluation
    def criterion_binary(outputs, targets):
        return nn.BCELoss()(outputs, targets)
    
    val_epoch_loss, val_epoch_auroc, val_epoch_custom_metric, tmp_predictions_all, tmp_targets_all = valid_one_epoch(
        model, 
        df_train_loader, 
        device=CONFIG['device'], 
        epoch=1, 
        optimizer=optimizer, 
        criterion=criterion_binary, 
        use_custom_score=False,
        metric_function=multiclass_auroc, 
        num_classes=3,
        return_preds=True)

    df_train['old_set_0'] = tmp_predictions_all[:, 0]
    df_train['old_set_1'] = tmp_predictions_all[:, 1]
    df_train['old_set_2'] = tmp_predictions_all[:, 2]
    
    # Save predictions
    output_path = os.path.join(args.artifacts_dir, 'old_data_model_forecast_large.parquet')
    df_train[['isic_id', 'old_set_0', 'old_set_1', 'old_set_2']].to_parquet(output_path)
    print(f"Predictions saved to: {output_path}")
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
