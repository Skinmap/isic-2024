#!/usr/bin/env python3
"""
ISIC 2024 Inference Pipeline
Minimal inference script that reuses components from train_ensemble.py
"""

from train_ensemble import (
    Config,
    engineer_features,
    GradientBoostingPipeline,
    clean_data,
    add_lof_features,
    add_patient_norm,
    ISICModel,
    ISICModelEdgenext,
    generate_predictions,
    # DataLeakageChecker,
    setup_logging
)
import os
import sys
import pandas as pd
import numpy as np
import polars as pl
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import logging

# Global logger for compatibility with train_ensemble functions
LOGGER = None

# Initialize logger immediately for train_ensemble compatibility


def _initialize_logger():
    global LOGGER
    import train_ensemble
    if LOGGER is None:
        LOGGER = setup_logging("inference")
        # Set LOGGER in train_ensemble module for imported functions
        train_ensemble.LOGGER = LOGGER
    return LOGGER


# Initialize on import
_initialize_logger()


class ISICInferenceDataset(Dataset):
    """Dataset class for loading JPG images during inference"""

    def __init__(self, df, image_dir, transforms=None):
        self.df = df
        self.image_dir = Path(image_dir)
        self.isic_ids = df['isic_id'].values
        self.transforms = transforms

    def __len__(self):
        return len(self.isic_ids)

    def __getitem__(self, index):
        isic_id = self.isic_ids[index]
        img_path = self.image_dir / f"{isic_id}.jpg"

        # Load image
        img = np.array(Image.open(img_path).convert('RGB'))

        if self.transforms:
            img = self.transforms(image=img)["image"]

        return {
            'image': img,
            'target': 0,  # Dummy target for inference
        }


class InferenceConfig:
    """Modified config for inference"""

    # Image directory path
    image_dir = "/data/isic-data/isic-2024-challenge/train-image/image"

    # Model paths (use the same as training config)
    model_save_dir = Path("/data/models") / Config.run_tags

    # Deep learning model paths
    eva_model_path = Config.eva_model_path
    edg_model_path = Config.edg_model_path
    old_model_path = Config.old_model_path

    # Model patterns
    eva_model_pattern = Config.eva_model_pattern
    edg_model_pattern = Config.edg_model_pattern

    # Other config values
    device = Config.device
    eva_config = Config.eva_config
    edg_config = Config.edg_config
    id_col = Config.id_col
    cat_cols = Config.cat_cols
    num_cols = Config.num_cols
    lof_features = Config.lof_features
    columns_to_drop = Config.columns_to_drop
    err = Config.err


def extract_dl_features_jpg(test_df, image_dir, leakage_checker=None):
    """
    Extract deep learning features from JPG images for inference.
    Modified version of extract_dl_features that loads from JPG files.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting deep learning feature extraction from JPG images for {len(test_df)} samples")

    import time
    start_time = time.time()

    # Define transforms (same as training)
    transform_eva = A.Compose([
        A.Resize(InferenceConfig.eva_config['img_size'], InferenceConfig.eva_config['img_size']),
        A.Normalize(mean=[0.4815, 0.4578, 0.4082],
                    std=[0.2686, 0.2613, 0.2758],
                    max_pixel_value=255.0, p=1.0),
        ToTensorV2(),
    ], p=1.)

    transform_edg = A.Compose([
        A.Resize(InferenceConfig.edg_config['img_size'], InferenceConfig.edg_config['img_size']),
        A.Normalize(mean=[0.4815, 0.4578, 0.4082],
                    std=[0.2686, 0.2613, 0.2758],
                    max_pixel_value=255.0, p=1.0),
        ToTensorV2(),
    ], p=1.)

    test_df['target'] = 0  # Dummy target

    # 1. Old 3-class model predictions
    logger.info("Extracting features from old 3-class model...")
    dataset = ISICInferenceDataset(test_df, image_dir, transforms=transform_eva)
    dataloader = DataLoader(dataset, batch_size=InferenceConfig.eva_config['valid_batch_size'],
                            num_workers=2, shuffle=False, pin_memory=True)

    model = ISICModel(InferenceConfig.eva_config['model_name'], pretrained=False, num_classes=3)
    model.load_state_dict(torch.load(InferenceConfig.old_model_path, weights_only=True))
    model.to(InferenceConfig.device)

    _, predictions = generate_predictions(model, dataloader, InferenceConfig.device)
    test_df['old_set_0'] = predictions[:, 0]
    test_df['old_set_1'] = predictions[:, 1]
    test_df['old_set_2'] = predictions[:, 2]
    model.to('cpu')
    logger.info("Finished extracting old 3-class model features")

    # 2. EVA model predictions
    logger.info("Extracting features from EVA models...")
    eva_predictions = []
    for i in tqdm(range(5), desc="EVA model inference"):
        model_path = os.path.join(InferenceConfig.eva_model_path, InferenceConfig.eva_model_pattern.format(i))
        model = ISICModel(InferenceConfig.eva_config['model_name'], pretrained=False, num_classes=1)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.to(InferenceConfig.device)

        _, predictions = generate_predictions(model, dataloader, InferenceConfig.device)
        eva_predictions.append(predictions)
        model.to('cpu')

    # Load normalization stats from OOF data
    oof_eva = pd.read_parquet(Config.eva_oof_path)
    mean_pred = oof_eva.groupby('fold_n')['tmp_predictions_all'].mean().iloc[0]
    std_pred = oof_eva.groupby('fold_n')['tmp_predictions_all'].std().iloc[0]

    # Normalize EVA predictions
    for i in range(5):
        eva_predictions[i] = (eva_predictions[i] - mean_pred) / std_pred

    test_df['predictions_eva'] = np.mean(eva_predictions, axis=0)
    logger.info("Finished extracting EVA features")

    # 3. EdgeNext model predictions
    logger.info("Extracting features from EdgeNext models...")
    dataset = ISICInferenceDataset(test_df, image_dir, transforms=transform_edg)
    dataloader = DataLoader(dataset, batch_size=InferenceConfig.edg_config['valid_batch_size'],
                            num_workers=2, shuffle=False, pin_memory=True)

    edg_predictions = []
    for i in tqdm(range(5), desc="EdgeNext model inference"):
        model_path = os.path.join(InferenceConfig.edg_model_path, InferenceConfig.edg_model_pattern.format(i))
        model = ISICModelEdgenext('edgenext_base.in21k_ft_in1k', pretrained=False)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.to(InferenceConfig.device)

        _, predictions = generate_predictions(model, dataloader, InferenceConfig.device)
        edg_predictions.append(predictions)
        model.to('cpu')

    # Load normalization stats from OOF data
    oof_edg = pd.read_parquet(Config.edg_oof_path)
    mean_pred = oof_edg.groupby('fold_n')['tmp_predictions_all'].mean().iloc[0]
    std_pred = oof_edg.groupby('fold_n')['tmp_predictions_all'].std().iloc[0]

    # Normalize EdgeNext predictions
    for i in range(5):
        edg_predictions[i] = (edg_predictions[i] - mean_pred) / std_pred

    test_df['predictions_edg'] = np.mean(edg_predictions, axis=0)
    logger.info("Finished extracting EdgeNext features")

    total_elapsed = time.time() - start_time
    logger.info(f"Deep learning feature extraction completed in {total_elapsed:.2f} seconds")

    return test_df


def run_inference(csv_file_path, output_path=None):
    """
    Main inference function that processes a CSV file and generates predictions.

    Args:
        csv_file_path (str): Path to input CSV file containing isic_id column
        output_path (str, optional): Path to save predictions. If None, saves to current directory.
    """

    # Setup logging
    logger = setup_logging("inference")

    # Set global LOGGER for compatibility with train_ensemble functions
    global LOGGER
    LOGGER = logger

    logger.info("Starting ISIC 2024 Inference Pipeline")
    logger.info("=" * 50)

    # Load input CSV
    logger.info(f"Loading input CSV from {csv_file_path}")
    df_input = pd.read_csv(csv_file_path)

    if 'isic_id' not in df_input.columns:
        raise ValueError("Input CSV must contain 'isic_id' column")

    # Check if input has target column for validation/evaluation
    has_target_column = 'target' in df_input.columns
    if has_target_column:
        logger.info("Input CSV contains 'target' column - will include in output for evaluation")
        original_targets = df_input[['isic_id', 'target']].copy()
    else:
        logger.info("Input CSV does not contain 'target' column - inference only mode")
        original_targets = None

    logger.info(f"Loaded {len(df_input)} samples for inference")

    # Convert to polars for feature engineering (matching training pipeline)
    df_pl = pl.from_pandas(df_input)

    # Add dummy columns that feature engineering expects
    required_cols = ['patient_id', 'target'] + InferenceConfig.num_cols + InferenceConfig.cat_cols
    for col in required_cols:
        if col not in df_pl.columns:
            if col == 'target':
                df_pl = df_pl.with_columns(pl.lit(0).alias(col))
            elif col == 'patient_id':
                # Create dummy patient IDs based on isic_id
                df_pl = df_pl.with_columns(
                    pl.col('isic_id').str.slice(0, 10).alias('patient_id')
                )
            elif col in InferenceConfig.num_cols:
                df_pl = df_pl.with_columns(pl.lit(0.0).alias(col))
            elif col in InferenceConfig.cat_cols:
                df_pl = df_pl.with_columns(pl.lit("unknown").alias(col))

    # Engineer features
    logger.info("Engineering features...")
    df_processed = engineer_features(df_pl)

    # Load saved models and metadata
    logger.info(f"Loading trained models from {InferenceConfig.model_save_dir}")
    gb_pipeline, metadata, encoder = GradientBoostingPipeline.load_models(InferenceConfig.model_save_dir)

    feature_cols = metadata['feature_cols']
    columns_to_drop = metadata['columns_to_drop']

    # One-hot encode categorical features using saved encoder
    logger.info("Applying one-hot encoding...")
    new_cat_cols = [f'onehot_{i}' for i in range(len(encoder.get_feature_names_out()))]
    df_processed[new_cat_cols] = encoder.transform(df_processed[InferenceConfig.cat_cols])

    # Extract deep learning features from JPG images
    logger.info("Extracting deep learning features...")
    df_processed = extract_dl_features_jpg(
        df_processed.reset_index(),
        InferenceConfig.image_dir
    )

    # Add patient normalizations for DL features
    logger.info("Adding patient normalizations...")
    df_processed = add_patient_norm(df_processed, 'old_set_0', 'old_set_0_m')
    df_processed = add_patient_norm(df_processed, 'old_set_1', 'old_set_1_m')
    df_processed = add_patient_norm(df_processed, 'old_set_2', 'old_set_2_m')
    df_processed = add_patient_norm(df_processed, 'predictions_eva', 'predictions_eva_m')
    df_processed = add_patient_norm(df_processed, 'predictions_edg', 'predictions_edg_m')

    # Clean data and add LOF features
    logger.info("Cleaning data and adding LOF features...")
    df_processed = clean_data(df_processed, feature_cols, "inference")
    df_processed = add_lof_features(df_processed, InferenceConfig.lof_features)

    # Generate predictions
    logger.info("Generating predictions...")
    predictions = gb_pipeline.predict(df_processed, feature_cols, columns_to_drop)

    # Create output dataframe
    df_output = pd.DataFrame({
        'isic_id': df_processed['isic_id'],
        'prediction': predictions
    })

    # Add target column if it was present in input CSV
    if has_target_column and original_targets is not None:
        df_output = df_output.merge(original_targets, on='isic_id', how='left')
        logger.info("Added original 'target' column to output for evaluation")

        # Calculate metrics if target column is present
        from sklearn.metrics import roc_auc_score
        try:
            auc_score = roc_auc_score(df_output['target'], df_output['prediction'])
            logger.info(f"AUC Score: {auc_score:.4f}")
        except Exception as e:
            logger.warning(f"Could not calculate AUC score: {e}")

    # Save results
    if output_path is None:
        output_path = f"inference_predictions_{Config.run_tags}.csv"

    df_output.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")
    logger.info(f"Generated predictions for {len(df_output)} samples")
    logger.info(f"Prediction statistics: mean={predictions.mean():.4f}, std={predictions.std():.4f}")

    if has_target_column:
        logger.info("Output includes both 'prediction' and 'target' columns")
    else:
        logger.info("Output includes 'prediction' column only")

    return df_output


if __name__ == "__main__":
    import argparse
    default_input_csv = '/home/ubuntu/skinmap/isic-feature-generation/feature_generation/filtered_test_calculated_umaneo_metrics.csv'

    parser = argparse.ArgumentParser(description="ISIC 2024 Inference Pipeline")
    parser.add_argument("--csv_file", help="Path to input CSV file with isic_id column", default=default_input_csv)
    parser.add_argument("--output", "-o", help="Output CSV file path", default=None)

    args = parser.parse_args()

    try:
        run_inference(args.csv_file, args.output)
    except Exception as e:
        print(f"Error during inference: {e}")
        sys.exit(1)
