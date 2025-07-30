#!/usr/bin/env python3
"""
ISIC 2024 Challenge - First Place Solution
Combines deep learning feature extraction with gradient boosting ensemble
"""

from pydantic import BaseModel
import os
import gc
import h5py
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import logging
import warnings
import sys
import time

# Sklearn imports
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline

# Model imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import lightgbm as lgb
import catboost as cb
import xgboost as xgb

# ============================================================================
# LOGGING SETUP
# ============================================================================


def setup_logging():
    """Setup logging configuration"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_ensemble_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)

    # Only capture Python warnings, don't redirect stderr
    # This avoids conflicts with tqdm and other libraries
    logging.captureWarnings(True)
    warnings_logger = logging.getLogger('py.warnings')
    warnings_logger.setLevel(logging.WARNING)

    logger.info(f"Logging initialized. Log file: {log_file}")
    logger.info("Python warnings will be logged")
    return logger


if __name__ == "__main__":
    LOGGER = setup_logging()

# ============================================================================
# CONFIGURATION
# ============================================================================


class Config:
    # Paths
    root = Path('/home/ubuntu/skinmap/isic-feature-generation/feature_generation')
    train_path = root / "filtered_train_calculated_umaneo_metrics.csv"
    test_path = root / 'filtered_test_calculated_umaneo_metrics.csv'
    test_h5 = root / 'filtered_test_calculated_umaneo_metrics.hdf5'
    subm_path = root / 'sample_submission.csv'

    # oof paths
    old_model_preds_path = "/data/models/old-models-predictions/old_data_model_forecast.parquet"
    eva_oof_path = "/data/10ktests/oof_forecasts_eva_base.parquet"
    edg_oof_path = "/data/10ktests/oof_forecasts_edgenext_base.parquet"

    # Model paths
    eva_model_path = "/data/10kmodels/oof_eva_base"
    edg_model_path = "/data/10kmodels/oof_edgenext_base"
    old_model_path = "/data/models/skin-models-base/skin-models-base/ema_small_pretrained"

    # Column definitions
    id_col = 'isic_id'
    target_col = 'target'
    group_col = 'patient_id'

    # Training parameters
    seed = 42
    err = 1e-5
    # TODO: Change back
    # sampling_ratio = 0.01
    sampling_ratio = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Deep learning configs
    eva_config = {
        "img_size": 336,
        "model_name": 'eva02_small_patch14_336.mim_in22k_ft_in1k',
        "valid_batch_size": 64,
    }

    edg_config = {
        "img_size": 256,
        "model_name": 'edgenext_base.in21k_ft_in1k',
        "valid_batch_size": 64,
    }

    # Feature columns
    num_cols = [
        'age_approx', 'clin_size_long_diam_mm', 'tbp_lv_A', 'tbp_lv_Aext',
        'tbp_lv_B', 'tbp_lv_Bext', 'tbp_lv_C', 'tbp_lv_Cext', 'tbp_lv_H',
        'tbp_lv_Hext', 'tbp_lv_L', 'tbp_lv_Lext', 'tbp_lv_areaMM2',
        'tbp_lv_area_perim_ratio', 'tbp_lv_color_std_mean', 'tbp_lv_deltaA',
        'tbp_lv_deltaB', 'tbp_lv_deltaL', 'tbp_lv_deltaLB', 'tbp_lv_deltaLBnorm',
        'tbp_lv_eccentricity', 'tbp_lv_minorAxisMM', 'tbp_lv_nevi_confidence',
        'tbp_lv_norm_border', 'tbp_lv_norm_color', 'tbp_lv_perimeterMM',
        'tbp_lv_radial_color_std_max', 'tbp_lv_stdL', 'tbp_lv_stdLExt',
        'tbp_lv_symm_2axis', 'tbp_lv_symm_2axis_angle', 'tbp_lv_x', 'tbp_lv_y', 'tbp_lv_z',
    ]

    cat_cols = ['sex', 'anatom_site_general', 'tbp_tile_type',
                'tbp_lv_location', 'tbp_lv_location_simple', 'attribution']

    # Columns to drop for final models
    columns_to_drop = [
        'tbp_lv_B', 'tbp_lv_C', 'tbp_lv_H', 'tbp_lv_L',
        'tbp_lv_radial_color_std_max', 'tbp_lv_y', 'tbp_lv_z',
        'luminance_contrast', 'lesion_color_difference', 'normalized_lesion_size',
        'tbp_lv_norm_border_patient_norm', 'lesion_color_difference_patient_norm',
        'age_normalized_nevi_confidence_2_patient_norm', 'tbp_lv_deltaA',
        "lesion_id",
        "iddx_full",
        "iddx_1",
        "iddx_2",
        "iddx_3",
        "iddx_4",
        "iddx_5",
        "mel_mitotic_index",
        "mel_thick_mm",
        "tbp_lv_dnn_lesion_confidence",
    ]

    # LOF features
    lof_features = [
        'tbp_lv_H', 'hue_contrast', 'age_normalized_nevi_confidence_2',
        'tbp_lv_deltaB', 'color_uniformity', 'tbp_lv_z', 'clin_size_long_diam_mm',
        'tbp_lv_y', 'position_distance_3d', 'tbp_lv_stdLExt', 'mean_hue_difference',
        'age_normalized_nevi_confidence', 'lesion_visibility_score',
        'tbp_lv_minorAxisMM', 'tbp_lv_Hext'
    ]


class ModelConfigCB(BaseModel):
    iterations: int = 1000
    learning_rate: float = 0.06936242010150652
    l2_leaf_reg: float = 6.216113851699493
    loss_function: str = "Logloss"
    bagging_temperature: float = 1
    random_seed: int = Config.seed
    border_count: int = 128
    grow_policy: str = "SymmetricTree"  # Depthwise, Lossguide
    min_data_in_leaf: int = 24
    depth: int = 7
    do_sample: bool = True
    random_strength: float = 0
    scale_pos_weight: float = 2.6149345838209532
    subsample: float = 0.6249261779711819
    verbose: bool = False
    task_type: str = 'GPU'
    eval_metric: str = 'AUC'
    od_wait: int = 100
    devices: str = '0'
    bootstrap_type: str = "Bernoulli"

    def to_catboost_params(self, random_seed=None, cat_features=None):
        params = {
            'loss_function': self.loss_function,
            'iterations': self.iterations,
            'learning_rate': self.learning_rate,
            'depth': self.depth,
            'l2_leaf_reg': self.l2_leaf_reg,
            'min_data_in_leaf': self.min_data_in_leaf,
            'scale_pos_weight': self.scale_pos_weight,
            'subsample': self.subsample,
            'verbose': self.verbose,
            'random_seed': random_seed or self.random_seed,
            'bootstrap_type': self.bootstrap_type,
        }

        # Add categorical features if provided
        if cat_features is not None:
            params['cat_features'] = cat_features

        # Only add non-default values
        if self.task_type:
            params['task_type'] = self.task_type
        if self.eval_metric:
            params['eval_metric'] = self.eval_metric
        if self.od_wait != 100:
            params['od_wait'] = self.od_wait
        if self.devices:
            params['devices'] = self.devices
        if self.bagging_temperature != 1:
            params['bagging_temperature'] = self.bagging_temperature
        if self.border_count != 128:
            params['border_count'] = self.border_count
        if self.grow_policy != 'SymmetricTree':
            params['grow_policy'] = self.grow_policy
        if self.random_strength > 0:
            params['random_strength'] = self.random_strength

        return params


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def engineer_features(df):
    """Create all engineered features"""
    LOGGER.info(f"Starting feature engineering for {len(df)} samples")

    start_time = time.time()
    result = (df
              .with_columns(
                  pl.col('age_approx').cast(pl.String).replace('NA', np.nan).cast(pl.Float64),
              )
              .with_columns(
                  pl.col(pl.Float64).fill_nan(pl.col(pl.Float64).median()),
              )
              # Basic engineered features
              .with_columns(
                  lesion_size_ratio=pl.col('tbp_lv_minorAxisMM') / pl.col('clin_size_long_diam_mm'),
                  lesion_shape_index=pl.col('tbp_lv_areaMM2') / (pl.col('tbp_lv_perimeterMM') ** 2),
                  hue_contrast=(pl.col('tbp_lv_H') - pl.col('tbp_lv_Hext')).abs(),
                  luminance_contrast=(pl.col('tbp_lv_L') - pl.col('tbp_lv_Lext')).abs(),
                  lesion_color_difference=(pl.col('tbp_lv_deltaA') ** 2
                                           + pl.col('tbp_lv_deltaB') ** 2
                                           + pl.col('tbp_lv_deltaL') ** 2).sqrt(),
                  border_complexity=pl.col('tbp_lv_norm_border') + pl.col('tbp_lv_symm_2axis'),
                  color_uniformity=pl.col('tbp_lv_color_std_mean') / (pl.col('tbp_lv_radial_color_std_max') + Config.err),
              )
              # More complex features
              .with_columns(
                  position_distance_3d=(pl.col('tbp_lv_x') ** 2
                                        + pl.col('tbp_lv_y') ** 2
                                        + pl.col('tbp_lv_z') ** 2).sqrt(),
                  perimeter_to_area_ratio=pl.col('tbp_lv_perimeterMM') / pl.col('tbp_lv_areaMM2'),
                  area_to_perimeter_ratio=pl.col('tbp_lv_areaMM2') / pl.col('tbp_lv_perimeterMM'),
                  lesion_visibility_score=pl.col('tbp_lv_deltaLBnorm') + pl.col('tbp_lv_norm_color'),
                  symmetry_border_consistency=pl.col('tbp_lv_symm_2axis') * pl.col('tbp_lv_norm_border'),
                  consistency_symmetry_border=pl.col('tbp_lv_symm_2axis') * pl.col('tbp_lv_norm_border')
                  / (pl.col('tbp_lv_symm_2axis') + pl.col('tbp_lv_norm_border')),
              )
              # Additional engineered features (abbreviated for space)
              .with_columns(
                  color_consistency=pl.col('tbp_lv_stdL') / pl.col('tbp_lv_Lext'),
                  size_age_interaction=pl.col('clin_size_long_diam_mm') * pl.col('age_approx'),
                  log_lesion_area=(pl.col('tbp_lv_areaMM2') + 1).log(),
                  normalized_lesion_size=pl.col('clin_size_long_diam_mm') / pl.col('age_approx'),
                  # ... more features as in original
              )
              # Patient-level normalizations
              .with_columns(
                  ((pl.col(col) - pl.col(col).mean().over('patient_id'))
                   / (pl.col(col).std().over('patient_id') + Config.err)).alias(f'{col}_patient_norm')
                  for col in Config.num_cols
              )
              # Patient-level aggregations
              .with_columns(
                  count_per_patient=pl.col('isic_id').count().over('patient_id'),
                  tbp_lv_areaMM2_patient=pl.col('tbp_lv_areaMM2').sum().over('patient_id'),
                  tbp_lv_areaMM2_bp=pl.col('tbp_lv_areaMM2').sum().over(['patient_id', 'anatom_site_general']),
              )
              .with_columns(
                  age_normalized_nevi_confidence=pl.col('tbp_lv_nevi_confidence') / pl.col('age_approx'),
                  age_normalized_nevi_confidence_2=(pl.col('clin_size_long_diam_mm')**2 + pl.col('age_approx')**2).sqrt(),
                  mean_hue_difference=(pl.col('tbp_lv_H') + pl.col('tbp_lv_Hext')) / 2,
              )
              .with_columns(
                  pl.col(Config.cat_cols).cast(pl.Categorical),
              )
              .to_pandas()
              .set_index(Config.id_col)
              )

    elapsed_time = time.time() - start_time
    LOGGER.info(f"Feature engineering completed in {elapsed_time:.2f} seconds")
    LOGGER.info(f"Final feature matrix shape: {result.shape}")

    return result


def load_training_dl_features(df_train):
    """Load pre-computed deep learning features for training data"""
    LOGGER.info("Loading pre-computed training deep learning features...")

    # Reset index to make isic_id a column for merging
    df_train = df_train.reset_index()

    # 1. Load old 3-class model predictions
    LOGGER.info("Loading old 3-class model predictions...")
    old_data_model_preds = pd.read_parquet(Config.old_model_preds_path)
    df_train = df_train.merge(old_data_model_preds, how="left", on="isic_id")

    # Add patient normalizations for old model predictions
    df_train = add_patient_norm(df_train, 'old_set_0', 'old_set_0_m')
    df_train = add_patient_norm(df_train, 'old_set_1', 'old_set_1_m')
    df_train = add_patient_norm(df_train, 'old_set_2', 'old_set_2_m')

    # 2. Load EVA model OOF predictions
    LOGGER.info("Loading EVA model OOF predictions...")
    oof_forecasts_eva = pd.read_parquet(Config.eva_oof_path)

    # add tmp_predictions_all__pr (z-score) if the parquet doesn't have it
    if 'tmp_predictions_all__pr' not in oof_forecasts_eva.columns:
        print("Calculating 'tmp_predictions_all__pr' (z-score)...")

        # Calculate the mean of predictions for each fold
        # Using transform() ensures the output is a Series with the same index as the original DataFrame
        mean_preds = oof_forecasts_eva.groupby('fold_n')['tmp_predictions_all'].transform('mean')

        # Calculate the standard deviation of predictions for each fold
        std_preds = oof_forecasts_eva.groupby('fold_n')['tmp_predictions_all'].transform('std')

        # Calculate the z-score and create the new column
        # z = (x - mean) / std
        oof_forecasts_eva['tmp_predictions_all__pr'] = (oof_forecasts_eva['tmp_predictions_all'] - mean_preds) / std_preds

        # Handle cases where std_preds is 0 to avoid division by zero, filling with 0
        oof_forecasts_eva['tmp_predictions_all__pr'] = oof_forecasts_eva['tmp_predictions_all__pr'].fillna(0)

    # Keep the original format and do patient norm BEFORE merging
    oof_forecasts_eva_clean = oof_forecasts_eva[['isic_id', 'patient_id', 'tmp_predictions_all__pr']].rename(columns={
        'tmp_predictions_all__pr': 'predictions_eva'
    })

    # Do patient normalization on the OOF data using ITS patient_id
    oof_forecasts_eva_clean = add_patient_norm(oof_forecasts_eva_clean, 'predictions_eva', 'predictions_eva_m')

    # Now merge only the prediction columns (both raw and patient-normalized)
    df_train = df_train.merge(
        oof_forecasts_eva_clean[['isic_id', 'predictions_eva', 'predictions_eva_m']],
        how="left", on='isic_id'
    )

    # 3. Load EdgeNext model OOF predictions
    LOGGER.info("Loading EdgeNext model OOF predictions...")
    oof_forecasts_edgenext = pd.read_parquet(Config.edg_oof_path)

    # add tmp_predictions_all__pr (z-score) if the parquet doesn't have it
    if 'tmp_predictions_all__pr' not in oof_forecasts_edgenext.columns:
        print("Calculating 'tmp_predictions_all__pr' (z-score)...")

        # Calculate the mean of predictions for each fold
        # Using transform() ensures the output is a Series with the same index as the original DataFrame
        mean_preds = oof_forecasts_edgenext.groupby('fold_n')['tmp_predictions_all'].transform('mean')

        # Calculate the standard deviation of predictions for each fold
        std_preds = oof_forecasts_edgenext.groupby('fold_n')['tmp_predictions_all'].transform('std')

        # Calculate the z-score and create the new column
        # z = (x - mean) / std
        oof_forecasts_edgenext['tmp_predictions_all__pr'] = (oof_forecasts_edgenext['tmp_predictions_all'] - mean_preds) / std_preds

        # Handle cases where std_preds is 0 to avoid division by zero, filling with 0
        oof_forecasts_edgenext['tmp_predictions_all__pr'] = oof_forecasts_edgenext['tmp_predictions_all__pr'].fillna(0)

    # Same approach for EdgeNext
    oof_forecasts_edgenext_clean = oof_forecasts_edgenext[['isic_id', 'patient_id', 'tmp_predictions_all__pr']].rename(columns={
        'tmp_predictions_all__pr': 'predictions_edg'
    })

    # Do patient normalization on the OOF data using ITS patient_id
    oof_forecasts_edgenext_clean = add_patient_norm(oof_forecasts_edgenext_clean, 'predictions_edg', 'predictions_edg_m')

    # Merge only the prediction columns
    df_train = df_train.merge(
        oof_forecasts_edgenext_clean[['isic_id', 'predictions_edg', 'predictions_edg_m']],
        how="left", on='isic_id'
    )

    # Set index back to isic_id
    df_train = df_train.set_index('isic_id')

    LOGGER.info("Finished loading training deep learning features")
    return df_train


def add_patient_norm(df, column_name, column_name_new):
    """Add patient-normalized version of a column"""
    df = df.merge(
        df.groupby("patient_id").agg(**{
            column_name_new: pd.NamedAgg(column_name, 'mean')
        }).reset_index(), how="left", on=["patient_id"])

    df[column_name_new] = df[column_name] / df[column_name_new]
    return df

# ============================================================================
# DEEP LEARNING COMPONENTS
# ============================================================================


class ISICDataset(Dataset):
    def __init__(self, df, file_hdf, transforms=None):
        self.df = df
        self.fp_hdf = h5py.File(file_hdf, mode="r")
        self.isic_ids = df['isic_id'].values
        self.targets = df['target'].values
        self.transforms = transforms

    def __len__(self):
        return len(self.isic_ids)

    def __getitem__(self, index):
        isic_id = self.isic_ids[index]
        img = np.array(Image.open(BytesIO(self.fp_hdf[isic_id][()])))
        target = self.targets[index]

        if self.transforms:
            img = self.transforms(image=img)["image"]

        return {
            'image': img,
            'target': target,
        }


class ISICModel(nn.Module):
    def __init__(self, model_name, num_classes=1, pretrained=True):
        super(ISICModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, num_classes)
        self.sigmoid = nn.Sigmoid() if num_classes == 1 else nn.Softmax()

    def forward(self, images):
        return self.sigmoid(self.model(images))


class ISICModelEdgenext(nn.Module):
    def __init__(self, model_name, num_classes=1, pretrained=True):
        super(ISICModelEdgenext, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained,
                                       num_classes=num_classes, global_pool='avg')
        self.sigmoid = nn.Sigmoid()

    def forward(self, images):
        return self.sigmoid(self.model(images))


@torch.inference_mode()
def generate_predictions(model, dataloader, device):
    """Generate predictions from a deep learning model"""
    logger = logging.getLogger(__name__)
    model.eval()

    predictions_all = []
    targets_all = []

    logger.info(f"Generating predictions for {len(dataloader)} batches")
    start_time = time.time()

    for step, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        images = data['image'].to(device, dtype=torch.float)
        targets = data['target'].to(device, dtype=torch.float)

        outputs = model(images).squeeze()

        predictions_all.append(outputs.cpu().numpy())
        targets_all.append(targets.cpu().numpy())

    gc.collect()

    targets_all = np.concatenate(targets_all)
    predictions_all = np.concatenate(predictions_all)

    elapsed_time = time.time() - start_time
    logger.info(f"Prediction generation completed in {elapsed_time:.2f} seconds")
    logger.info(f"Generated {len(predictions_all)} predictions")

    return targets_all, predictions_all


def extract_dl_features(test_df, test_h5):
    """Extract features from all deep learning models"""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting deep learning feature extraction for {len(test_df)} samples")

    start_time = time.time()

    # Define transforms
    transform_eva = A.Compose([
        A.Resize(Config.eva_config['img_size'], Config.eva_config['img_size']),
        A.Normalize(mean=[0.4815, 0.4578, 0.4082],
                    std=[0.2686, 0.2613, 0.2758],
                    max_pixel_value=255.0, p=1.0),
        ToTensorV2(),
    ], p=1.)

    transform_edg = A.Compose([
        A.Resize(Config.edg_config['img_size'], Config.edg_config['img_size']),
        A.Normalize(mean=[0.4815, 0.4578, 0.4082],
                    std=[0.2686, 0.2613, 0.2758],
                    max_pixel_value=255.0, p=1.0),
        ToTensorV2(),
    ], p=1.)

    test_df['target'] = 0

    # 1. Old 3-class model predictions
    logger.info("Extracting features from old 3-class model...")
    dataset = ISICDataset(test_df, test_h5, transforms=transform_eva)
    dataloader = DataLoader(dataset, batch_size=Config.eva_config['valid_batch_size'],
                            num_workers=2, shuffle=False, pin_memory=True)

    model = ISICModel(Config.eva_config['model_name'], pretrained=False, num_classes=3)
    model.load_state_dict(torch.load(Config.old_model_path, weights_only=True))
    model.to(Config.device)

    _, predictions = generate_predictions(model, dataloader, Config.device)
    test_df['old_set_0'] = predictions[:, 0]
    test_df['old_set_1'] = predictions[:, 1]
    test_df['old_set_2'] = predictions[:, 2]
    model.to('cpu')
    logger.info(f"Finished extracting old 3-class model features")

    # 2. EVA model predictions
    eva_start_time = time.time()
    logger.info("Extracting features from EVA models...")
    eva_predictions = []
    for i in range(5):
        model_path = os.path.join(Config.eva_model_path, f"model__{i}")
        model = ISICModel(Config.eva_config['model_name'], pretrained=False, num_classes=1)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.to(Config.device)

        _, predictions = generate_predictions(model, dataloader, Config.device)
        eva_predictions.append(predictions)
        model.to('cpu')

    eva_elapsed = time.time() - eva_start_time
    logger.info(f"Finished extracting EVA features in {eva_elapsed:.2f} seconds")

    logger.info("Normalizing EVA features...")
    # Normalize EVA predictions
    oof_eva = pd.read_parquet(Config.eva_oof_path)
    mean_pred = oof_eva.groupby('fold_n')['tmp_predictions_all'].mean().iloc[0]
    std_pred = oof_eva.groupby('fold_n')['tmp_predictions_all'].std().iloc[0]

    for i in range(5):
        eva_predictions[i] = (eva_predictions[i] - mean_pred) / std_pred

    test_df['predictions_eva'] = np.mean(eva_predictions, axis=0)

    logger.info("Finished normalizing EVA features")

    # 3. EdgeNext model predictions
    logger.info("Extracting features from EdgeNext models...")
    dataset = ISICDataset(test_df, test_h5, transforms=transform_edg)
    dataloader = DataLoader(dataset, batch_size=Config.edg_config['valid_batch_size'],
                            num_workers=2, shuffle=False, pin_memory=True)

    edg_start_time = time.time()
    edg_predictions = []
    for i in range(5):
        model_path = os.path.join(Config.edg_model_path, f"model__{i}")
        model = ISICModelEdgenext('edgenext_base.in21k_ft_in1k', pretrained=False)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.to(Config.device)

        _, predictions = generate_predictions(model, dataloader, Config.device)
        edg_predictions.append(predictions)
        model.to('cpu')

    edg_elapsed = time.time() - edg_start_time
    logger.info(f"Finished extracting EdgeNext features in {edg_elapsed:.2f} seconds")

    logger.info("Normalizing EdgeNext predictions")
    # Normalize EdgeNext predictions
    oof_edg = pd.read_parquet(Config.edg_oof_path)
    mean_pred = oof_edg.groupby('fold_n')['tmp_predictions_all'].mean().iloc[0]
    std_pred = oof_edg.groupby('fold_n')['tmp_predictions_all'].std().iloc[0]

    for i in range(5):
        edg_predictions[i] = (edg_predictions[i] - mean_pred) / std_pred

    test_df['predictions_edg'] = np.mean(edg_predictions, axis=0)

    logger.info("Finished normalizing EdgeNext predictions")

    total_elapsed = time.time() - start_time
    logger.info(f"Deep learning feature extraction completed in {total_elapsed:.2f} seconds")

    return test_df


class OOFNormalizer:
    def __init__(self, oof_path):
        self.oof_df = pd.read_parquet(oof_path)
        self.stats = self._calculate_stats()

    def _calculate_stats(self):
        stats = self.oof_df.groupby('fold_n').agg({
            'tmp_predictions_all': ['mean', 'std'],
            'temp_predictions_all__pr': ['mean', 'std']
        })
        return stats

    def normalize_predictions(self, predictions, fold=0):
        mean = self.stats.loc[fold, ('tmp_predictions_all', 'mean')]
        std = self.stats.loc[fold, ('tmp_predictions_all', 'std')]
        return (predictions - mean) / std


# ============================================================================
# ADDITIONAL FEATURE ENGINEERING
# ============================================================================


def add_lof_features(df, features):
    """Add Local Outlier Factor scores"""
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df[features])
    outlier_factors = []

    for patient_id in tqdm(df.patient_id.unique()):
        mask = (df['patient_id'] == patient_id).values
        sum_mask = sum(mask)
        if sum_mask < 3:
            continue

        patient_emb = scaled_array[mask]
        clf = LocalOutlierFactor(n_neighbors=min(30, sum_mask))
        clf.fit_predict(patient_emb)

        outlier_factors.append(pd.DataFrame({
            "isic_id": df[mask].index.values,
            "of": clf.negative_outlier_factor_
        }))

    if len(outlier_factors) == 0:
        df['of'] = -1
        return df

    outlier_factors = pd.concat(outlier_factors).reset_index(drop=True)
    df = df.merge(outlier_factors.set_index('isic_id'),
                  how="left", left_index=True, right_index=True)
    df['of'] = df['of'].fillna(-1)

    return df


# ============================================================================
# GRADIENT BOOSTING MODELS
# ============================================================================

def custom_metric(y_hat, y_true):
    """Competition metric: partial AUC at 80% TPR"""
    min_tpr = 0.80
    max_fpr = abs(1 - min_tpr)

    v_gt = abs(y_true - 1)
    v_pred = np.array([1.0 - x for x in y_hat])

    partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)

    return partial_auc


class GradientBoostingPipeline:
    def __init__(self, model_params):
        self.lgb_params = model_params['lgb']
        self.cb_params = model_params['cb']
        self.xgb_params = model_params['xgb']
        self.models = {'lgb': [], 'cb': [], 'xgb': []}

    def train_models(self, df_train, feature_cols, cat_cols, columns_to_drop):
        """Train all gradient boosting models with cross-validation"""

        # LightGBM models
        LOGGER.info("Training LightGBM models...")
        for random_seed in range(1, 10):
            seed = random_seed * 10 + 17
            cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)

            for fold, (train_idx, val_idx) in enumerate(cv.split(df_train,
                                                                 y=df_train.target,
                                                                 groups=df_train['patient_id'])):
                X_train = df_train.iloc[train_idx][[c for c in feature_cols if c not in columns_to_drop]]
                y_train = df_train.iloc[train_idx]['target']

                for col in ['predictions_edg', 'predictions_edg_m', 'predictions_eva', 'predictions_eva_m']:
                    if col in X_train.columns:
                        noise = np.random.normal(0, 0.1, len(X_train)).astype(X_train[col].dtype)
                        X_train.loc[:, col] = X_train[col] + noise

                # LightGBM with one-hot encoded features
                model = Pipeline([
                    # TODO: Change back
                    # ('sampler_1', RandomOverSampler(sampling_strategy=0.003, random_state=seed)),
                    ('sampler_1', RandomOverSampler(sampling_strategy=0.05, random_state=seed)),
                    ('sampler_2', RandomUnderSampler(sampling_strategy=Config.sampling_ratio, random_state=seed)),
                    ('classifier', lgb.LGBMClassifier(**self.lgb_params)),
                ])

                # callbacks = [lgb.log_evaluation(period=25)]
                # model.fit(X_train, y_train, classifier__callbacks=callbacks)

                model.fit(X_train, y_train)
                self.models['lgb'].append(model)

        LOGGER.info("Training CatBoost models...")
        for random_seed in range(1, 10):
            cv = StratifiedGroupKFold(5, shuffle=True, random_state=random_seed)

            for fold, (train_idx, val_idx) in enumerate(cv.split(df_train,
                                                                 y=df_train.target,
                                                                 groups=df_train['patient_id'])):
                X_train = df_train.iloc[train_idx][[c for c in feature_cols if c not in columns_to_drop]]
                y_train = df_train.iloc[train_idx]['target']

                # Add noise to DL predictions
                for col in ['predictions_edg', 'predictions_edg_m', 'predictions_eva', 'predictions_eva_m']:
                    if col in X_train.columns:
                        noise = np.random.normal(0, 0.1, len(X_train)).astype(X_train[col].dtype)
                        X_train[col] = X_train[col] + noise

                cb_params = self.cb_params['base_params'].to_catboost_params(cat_features=Config.cat_cols)
                cb_params['random_seed'] = random_seed

                cb_classifier = cb.CatBoostClassifier(**cb_params)

                model = Pipeline([
                    # TODO: Change back
                    # ('sampler_1', RandomOverSampler(sampling_strategy=0.003, random_state=random_seed)),
                    ('sampler_1', RandomOverSampler(sampling_strategy=0.05, random_state=random_seed)),
                    ('sampler_2', RandomUnderSampler(sampling_strategy=Config.sampling_ratio, random_state=random_seed)),
                    ('classifier', cb_classifier),
                ])

                model.fit(X_train, y_train)
                self.models['cb'].append(model)

        LOGGER.info("Training XGBoost models...")
        for random_seed in range(1, 10):
            seed = random_seed * 10 + 88
            cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)

            for fold, (train_idx, val_idx) in enumerate(cv.split(df_train,
                                                                 y=df_train.target,
                                                                 groups=df_train['patient_id'])):
                X_train = df_train.iloc[train_idx][[c for c in feature_cols if c not in columns_to_drop]]
                y_train = df_train.iloc[train_idx]['target']

                # Add noise to DL predictions
                for col in ['predictions_edg', 'predictions_edg_m', 'predictions_eva', 'predictions_eva_m']:
                    if col in X_train.columns:
                        X_train[col] = X_train[col] + np.random.normal(0, 0.1, len(X_train))

                model = Pipeline([
                    # TODO: Change back
                    # ('sampler_1', RandomOverSampler(sampling_strategy=0.003, random_state=seed)),
                    ('sampler_1', RandomOverSampler(sampling_strategy=0.05, random_state=seed)),
                    ('sampler_2', RandomUnderSampler(sampling_strategy=Config.sampling_ratio, random_state=seed)),
                    ('classifier', xgb.XGBClassifier(**self.xgb_params)),
                ])

                model.fit(X_train, y_train)
                self.models['xgb'].append(model)

    def predict(self, df_test, feature_cols, columns_to_drop):
        """Generate predictions using all trained models"""
        predictions = {}

        for model_type in ['lgb', 'cb', 'xgb']:
            model_preds = []

            for model in self.models[model_type]:
                X_test = df_test[[c for c in feature_cols if c not in columns_to_drop]]
                preds = model.predict_proba(X_test)[:, 1]
                # Convert to ranks
                preds = pd.Series(preds).rank(pct=True).values
                model_preds.append(preds)

            predictions[model_type] = np.mean(model_preds, axis=0)

        # Average across all model types
        final_predictions = np.mean([predictions['lgb'],
                                     predictions['cb'],
                                     predictions['xgb']], axis=0)

        return final_predictions


def clean_data(df, feature_cols, data_type=""):
    """Clean data by handling inf values and missing values"""
    LOGGER.info(f"Cleaning {data_type} data...")

    # First, handle inf values for all columns
    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    # Get categorical columns (multiple ways to detect them)
    cat_columns = list(df.select_dtypes('category').columns)
    string_columns = list(df.select_dtypes('object').columns)
    all_non_numeric = cat_columns + string_columns

    LOGGER.info(f"Found {len(cat_columns)} categorical columns and {len(string_columns)} string columns")

    # Handle missing values differently for numeric vs non-numeric columns
    for col in feature_cols:
        if col in df.columns:
            if col not in all_non_numeric:
                # Numeric columns - use median
                try:
                    if df[col].isna().mean() != 1:  # If not all values are NaN
                        filler = df[col].median()
                    else:
                        filler = 0
                    df[col] = df[col].fillna(filler)
                except Exception as e:
                    LOGGER.warning(f"Could not compute median for column {col} (dtype: {df[col].dtype}), using 0")
                    df[col] = df[col].fillna(0)
            else:
                # Categorical/string columns - use mode
                vc = df[col].value_counts()
                if vc.shape[0] == 0:
                    filler = 'unknown'
                else:
                    filler = vc.index[0]
                df[col] = df[col].fillna(filler)

    return df

# ============================================================================
# MAIN PIPELINE
# ============================================================================


def main():
    LOGGER.info("ISIC 2024 First Place Solution")
    LOGGER.info("=" * 50)

    # Load and engineer features
    LOGGER.info("\n1. Loading and engineering features...")
    df_train = engineer_features(pl.read_csv(Config.train_path))
    df_test = engineer_features(pl.read_csv(Config.test_path))
    # df_subm = pd.read_csv(Config.subm_path, index_col=Config.id_col)

    # One-hot encode categorical features
    LOGGER.info("\n2. One-hot encoding categorical features...")
    encoder = OneHotEncoder(sparse_output=False, dtype=np.int32, handle_unknown='ignore')
    encoder.fit(df_train[Config.cat_cols])

    new_cat_cols = [f'onehot_{i}' for i in range(len(encoder.get_feature_names_out()))]

    df_train[new_cat_cols] = encoder.transform(df_train[Config.cat_cols])
    df_test[new_cat_cols] = encoder.transform(df_test[Config.cat_cols])

    # Update feature columns
    feature_cols = [
        col for col in df_train.columns
        if df_train[col].dtype != 'object' and col != Config.target_col
    ]

    # Load deep learning features
    LOGGER.info("\n3. Loading deep learning features...")

    # Load pre-computed training features
    df_train = load_training_dl_features(df_train)

    # Extract test features
    test_dl_features = extract_dl_features(df_test.reset_index(), Config.test_h5)
    df_test = df_test.merge(test_dl_features[['isic_id', 'old_set_0', 'old_set_1', 'old_set_2',
                                             'predictions_eva', 'predictions_edg']],
                            on='isic_id', how='left')

    # Add patient normalizations for test DL features
    df_test = add_patient_norm(df_test, 'old_set_0', 'old_set_0_m')
    df_test = add_patient_norm(df_test, 'old_set_1', 'old_set_1_m')
    df_test = add_patient_norm(df_test, 'old_set_2', 'old_set_2_m')
    df_test = add_patient_norm(df_test, 'predictions_eva', 'predictions_eva_m')
    df_test = add_patient_norm(df_test, 'predictions_edg', 'predictions_edg_m')

    # Update feature columns with new features
    dl_features = ['old_set_0', 'old_set_1', 'old_set_2', 'old_set_0_m', 'old_set_1_m', 'old_set_2_m',
                   'predictions_eva', 'predictions_eva_m', 'predictions_edg', 'predictions_edg_m']
    feature_cols.extend(dl_features)

    LOGGER.info("\n4. Cleaning data...")
    df_train = clean_data(df_train, feature_cols, "training")
    df_test = clean_data(df_test, feature_cols, "test")

    # Add LOF features (after cleaning data to remove NaN values)
    LOGGER.info("\n5. Adding Local Outlier Factor features...")
    df_train = add_lof_features(df_train, Config.lof_features)
    df_test = add_lof_features(df_test, Config.lof_features)

    # Add 'of' feature to feature columns
    feature_cols.append('of')

    cb_config = ModelConfigCB(
        iterations=200,
        learning_rate=0.06936242010150652,
        depth=7,
        scale_pos_weight=2.6149345838209532,
        l2_leaf_reg=6.216113851699493,
        subsample=0.6249261779711819,
        min_data_in_leaf=24,
        do_sample=True,
    )

    # Define model parameters
    model_params = {
        'lgb': {
            'objective': 'binary',
            'verbosity': -1,
            'n_iter': 200,
            'boosting_type': 'gbdt',
            'device': 'gpu',
            'random_state': Config.seed,
            'lambda_l1': 0.08758718919397321,
            'lambda_l2': 0.0039689175176025465,
            'learning_rate': 0.03231007103195577,
            'max_depth': 4,
            'num_leaves': 103,
            'colsample_bytree': 0.8329551585827726,
            'colsample_bynode': 0.4025961355653304,
            'bagging_fraction': 0.7738954452473223,
            'bagging_freq': 4,
            'min_data_in_leaf': 85,
            'scale_pos_weight': 2.7984184778875543,
        },
        'cb': {
            'base_params': cb_config,
            'do_sample': cb_config.do_sample,
        },
        'xgb': {
            'enable_categorical': True,
            'tree_method': 'hist',
            'random_state': Config.seed,
            'learning_rate': 0.08501257473292347,
            'lambda': 8.879624125465703,
            'alpha': 0.6779926606782505,
            'max_depth': 6,
            'subsample': 0.6012681388711075,
            'colsample_bytree': 0.8437772277074493,
            'colsample_bylevel': 0.5476090898823716,
            'colsample_bynode': 0.9928601203635129,
            'scale_pos_weight': 3.29440313334688,
            'device': 'cuda',
        }
    }

    # Train gradient boosting models
    LOGGER.info("\n6. Training gradient boosting models...")
    gb_pipeline = GradientBoostingPipeline(model_params)

    # Note: In production, you would load pre-trained models here
    gb_pipeline.train_models(df_train, feature_cols, new_cat_cols, Config.columns_to_drop)

    # Generate predictions
    LOGGER.info("\n7. Generating final predictions...")
    predictions = gb_pipeline.predict(df_test, feature_cols, columns_to_drop=Config.columns_to_drop)

    # Save submission
    df_subm = pd.DataFrame({
        'isic_id': df_test["isic_id"],
        'target': df_test['target'],
        'prediction': predictions
    })
    df_subm.set_index(Config.id_col, inplace=True)
    df_subm.to_csv('submission.csv')

    auc_score = custom_metric(df_subm["prediction"], df_subm["target"])
    LOGGER.info("\nSubmission saved to submission.csv")
    LOGGER.info(f"\nAUROC Score: {auc_score}")
    LOGGER.info(f"Shape: {df_subm.shape}")
    LOGGER.info(df_subm.head())


if __name__ == "__main__":
    main()
