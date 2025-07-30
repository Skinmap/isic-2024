#!/usr/bin/env python3
"""
Submission Analysis Script for ISIC 2024 Challenge
Analyzes submission.csv file and calculates pAUC (partial AUC) score
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import argparse
from pathlib import Path


def calculate_pauc(y_true, y_pred, min_tpr=0.80):
    """
    Calculate partial AUC at specified TPR threshold

    This is the competition metric: partial AUC at 80% TPR
    The formula calculates the area under the ROC curve for the region
    where True Positive Rate >= min_tpr (80%)

    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted probabilities 
        min_tpr: Minimum True Positive Rate threshold (default 0.80)

    Returns:
        partial_auc: Partial AUC score
    """
    max_fpr = abs(1 - min_tpr)

    # Flip labels and predictions for partial AUC calculation
    v_gt = abs(y_true - 1)
    v_pred = np.array([1.0 - x for x in y_pred])

    # Calculate scaled partial AUC
    partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)

    # Convert to actual partial AUC
    partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)

    return partial_auc


def analyze_submission(submission_path, verbose=True):
    """
    Analyze submission file and calculate metrics

    Args:
        submission_path: Path to submission.csv file
        verbose: Whether to print detailed analysis

    Returns:
        dict: Analysis results
    """
    # Load submission file
    if verbose:
        print(f"Loading submission file: {submission_path}")

    df = pd.read_csv(submission_path, index_col='isic_id')

    if verbose:
        print(f"Submission shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

    # Basic statistics
    results = {
        'num_samples': len(df),
        'num_positive': df['target'].sum(),
        'num_negative': len(df) - df['target'].sum(),
        'positive_rate': df['target'].mean(),
    }

    # Prediction statistics
    results.update({
        'pred_mean': df['prediction'].mean(),
        'pred_std': df['prediction'].std(),
        'pred_min': df['prediction'].min(),
        'pred_max': df['prediction'].max(),
        'pred_median': df['prediction'].median(),
    })

    # Calculate pAUC score
    pauc_score = calculate_pauc(df['target'], df['prediction'])
    results['pauc_score'] = pauc_score

    # Calculate regular AUC for comparison
    regular_auc = roc_auc_score(df['target'], df['prediction'])
    results['regular_auc'] = regular_auc

    if verbose:
        print("\n" + "=" * 50)
        print("SUBMISSION ANALYSIS RESULTS")
        print("=" * 50)

        print(f"\nDataset Statistics:")
        print(f"  Total samples: {results['num_samples']:,}")
        print(f"  Positive samples: {results['num_positive']:,}")
        print(f"  Negative samples: {results['num_negative']:,}")
        print(f"  Positive rate: {results['positive_rate']:.4f}")

        print("\nPrediction Statistics:")
        print(f"  Mean: {results['pred_mean']:.6f}")
        print(f"  Std:  {results['pred_std']:.6f}")
        print(f"  Min:  {results['pred_min']:.6f}")
        print(f"  Max:  {results['pred_max']:.6f}")
        print(f"  Median: {results['pred_median']:.6f}")

        print("\nMetric Scores:")
        print(f"  pAUC (80% TPR): {results['pauc_score']:.6f}")
        print(f"  Regular AUC:    {results['regular_auc']:.6f}")

        print("\nTop 10 highest predictions:")
        top_preds = df.nlargest(10, 'prediction')[['target', 'prediction']]
        for idx, row in top_preds.iterrows():
            print(f"  {idx}: target={row['target']}, pred={row['prediction']:.6f}")

        print("\nTop 10 lowest predictions:")
        low_preds = df.nsmallest(10, 'prediction')[['target', 'prediction']]
        for idx, row in low_preds.iterrows():
            print(f"  {idx}: target={row['target']}, pred={row['prediction']:.6f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Analyze ISIC 2024 submission file')
    parser.add_argument('--submission', '-s', type=str, default='src/submission.csv',
                        help='Path to submission.csv file (default: src/submission.csv)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress verbose output')

    args = parser.parse_args()

    submission_path = Path(args.submission)

    if not submission_path.exists():
        print(f"Error: Submission file not found: {submission_path}")
        return

    # Analyze submission
    results = analyze_submission(submission_path, verbose=not args.quiet)

    # Always print the key metric
    if args.quiet:
        print(f"pAUC Score: {results['pauc_score']:.6f}")


if __name__ == "__main__":
    main()
