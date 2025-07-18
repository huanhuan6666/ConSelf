import os
import json
import argparse
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from tqdm import tqdm
import re
from sklearn.preprocessing import MinMaxScaler

BASE_METRICS = [
    "semantic_entropy",             
    "avg_all_samples_nll",          
    "avg_all_samples_top_k_entropy",
    "crash_rate"                    
]

TARGET_METRIC = "pass@1"

def load_single_result(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if TARGET_METRIC not in data:
            return None

        metrics_data = {TARGET_METRIC: data[TARGET_METRIC]}
        for metric in BASE_METRICS:
            metrics_data[metric] = data.get(metric, np.nan)

        metrics_data['question_id'] = data.get('question_id', os.path.basename(filepath))
        return metrics_data

    except json.JSONDecodeError:
        print(f"Error: Failed to parse JSON file {filepath}. Skipping.")
        return None
    except Exception as e:
        print(f"Error: Error processing file {filepath}: {e}. Skipping.")
        return None

def load_results_directory(dir_path):
    all_results = []
    if not os.path.isdir(dir_path):
        print(f"Error: Directory not found: {dir_path}")
        return None, []

    print(f"Loading results from directory: {dir_path}")
    try:
        file_list = [entry for entry in os.scandir(dir_path) if entry.is_file() and entry.name.endswith('.json')]
    except FileNotFoundError:
        print(f"Error: Failed to access directory {dir_path}")
        return None, []

    if not file_list:
        print("Error: No JSON files found in directory.")
        return None, []

    for entry in tqdm(file_list, desc="Loading JSON files"):
        result_data = load_single_result(entry.path)
        if result_data:
            all_results.append(result_data)

    if not all_results:
        print("\nError: No valid results found.")
        return None, []

    df = pd.DataFrame(all_results)

    loaded_base_metrics = []
    columns_to_convert = [TARGET_METRIC] + BASE_METRICS
    for col in columns_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if col in BASE_METRICS:
                loaded_base_metrics.append(col)

    columns_to_check_for_nan = [TARGET_METRIC] + loaded_base_metrics
    initial_rows = len(df)
    df.dropna(subset=columns_to_check_for_nan, inplace=True)
    rows_after_drop = len(df)
    if rows_after_drop < initial_rows:
        print(f"\nDropped {initial_rows - rows_after_drop} rows due to missing/invalid values in key base metrics.")

    if df.empty:
        print("Error: DataFrame is empty after processing base metrics.")
        return None, []
    return df, loaded_base_metrics

def plot_roc_pr_curves(df, metric, output_dir):
    """Plots ROC and Precision-Recall curves for a given metric."""
    if metric not in df.columns or df[metric].isnull().all():
        return None

    binary_target = (df[TARGET_METRIC] > 0).astype(int)

    higher_is_better = False
    if higher_is_better:
        confidence_score = df[metric]
    else:
        confidence_score = -df[metric]
    valid_idx = confidence_score.notna() & binary_target.notna()
    if not valid_idx.any():
        return None
    confidence_score = confidence_score[valid_idx]
    binary_target = binary_target[valid_idx]

    if len(binary_target.unique()) < 2:
        return None
    try:
        fpr, tpr, _ = roc_curve(binary_target, confidence_score)
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(binary_target, confidence_score)
    except ValueError as e:
        print(f"Warning: Error calculating ROC/PR for metric '{metric}': {e}. Skipping plot.")
        return None
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {metric}'); plt.legend(loc="lower right")
    plt.grid(True, alpha=0.6)
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.ylim([0.0, 1.05]); plt.xlim([0.0, 1.0])
    plt.title(f'Precision-Recall Curve for {metric}')
    plt.grid(True, alpha=0.6)

    plt.tight_layout()
    safe_metric_name = re.sub(r'[^\w\-]+', '_', metric)
    plot_filename = f'roc_pr_curve_{safe_metric_name}.png'
    try:
        plt.savefig(os.path.join(output_dir, plot_filename))
    except Exception as e:
        print(f"Error: Failed to save plot '{plot_filename}': {e}")
    plt.close()

    return roc_auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default='curriculum_selection/data/processed/taco/CodeLlama-7b-Instruct-hf/evaluate_results')
    parser.add_argument("--output_dir", default="analysis_output")
    args = parser.parse_args()

    results_df, loaded_base_metrics_list = load_results_directory(args.results_dir)
    scaler = MinMaxScaler()
    metrics_to_normalize = [m for m in loaded_base_metrics_list if m in results_df.columns]
    if metrics_to_normalize:
        results_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        nan_counts_before_norm = results_df[metrics_to_normalize].isnull().sum().sum()
        if nan_counts_before_norm > 0:
            results_df.dropna(subset=metrics_to_normalize, inplace=True)

        if not results_df.empty:
            results_df[metrics_to_normalize] = scaler.fit_transform(results_df[metrics_to_normalize])
        else:
            results_df = None

    if results_df is not None and not results_df.empty:
        all_metrics_analyzed = loaded_base_metrics_list
        correlations = {}
        metrics_to_correlate = [m for m in all_metrics_analyzed if m in results_df.columns]
        print(f"{'Metirc':<35} | {'Pearson':<15} | {'Pearson p':<15} | {'Spearman':<15} | {'Spearman p':<15} | {'Kendall':<15} | {'Kendall p':<15}")
        print("-" * 100)
        for metric in metrics_to_correlate:
            if TARGET_METRIC not in results_df.columns or results_df[TARGET_METRIC].isnull().all():
                continue
            if results_df[metric].isnull().all():
                continue
            temp_df = results_df[[metric, TARGET_METRIC]].dropna()
            if len(temp_df) < 2 or temp_df[metric].nunique() <= 1 or temp_df[TARGET_METRIC].nunique() <= 1:
                pearson_corr, pearson_pval = np.nan, np.nan
                spearman_corr, spearman_pval = np.nan, np.nan
                kendall_corr, kendall_pval = np.nan, np.nan
            else:
                try:
                    pearson_corr, pearson_pval = pearsonr(temp_df[metric], temp_df[TARGET_METRIC])
                except ValueError:
                    pearson_corr, pearson_pval = np.nan, np.nan
                try:
                    spearman_corr, spearman_pval = spearmanr(temp_df[metric], temp_df[TARGET_METRIC])
                except ValueError:
                    spearman_corr, spearman_pval = np.nan, np.nan
                try:
                    kendall_corr, kendall_pval = kendalltau(temp_df[metric], temp_df[TARGET_METRIC])
                except ValueError:
                    kendall_corr, kendall_pval = np.nan, np.nan

            correlations[metric] = {
                'pearson_corr': pearson_corr, 'pearson_pval': pearson_pval,
                'spearman_corr': spearman_corr, 'spearman_pval': spearman_pval,
                'kendall_corr': kendall_corr, 'kendall_pval': kendall_pval
            }
            pearson_corr = correlations.get(metric, {}).get('pearson_corr', np.nan)
            pearson_pval = correlations.get(metric, {}).get('pearson_pval', np.nan)
            spearman_corr = correlations.get(metric, {}).get('spearman_corr', np.nan)
            spearman_pval = correlations.get(metric, {}).get('spearman_pval', np.nan)
            kendall_corr = correlations.get(metric, {}).get('kendall_corr', np.nan)
            kendall_pval = correlations.get(metric, {}).get('kendall_pval', np.nan)
            print(f"{metric:<35} | {pearson_corr:<15.4f} | {pearson_pval:<15.4e} | {spearman_corr:<15.4f} | {spearman_pval:<15.4e} | {kendall_corr:<15.4f} | {kendall_pval:<15.4e}")
        print("-" * 100)


        auc_scores = {}
        print(f"{'Metric':<35} | {'ROC AUC':<10}")
        print("-" * 50)
        for metric in metrics_to_correlate:
            roc_auc = plot_roc_pr_curves(results_df, metric, args.output_dir)
            if roc_auc is not None:
                auc_scores[metric] = roc_auc

        if auc_scores:
            sorted_auc = sorted(auc_scores.items(), key=lambda item: item[1], reverse=True)
            best_metric_info = ("N/A", float('-inf'))
            if sorted_auc:
                best_metric_info = sorted_auc[0]
            for metric, score in sorted_auc:
                print(f"{metric:<35} | {score:.4f}")
