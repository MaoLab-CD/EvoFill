# #!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   eval_metrics.py
@Time    :   2025/08/22 09:42:34
@Author  :   zqyin
@Version :   1.0
@Desc    :   None
'''

import pysam
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def load_gt_matrix(vcf_path):
    vcf = pysam.VariantFile(vcf_path)
    gt_matrix = {}
    for record in vcf:
        key = (record.contig, record.pos)
        gt_matrix[key] = {}
        for sample in record.samples:
            gt = record.samples[sample]['GT']
            if gt is None or None in gt:
                gt_str = "./."
            else:
                sep = "|" if record.samples[sample].phased else "/"
                gt_str = f"{gt[0]}{sep}{gt[1]}"
            gt_matrix[key][sample] = gt_str
    return gt_matrix

def encode_gt(gt):
    """Convert genotype to allele count (0, 1, 2, etc.)"""
    if gt in ["./.", ".|."]:
        return np.nan
    parts = gt.replace("|", "/").split("/")
    try:
        return sum(int(allele) for allele in parts)
    except:
        return np.nan


def calculate_maf_for_variant(genotypes, samples):
    """
    Calculate MAF for a specific variant across all samples
    Args:
        genotypes: dict of sample -> genotype for this variant
        samples: list of all samples
    Returns:
        MAF (float): Minor allele frequency
    """
    allele_counts = defaultdict(int)
    total_alleles = 0

    for sample in samples:
        if sample in genotypes:
            gt = genotypes[sample]
            if gt not in ["./.", ".|."]:
                # Parse genotype (e.g., "0|1" -> [0, 1])
                parts = gt.replace("|", "/").split("/")
                try:
                    for allele in parts:
                        allele_counts[int(allele)] += 1
                        total_alleles += 1
                except:
                    continue

    if total_alleles == 0:
        return 0.0

    # Calculate frequency for each allele
    allele_freqs = []
    for allele, count in allele_counts.items():
        freq = count / total_alleles
        allele_freqs.append(freq)

    # MAF is the frequency of the second most common allele
    # (or the frequency of non-reference allele if only 2 alleles)
    if len(allele_freqs) == 0:
        return 0.0
    elif len(allele_freqs) == 1:
        return 0.0  # monomorphic
    else:
        allele_freqs.sort(reverse=True)  # Sort in descending order
        return allele_freqs[1] if len(allele_freqs) > 1 else min(allele_freqs[0], 1 - allele_freqs[0])


def compute_iqs(y_true, y_pred):
    """Calculate IQS (Cohen's Kappa)"""
    if len(y_true) == 0:
        return np.nan

    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    n_total = np.sum(cm)
    if n_total == 0:
        return np.nan

    # P0 (observed agreement)
    p0 = np.trace(cm) / n_total

    # Pe (chance agreement)
    row_sums = np.sum(cm, axis=1)
    col_sums = np.sum(cm, axis=0)
    pe = np.sum(row_sums * col_sums) / (n_total ** 2)

    # IQS = (P0 - Pe) / (1 - Pe)
    if pe >= 1:
        return np.nan
    else:
        return (p0 - pe) / (1 - pe)


def analyze_by_maf_bins(test_gt, pred_gt, true_gt):
    """
    Analyze imputation performance across different MAF bins
    """
    print("=== MAF-based Analysis ===")

    # Get all samples
    all_samples = set()
    for variant_data in true_gt.values():
        all_samples.update(variant_data.keys())
    all_samples = list(all_samples)

    print(f"Total samples: {len(all_samples)}")

    # Calculate MAF for each variant and collect results
    variant_results = []

    for variant_key in test_gt:
        if variant_key not in true_gt:
            continue

        # Calculate MAF from true genotypes
        maf = calculate_maf_for_variant(true_gt[variant_key], all_samples)

        # Collect imputation results for this variant
        variant_true = []
        variant_pred = []

        for sample in test_gt[variant_key]:
            if test_gt[variant_key][sample] in [".|.", "./."]:  # Missing genotype
                try:
                    pred = pred_gt[variant_key][sample]
                    true = true_gt[variant_key][sample]
                    variant_true.append(true)
                    variant_pred.append(pred)
                except KeyError:
                    continue

        if len(variant_true) > 0:
            # Calculate accuracy for this variant
            correct = sum(1 for t, p in zip(variant_true, variant_pred) if t == p)
            accuracy = correct / len(variant_true)

            # Calculate other metrics
            iqs = compute_iqs(variant_true, variant_pred)

            variant_results.append({
                'variant': variant_key,
                'maf': maf,
                'n_imputed': len(variant_true),
                'accuracy': accuracy,
                'iqs': iqs,
                'y_true': variant_true,
                'y_pred': variant_pred
            })

    print(f"Total variants analyzed: {len(variant_results)}")

    # Define MAF bins (custom 6 bins)
    maf_bins = [
        (0.0, 0.05, "0.00-0.05"),
        (0.05, 0.1, "0.05-0.10"),
        (0.1, 0.2, "0.10-0.20"),
        (0.2, 0.3, "0.20-0.30"),
        (0.3, 0.4, "0.30-0.40"),
        (0.4, 0.5, "0.40-0.50")
    ]

    # Group variants by MAF bins
    bin_results = {}
    bin_detailed_results = {}  # Store detailed results for violin plots

    for min_maf, max_maf, bin_name in maf_bins:
        bin_variants = [v for v in variant_results if min_maf <= v['maf'] < max_maf]
        if bin_variants:
            # Aggregate results for this bin
            all_true = []
            all_pred = []
            accuracies = []
            iqs_values = []

            for v in bin_variants:
                all_true.extend(v['y_true'])
                all_pred.extend(v['y_pred'])
                accuracies.append(v['accuracy'])
                if not np.isnan(v['iqs']):
                    iqs_values.append(v['iqs'])

            # Calculate bin-level metrics
            bin_accuracy = sum(1 for t, p in zip(all_true, all_pred) if t == p) / len(all_true)
            bin_iqs = compute_iqs(all_true, all_pred)

            # Calculate INFO score and MaCH-Rsq if possible
            y_true_num = [encode_gt(gt) for gt in all_true]
            y_pred_num = [encode_gt(gt) for gt in all_pred]

            # Remove NaN values
            valid_pairs = [(t, p) for t, p in zip(y_true_num, y_pred_num)
                           if not (np.isnan(t) or np.isnan(p))]

            if len(valid_pairs) > 1:
                y_true_clean, y_pred_clean = zip(*valid_pairs)
                y_true_clean = np.array(y_true_clean)
                y_pred_clean = np.array(y_pred_clean)

                # Pearson R2
                r, _ = pearsonr(y_true_clean, y_pred_clean)
                r2 = r ** 2

                # INFO Score
                var_pred = np.var(y_pred_clean)
                var_true = np.var(y_true_clean)
                info_score = 1 - (var_pred / var_true) if var_true != 0 else np.nan

                # MaCH-Rsq
                true_centered = y_true_clean - np.mean(y_true_clean)
                pred_centered = y_pred_clean - np.mean(y_pred_clean)
                numerator = np.sum(true_centered * pred_centered) ** 2
                denominator = np.sum(true_centered ** 2) * np.sum(pred_centered ** 2)
                mach_rsq = numerator / denominator if denominator != 0 else np.nan
            else:
                r2 = np.nan
                info_score = np.nan
                mach_rsq = np.nan

            # Calculate individual variant metrics for violin plots
            variant_info_scores = []
            variant_mach_rsqs = []

            for v in bin_variants:
                y_true_var = [encode_gt(gt) for gt in v['y_true']]
                y_pred_var = [encode_gt(gt) for gt in v['y_pred']]

                valid_var_pairs = [(t, p) for t, p in zip(y_true_var, y_pred_var)
                                   if not (np.isnan(t) or np.isnan(p))]

                if len(valid_var_pairs) > 1:
                    y_true_var_clean, y_pred_var_clean = zip(*valid_var_pairs)
                    y_true_var_clean = np.array(y_true_var_clean)
                    y_pred_var_clean = np.array(y_pred_var_clean)

                    # INFO Score for this variant
                    var_pred_var = np.var(y_pred_var_clean)
                    var_true_var = np.var(y_true_var_clean)
                    info_var = 1 - (var_pred_var / var_true_var) if var_true_var != 0 else np.nan

                    # MaCH-Rsq for this variant
                    true_centered_var = y_true_var_clean - np.mean(y_true_var_clean)
                    pred_centered_var = y_pred_var_clean - np.mean(y_pred_var_clean)
                    numerator_var = np.sum(true_centered_var * pred_centered_var) ** 2
                    denominator_var = np.sum(true_centered_var ** 2) * np.sum(pred_centered_var ** 2)
                    mach_rsq_var = numerator_var / denominator_var if denominator_var != 0 else np.nan

                    if not np.isnan(info_var):
                        variant_info_scores.append(info_var)
                    if not np.isnan(mach_rsq_var):
                        variant_mach_rsqs.append(mach_rsq_var)

            bin_results[bin_name] = {
                'n_variants': len(bin_variants),
                'n_imputations': len(all_true),
                'accuracy': bin_accuracy,
                'iqs': bin_iqs,
                'r2': r2,
                'info_score': info_score,
                'mach_rsq': mach_rsq,
                'mean_maf': np.mean([v['maf'] for v in bin_variants])
            }

            # Store detailed results for violin plots
            bin_detailed_results[bin_name] = {
                'accuracies': accuracies,
                'iqs_values': iqs_values,
                'info_scores': variant_info_scores,
                'mach_rsqs': variant_mach_rsqs
            }

    return bin_results, bin_detailed_results


def plot_maf_results(bin_results):
    """Plot results similar to the figure"""

    # Prepare data for plotting
    bin_names = list(bin_results.keys())
    accuracies = [bin_results[b]['accuracy'] for b in bin_names]
    info_scores = [bin_results[b]['info_score'] for b in bin_names]
    mach_rsqs = [bin_results[b]['mach_rsq'] for b in bin_names]
    iqs_values = [bin_results[b]['iqs'] for b in bin_names]

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Imputation Performance vs MAF', fontsize=16)

    # Plot 1: Accuracy
    axes[0, 0].plot(bin_names, accuracies, 'o-', linewidth=2, markersize=8)
    axes[0, 0].set_title('Accuracy vs MAF')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: INFO Score
    axes[0, 1].plot(bin_names, info_scores, 's-', color='orange', linewidth=2, markersize=8)
    axes[0, 1].set_title('INFO Score vs MAF')
    axes[0, 1].set_ylabel('INFO Score')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: MaCH-Rsq
    axes[1, 0].plot(bin_names, mach_rsqs, '^-', color='green', linewidth=2, markersize=8)
    axes[1, 0].set_title('MaCH-Rsq vs MAF')
    axes[1, 0].set_ylabel('MaCH-Rsq')
    axes[1, 0].set_xlabel('MAF Bins')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: IQS
    axes[1, 1].plot(bin_names, iqs_values, 'd-', color='red', linewidth=2, markersize=8)
    axes[1, 1].set_title('IQS vs MAF')
    axes[1, 1].set_ylabel('IQS')
    axes[1, 1].set_xlabel('MAF Bins')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('maf_analysis_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_stacked_bar_chart(bin_results):
    """Create stacked bar chart for the four evaluation metrics"""

    bin_names = list(bin_results.keys())
    n_variants = [bin_results[b]['n_variants'] for b in bin_names]

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Stacked Bar Charts - Performance Metrics by MAF Bins', fontsize=16)

    # Define colors for each MAF bin
    colors = plt.cm.Set3(np.linspace(0, 1, len(bin_names)))

    # Subplot 1: Accuracy
    accuracies = [bin_results[b]['accuracy'] for b in bin_names]
    bars1 = axes[0, 0].bar(bin_names, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    axes[0, 0].set_title('Accuracy by MAF Bins', fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars1, accuracies)):
        axes[0, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Subplot 2: INFO Score
    info_scores = [bin_results[b]['info_score'] for b in bin_names]
    bars2 = axes[0, 1].bar(bin_names, info_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    axes[0, 1].set_title('INFO Score by MAF Bins', fontweight='bold')
    axes[0, 1].set_ylabel('INFO Score')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars2, info_scores)):
        if not np.isnan(val):
            axes[0, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                            f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Subplot 3: MaCH-Rsq
    mach_rsqs = [bin_results[b]['mach_rsq'] for b in bin_names]
    bars3 = axes[1, 0].bar(bin_names, mach_rsqs, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    axes[1, 0].set_title('MaCH-Rsq by MAF Bins', fontweight='bold')
    axes[1, 0].set_ylabel('MaCH-Rsq')
    axes[1, 0].set_xlabel('MAF Bins')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars3, mach_rsqs)):
        if not np.isnan(val):
            axes[1, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                            f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Subplot 4: IQS
    iqs_values = [bin_results[b]['iqs'] for b in bin_names]
    bars4 = axes[1, 1].bar(bin_names, iqs_values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    axes[1, 1].set_title('IQS by MAF Bins', fontweight='bold')
    axes[1, 1].set_ylabel('IQS')
    axes[1, 1].set_xlabel('MAF Bins')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars4, iqs_values)):
        if not np.isnan(val):
            axes[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                            f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig('maf_stacked_bar_chart.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_violin_plots(bin_detailed_results):
    """Create violin plots for the four evaluation metrics"""

    # Prepare data for violin plots
    violin_data = []

    for bin_name, data in bin_detailed_results.items():
        # Accuracy data
        for acc in data['accuracies']:
            violin_data.append({'MAF_Bin': bin_name, 'Metric': 'Accuracy', 'Value': acc})

        # IQS data
        for iqs in data['iqs_values']:
            if not np.isnan(iqs):
                violin_data.append({'MAF_Bin': bin_name, 'Metric': 'IQS', 'Value': iqs})

        # INFO Score data
        for info in data['info_scores']:
            if not np.isnan(info):
                violin_data.append({'MAF_Bin': bin_name, 'Metric': 'INFO Score', 'Value': info})

        # MaCH-Rsq data
        for mach in data['mach_rsqs']:
            if not np.isnan(mach):
                violin_data.append({'MAF_Bin': bin_name, 'Metric': 'MaCH-Rsq', 'Value': mach})

    # Convert to DataFrame
    df = pd.DataFrame(violin_data)

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Violin Plots - Distribution of Performance Metrics by MAF Bins', fontsize=16)

    # Define custom color palette
    bin_names = list(bin_detailed_results.keys())
    palette = dict(zip(bin_names, plt.cm.Set3(np.linspace(0, 1, len(bin_names)))))

    # Subplot 1: Accuracy
    accuracy_data = df[df['Metric'] == 'Accuracy']
    if not accuracy_data.empty:
        sns.violinplot(data=accuracy_data, x='MAF_Bin', y='Value',
                       palette=palette, ax=axes[0, 0], inner='quartile')
        axes[0, 0].set_title('Accuracy Distribution by MAF Bins', fontweight='bold')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_xlabel('')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        axes[0, 0].set_ylim(0, 1)

    # Subplot 2: INFO Score
    info_data = df[df['Metric'] == 'INFO Score']
    if not info_data.empty:
        sns.violinplot(data=info_data, x='MAF_Bin', y='Value',
                       palette=palette, ax=axes[0, 1], inner='quartile')
        axes[0, 1].set_title('INFO Score Distribution by MAF Bins', fontweight='bold')
        axes[0, 1].set_ylabel('INFO Score')
        axes[0, 1].set_xlabel('')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Subplot 3: MaCH-Rsq
    mach_data = df[df['Metric'] == 'MaCH-Rsq']
    if not mach_data.empty:
        sns.violinplot(data=mach_data, x='MAF_Bin', y='Value',
                       palette=palette, ax=axes[1, 0], inner='quartile')
        axes[1, 0].set_title('MaCH-Rsq Distribution by MAF Bins', fontweight='bold')
        axes[1, 0].set_ylabel('MaCH-Rsq')
        axes[1, 0].set_xlabel('MAF Bins')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Subplot 4: IQS
    iqs_data = df[df['Metric'] == 'IQS']
    if not iqs_data.empty:
        sns.violinplot(data=iqs_data, x='MAF_Bin', y='Value',
                       palette=palette, ax=axes[1, 1], inner='quartile')
        axes[1, 1].set_title('IQS Distribution by MAF Bins', fontweight='bold')
        axes[1, 1].set_ylabel('IQS')
        axes[1, 1].set_xlabel('MAF Bins')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('maf_violin_plots.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_stacked_bar_chart_with_error_bars(bin_results, bin_detailed_results):
    """Create stacked bar chart with error bars for the four evaluation metrics"""

    bin_names = list(bin_results.keys())
    n_variants = [bin_results[b]['n_variants'] for b in bin_names]

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Bar Charts with Error Bars - Performance Metrics by MAF Bins', fontsize=16)

    # Define colors for each MAF bin
    colors = plt.cm.Set3(np.linspace(0, 1, len(bin_names)))

    # Subplot 1: Accuracy
    accuracies = [bin_results[b]['accuracy'] for b in bin_names]
    # Calculate error bars for accuracy (standard deviation)
    accuracy_errors = []
    for bin_name in bin_names:
        if bin_name in bin_detailed_results:
            acc_values = bin_detailed_results[bin_name]['accuracies']
            if len(acc_values) > 1:
                accuracy_errors.append(np.std(acc_values))
            else:
                accuracy_errors.append(0)
        else:
            accuracy_errors.append(0)

    bars1 = axes[0, 0].bar(bin_names, accuracies, color=colors, alpha=0.8,
                           edgecolor='black', linewidth=0.5,
                           yerr=accuracy_errors, capsize=5, error_kw={'linewidth': 2, 'capthick': 2})
    axes[0, 0].set_title('Accuracy by MAF Bins (with Standard Deviation)', fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_ylim(0, min(1.1, max(accuracies) + max(accuracy_errors) + 0.1))
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, val, err) in enumerate(zip(bars1, accuracies, accuracy_errors)):
        axes[0, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + err + 0.01,
                        f'{val:.3f}±{err:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Subplot 2: INFO Score
    info_scores = [bin_results[b]['info_score'] for b in bin_names]
    # Calculate error bars for INFO Score
    info_errors = []
    for bin_name in bin_names:
        if bin_name in bin_detailed_results:
            info_values = bin_detailed_results[bin_name]['info_scores']
            if len(info_values) > 1:
                info_errors.append(np.std(info_values))
            else:
                info_errors.append(0)
        else:
            info_errors.append(0)

    # Handle NaN values for plotting
    info_scores_clean = []
    info_errors_clean = []
    for i, (score, err) in enumerate(zip(info_scores, info_errors)):
        if np.isnan(score):
            info_scores_clean.append(0)
            info_errors_clean.append(0)
        else:
            info_scores_clean.append(score)
            info_errors_clean.append(err)

    bars2 = axes[0, 1].bar(bin_names, info_scores_clean, color=colors, alpha=0.8,
                           edgecolor='black', linewidth=0.5,
                           yerr=info_errors_clean, capsize=5, error_kw={'linewidth': 2, 'capthick': 2})
    axes[0, 1].set_title('INFO Score by MAF Bins (with Standard Deviation)', fontweight='bold')
    axes[0, 1].set_ylabel('INFO Score')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, val, err) in enumerate(zip(bars2, info_scores, info_errors)):
        if not np.isnan(val):
            axes[0, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + err + 0.01,
                            f'{val:.3f}±{err:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    # Subplot 3: MaCH-Rsq
    mach_rsqs = [bin_results[b]['mach_rsq'] for b in bin_names]
    # Calculate error bars for MaCH-Rsq
    mach_errors = []
    for bin_name in bin_names:
        if bin_name in bin_detailed_results:
            mach_values = bin_detailed_results[bin_name]['mach_rsqs']
            if len(mach_values) > 1:
                mach_errors.append(np.std(mach_values))
            else:
                mach_errors.append(0)
        else:
            mach_errors.append(0)

    # Handle NaN values for plotting
    mach_rsqs_clean = []
    mach_errors_clean = []
    for i, (score, err) in enumerate(zip(mach_rsqs, mach_errors)):
        if np.isnan(score):
            mach_rsqs_clean.append(0)
            mach_errors_clean.append(0)
        else:
            mach_rsqs_clean.append(score)
            mach_errors_clean.append(err)

    bars3 = axes[1, 0].bar(bin_names, mach_rsqs_clean, color=colors, alpha=0.8,
                           edgecolor='black', linewidth=0.5,
                           yerr=mach_errors_clean, capsize=5, error_kw={'linewidth': 2, 'capthick': 2})
    axes[1, 0].set_title('MaCH-Rsq by MAF Bins (with Standard Deviation)', fontweight='bold')
    axes[1, 0].set_ylabel('MaCH-Rsq')
    axes[1, 0].set_xlabel('MAF Bins')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, val, err) in enumerate(zip(bars3, mach_rsqs, mach_errors)):
        if not np.isnan(val):
            axes[1, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + err + 0.01,
                            f'{val:.3f}±{err:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Subplot 4: IQS
    iqs_values = [bin_results[b]['iqs'] for b in bin_names]
    # Calculate error bars for IQS
    iqs_errors = []
    for bin_name in bin_names:
        if bin_name in bin_detailed_results:
            iqs_vals = bin_detailed_results[bin_name]['iqs_values']
            if len(iqs_vals) > 1:
                iqs_errors.append(np.std(iqs_vals))
            else:
                iqs_errors.append(0)
        else:
            iqs_errors.append(0)

    # Handle NaN values for plotting
    iqs_values_clean = []
    iqs_errors_clean = []
    for i, (score, err) in enumerate(zip(iqs_values, iqs_errors)):
        if np.isnan(score):
            iqs_values_clean.append(0)
            iqs_errors_clean.append(0)
        else:
            iqs_values_clean.append(score)
            iqs_errors_clean.append(err)

    bars4 = axes[1, 1].bar(bin_names, iqs_values_clean, color=colors, alpha=0.8,
                           edgecolor='black', linewidth=0.5,
                           yerr=iqs_errors_clean, capsize=5, error_kw={'linewidth': 2, 'capthick': 2})
    axes[1, 1].set_title('IQS by MAF Bins (with Standard Deviation)', fontweight='bold')
    axes[1, 1].set_ylabel('IQS')
    axes[1, 1].set_xlabel('MAF Bins')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, val, err) in enumerate(zip(bars4, iqs_values, iqs_errors)):
        if not np.isnan(val):
            axes[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + err + 0.01,
                            f'{val:.3f}±{err:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig('maf_bar_chart_with_error_bars.png', dpi=300, bbox_inches='tight')
    plt.show()