#!/usr/bin/env python3
"""
Illumination Robustness Statistical Tests: KAN-ACNet vs Baseline.

Included tests:
  1. Brightness-Stratified Paired Comparison (Wilcoxon per tertile)
  3. Variance Ratio Test (Brown-Forsythe)

Outputs to: ./illumination_stat_test_results
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
KAN_CSV = './illumination_robustness_results/combined_per_image.csv'
BASE_CSV = './baseline_illumination_robustness_results/combined_per_image.csv'
OUTPUT_DIR = './illumination_stat_test_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

RNG = np.random.default_rng(42)


def sig_label(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return 'ns'


# ──────────────────────────────────────────────────────────────────────
# Load & merge data
# ──────────────────────────────────────────────────────────────────────
kan_df = pd.read_csv(KAN_CSV)
base_df = pd.read_csv(BASE_CSV)
merged = kan_df.merge(base_df, on=['filename', 'dataset'], suffixes=('_kan', '_base'))

print(f"Merged {len(merged)} paired images across {merged['dataset'].nunique()} datasets\n")

merged['mean_brightness'] = merged['mean_brightness_kan']

all_results = []


# ======================================================================
# TEST 1: Brightness-Stratified Paired Comparison
# ======================================================================
print('=' * 70)
print('TEST 1: Brightness-Stratified Paired Comparison')
print('=' * 70)

merged['brightness_tertile'] = pd.qcut(
    merged['mean_brightness'], q=3, labels=['dark', 'medium', 'bright']
)

test1_rows = []

for tertile in ['dark', 'medium', 'bright']:
    sub = merged[merged['brightness_tertile'] == tertile]

    kan_dice = sub['dice_kan'].values
    base_dice = sub['dice_base'].values
    diff = kan_dice - base_dice

    w_stat, w_p = stats.wilcoxon(kan_dice, base_dice)

    cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0.0

    row = {
        'tertile': tertile,
        'n': len(sub),
        'kan_mean_dice': np.mean(kan_dice),
        'base_mean_dice': np.mean(base_dice),
        'mean_diff': np.mean(diff),
        'cohens_d': cohens_d,
        'wilcoxon_W': w_stat,
        'p_value': w_p,
        'significance': sig_label(w_p),
    }

    test1_rows.append(row)

    print(f"{tertile:>7s}  diff={np.mean(diff):+.4f}  d={cohens_d:+.4f}  "
          f"W={w_stat:.1f}  p={w_p:.2e} {sig_label(w_p)}")

test1_df = pd.DataFrame(test1_rows)
test1_df.to_csv(os.path.join(OUTPUT_DIR, 'test1_brightness_stratified.csv'), index=False)

all_results.append({
    'test': 'Brightness-Stratified Paired',
    'statistic_value': w_stat,
    'p_value': w_p,
    'effect_size': np.mean(diff),
})


# ======================================================================
# TEST 3: Variance Ratio (Brown-Forsythe)
# ======================================================================
print(f"\n{'=' * 70}")
print('TEST 3: Variance Ratio (Brown-Forsythe)')
print('=' * 70)

test3_rows = []

for tertile in ['dark', 'medium', 'bright']:
    sub = merged[merged['brightness_tertile'] == tertile]

    dice_kan = sub['dice_kan'].values
    dice_base = sub['dice_base'].values

    var_kan = np.var(dice_kan, ddof=1)
    var_base = np.var(dice_base, ddof=1)
    var_ratio = var_kan / var_base if var_base > 0 else np.nan

    bf_stat, bf_p = stats.levene(dice_kan, dice_base, center='median')

    row = {
        'brightness_bin': tertile,
        'n': len(sub),
        'var_kan': var_kan,
        'var_base': var_base,
        'var_ratio': var_ratio,
        'bf_statistic': bf_stat,
        'p_value': bf_p,
        'significance': sig_label(bf_p),
        'kan_lower_var': var_kan < var_base,
    }

    test3_rows.append(row)

    print(f"{tertile:>7s}  Var_KAN={var_kan:.6f}  Var_Base={var_base:.6f}  "
          f"ratio={var_ratio:.4f}  BF={bf_stat:.3f}  p={bf_p:.2e}")

# Overall
dice_kan_all = merged['dice_kan'].values
dice_base_all = merged['dice_base'].values

var_kan_all = np.var(dice_kan_all, ddof=1)
var_base_all = np.var(dice_base_all, ddof=1)

bf_all, bf_p_all = stats.levene(dice_kan_all, dice_base_all, center='median')

test3_rows.append({
    'brightness_bin': 'overall',
    'n': len(merged),
    'var_kan': var_kan_all,
    'var_base': var_base_all,
    'var_ratio': var_kan_all / var_base_all,
    'bf_statistic': bf_all,
    'p_value': bf_p_all,
    'significance': sig_label(bf_p_all),
    'kan_lower_var': var_kan_all < var_base_all,
})

test3_df = pd.DataFrame(test3_rows)
test3_df.to_csv(os.path.join(OUTPUT_DIR, 'test3_variance_ratio.csv'), index=False)

all_results.append({
    'test': 'Variance Ratio (Brown-Forsythe)',
    'statistic_value': bf_all,
    'p_value': bf_p_all,
    'effect_size': var_kan_all / var_base_all,
})


# ======================================================================
# Summary
# ======================================================================
summary_df = pd.DataFrame(all_results)
summary_df.to_csv(os.path.join(OUTPUT_DIR, 'summary_tests_1_and_3.csv'), index=False)


# ======================================================================
# Simple 2-panel figure
# ======================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Mean Dice per tertile
ax = axes[0]
x = np.arange(3)
w = 0.35
ax.bar(x - w/2, test1_df['kan_mean_dice'], w, label='KAN-ACNet')
ax.bar(x + w/2, test1_df['base_mean_dice'], w, label='Baseline')
ax.set_xticks(x)
ax.set_xticklabels(['dark', 'medium', 'bright'])
ax.set_title('Test 1: Stratified Dice')
ax.set_ylabel('Mean Dice')
ax.legend()

# Plot 2: Variance per bin
ax = axes[1]
bins_plot = ['dark', 'medium', 'bright', 'overall']
var_kan_vals = [test3_df.loc[test3_df['brightness_bin'] == b, 'var_kan'].values[0] for b in bins_plot]
var_base_vals = [test3_df.loc[test3_df['brightness_bin'] == b, 'var_base'].values[0] for b in bins_plot]
x2 = np.arange(len(bins_plot))
ax.bar(x2 - w/2, var_kan_vals, w, label='Var KAN')
ax.bar(x2 + w/2, var_base_vals, w, label='Var Base')
ax.set_xticks(x2)
ax.set_xticklabels(bins_plot)
ax.set_title('Test 3: Variance Comparison')
ax.set_ylabel('Variance')
ax.legend()

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'summary_2panel.png'), dpi=200)
plt.close()

print(f"\nAll outputs saved to {OUTPUT_DIR}/")
print("Done!")
