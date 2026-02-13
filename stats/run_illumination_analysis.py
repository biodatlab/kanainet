#!/usr/bin/env python3
"""
Standalone script to run the illumination robustness analysis (Cells A-J)
from 01_stats_tests_and_mixed_effects.ipynb.

Skips the paired t-test (no baseline comparison needed yet).
Saves all plots to illumination_robustness_results/ instead of plt.show().
"""

import os
import sys
import warnings

# Headless matplotlib backend
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import cv2
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# =============================================================================
# 2. MICHELSON CONTRAST  (from notebook cell 2)
# =============================================================================

def michelson_contrast(image_array: np.ndarray) -> float:
    if len(image_array.shape) == 3:
        if image_array.shape[2] == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array.mean(axis=2).astype(np.uint8)
    else:
        gray = image_array
    l_min = float(np.min(gray))
    l_max = float(np.max(gray))
    if l_max + l_min == 0:
        return 0.0
    return float((l_max - l_min) / (l_max + l_min))


def rms_contrast(image_array: np.ndarray) -> float:
    if len(image_array.shape) == 3:
        if image_array.shape[2] == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array.mean(axis=2).astype(np.uint8)
    else:
        gray = image_array
    gray_norm = gray.astype(np.float32) / 255.0
    return float(np.std(gray_norm))


def compute_illumination_stats(image_array: np.ndarray) -> dict:
    if len(image_array.shape) == 3:
        if image_array.shape[2] == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            color_img = image_array
        else:
            gray = image_array.mean(axis=2).astype(np.uint8)
            color_img = image_array
    else:
        gray = image_array
        color_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    return {
        'mean_brightness': float(np.mean(gray)),
        'std_brightness': float(np.std(gray)),
        'median_brightness': float(np.median(gray)),
        'min_brightness': float(np.min(gray)),
        'max_brightness': float(np.max(gray)),
        'range_brightness': float(np.max(gray)) - float(np.min(gray)),
        'michelson_contrast': michelson_contrast(image_array),
        'rms_contrast': rms_contrast(image_array),
        'mean_r': float(np.mean(color_img[:, :, 0])),
        'mean_g': float(np.mean(color_img[:, :, 1])),
        'mean_b': float(np.mean(color_img[:, :, 2])),
        'std_r': float(np.std(color_img[:, :, 0])),
        'std_g': float(np.std(color_img[:, :, 1])),
        'std_b': float(np.std(color_img[:, :, 2])),
    }


def analyze_contrast_performance(df_results, metric='dice', contrast_col='michelson_contrast'):
    clean_df = df_results[[metric, contrast_col]].dropna()
    if clean_df[contrast_col].nunique() < 2:
        return {'pearson_r': float('nan'), 'pearson_p': float('nan'),
                'spearman_r': float('nan'), 'spearman_p': float('nan'),
                'binned_stats': None, 'n_samples': len(clean_df)}
    pearson_r, pearson_p = scipy_stats.pearsonr(clean_df[contrast_col], clean_df[metric])
    spearman_r, spearman_p = scipy_stats.spearmanr(clean_df[contrast_col], clean_df[metric])
    clean_df = clean_df.copy()
    # qcut with duplicates='drop' may produce fewer than 4 bins; use labels=False then map
    try:
        clean_df['contrast_bin'] = pd.qcut(clean_df[contrast_col], q=4,
                                           labels=['Low', 'Medium-Low', 'Medium-High', 'High'],
                                           duplicates='drop')
    except ValueError:
        clean_df['contrast_bin'] = pd.qcut(clean_df[contrast_col], q=4,
                                           labels=False, duplicates='drop')
    binned_stats = clean_df.groupby('contrast_bin')[metric].agg(['mean', 'std', 'count'])
    return {
        'pearson_r': pearson_r, 'pearson_p': pearson_p,
        'spearman_r': spearman_r, 'spearman_p': spearman_p,
        'binned_stats': binned_stats, 'n_samples': len(clean_df),
    }


def plot_contrast_analysis(df_results, metric='dice', contrast_col='michelson_contrast',
                           model_name='Model', save_path=None):
    clean_df = df_results[[metric, contrast_col]].dropna()
    if clean_df[contrast_col].nunique() < 2:
        print(f"  Skipping plot for {model_name}/{contrast_col}: constant values")
        return None
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].scatter(clean_df[contrast_col], clean_df[metric], alpha=0.5, s=30)
    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
        clean_df[contrast_col], clean_df[metric])
    x_line = np.linspace(clean_df[contrast_col].min(), clean_df[contrast_col].max(), 100)
    axes[0].plot(x_line, slope * x_line + intercept, 'r-', linewidth=2,
                 label=f'y = {slope:.2f}x + {intercept:.2f}')
    axes[0].set_xlabel('Michelson Contrast'); axes[0].set_ylabel('Dice Score')
    axes[0].set_title(f'{model_name}: Contrast vs {metric.capitalize()}\nr = {r_value:.3f}, p = {p_value:.4f}')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].hist(clean_df[contrast_col], bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(clean_df[contrast_col].mean(), color='red', linestyle='--',
                    label=f"Mean = {clean_df[contrast_col].mean():.3f}")
    axes[1].set_xlabel('Michelson Contrast'); axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Michelson Contrast'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    clean_df_temp = clean_df.copy()
    try:
        clean_df_temp['contrast_bin'] = pd.qcut(clean_df_temp[contrast_col], q=4,
                                                labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'],
                                                duplicates='drop')
    except ValueError:
        clean_df_temp['contrast_bin'] = pd.qcut(clean_df_temp[contrast_col], q=4,
                                                labels=False, duplicates='drop')
    binned = clean_df_temp.groupby('contrast_bin')[metric].agg(['mean', 'std'])
    x_pos = np.arange(len(binned))
    axes[2].bar(x_pos, binned['mean'], yerr=binned['std'], capsize=5, alpha=0.7, edgecolor='black')
    axes[2].set_xticks(x_pos); axes[2].set_xticklabels(binned.index, rotation=15)
    axes[2].set_ylabel(f'Mean {metric.capitalize()}'); axes[2].set_title('Performance by Contrast Quartile')
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Plot saved -> {save_path}")
    plt.close(fig)
    return fig


# =============================================================================
# 3. MIXED-EFFECTS (from notebook cell 3)
# =============================================================================
try:
    # Patch for numpy >= 1.25 / statsmodels < 0.14 incompatibility
    if not hasattr(np, 'MachAr'):
        np.MachAr = lambda: type('MachAr', (), {'eps': np.finfo(np.float64).eps})()
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    STATSMODELS_AVAILABLE = True
except (ImportError, AttributeError):
    STATSMODELS_AVAILABLE = False
    sm = None
    smf = None

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def mixed_effects_model(df, dependent_var='dice', fixed_effects=None,
                        random_effects=None, groups=None):
    if fixed_effects is None:
        fixed_effects = ['michelson_contrast', 'mean_brightness']

    if not STATSMODELS_AVAILABLE:
        print("Running simple OLS regression (statsmodels not available)")
        X = df[fixed_effects].copy(); y = df[dependent_var]
        model = LinearRegression(); model.fit(X, y)
        y_pred = model.predict(X); r2 = r2_score(y, y_pred)
        return {'model_type': 'OLS',
                'coefficients': dict(zip(['intercept'] + fixed_effects,
                                         [model.intercept_] + list(model.coef_))),
                'r_squared': r2, 'predictions': y_pred, 'residuals': y - y_pred}

    fixed_formula = ' + '.join(fixed_effects)
    formula = f'{dependent_var} ~ {fixed_formula}'
    if groups and random_effects:
        formula += f" + ({' + '.join(random_effects)}|{groups})"
    elif groups:
        formula += f" + (1|{groups})"

    print(f"Fitting model: {formula}")
    try:
        model = smf.mixedlm(formula, df, groups=groups if groups else df.index)
        result = model.fit(reml=False)
        return {'model_type': 'Mixed-Effects', 'formula': formula, 'result': result,
                'coefficients': result.fe_params.to_dict(),
                'random_effects': result.random_effects if hasattr(result, 'random_effects') else None,
                'aic': result.aic, 'bic': result.bic, 'summary': result.summary()}
    except Exception as e:
        print(f"Mixed-effects model failed: {e}\nFalling back to OLS regression...")
        formula = f'{dependent_var} ~ {" + ".join(fixed_effects)}'
        result = smf.ols(formula, data=df).fit()
        return {'model_type': 'OLS (fallback)', 'formula': formula, 'result': result,
                'coefficients': result.params.to_dict(),
                'r_squared': result.rsquared, 'adj_r_squared': result.rsquared_adj,
                'aic': result.aic, 'bic': result.bic,
                'pvalues': result.pvalues.to_dict(), 'summary': result.summary()}


def print_model_summary(model_result):
    print(f"\n{'='*70}")
    print(f"MIXED-EFFECTS MODEL RESULTS ({model_result['model_type']})")
    print(f"{'='*70}")
    if 'formula' in model_result:
        print(f"\nFormula: {model_result['formula']}")
    print(f"\nCoefficients:")
    for param, value in model_result['coefficients'].items():
        print(f"  {param}: {value:.6f}")
    if 'r_squared' in model_result:
        print(f"\nR-squared: {model_result['r_squared']:.6f}")
    if 'aic' in model_result:
        print(f"AIC: {model_result['aic']:.2f}")
    if 'bic' in model_result:
        print(f"BIC: {model_result['bic']:.2f}")
    if 'pvalues' in model_result:
        print(f"\nP-values:")
        for param, pval in model_result['pvalues'].items():
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            print(f"  {param}: {pval:.6f} {sig}")
    print(f"{'='*70}\n")


def plot_mixed_effects_diagnostics(model_result, df, dependent_var='dice', save_path=None):
    if 'result' not in model_result or not STATSMODELS_AVAILABLE:
        print("Full diagnostics require statsmodels"); return None
    result = model_result['result']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    if hasattr(result, 'resid'):
        residuals = result.resid; fitted = result.fittedvalues
        axes[0, 0].scatter(fitted, residuals, alpha=0.5)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Fitted Values'); axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Fitted'); axes[0, 0].grid(True, alpha=0.3)
        sm.graphics.qqplot(residuals, line='45', fit=True, ax=axes[0, 1])
        axes[0, 1].set_title('Normal Q-Q Plot'); axes[0, 1].grid(True, alpha=0.3)
        axes[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('Residuals'); axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Residuals')
        axes[1, 0].axvline(x=0, color='r', linestyle='--'); axes[1, 0].grid(True, alpha=0.3)
        axes[1, 1].scatter(fitted, np.sqrt(np.abs(residuals)), alpha=0.5)
        axes[1, 1].set_xlabel('Fitted Values'); axes[1, 1].set_ylabel('sqrt|Residuals|')
        axes[1, 1].set_title('Scale-Location Plot'); axes[1, 1].grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Diagnostics saved -> {save_path}")
    plt.close(fig)
    return fig


# =============================================================================
# CELL A: Setup & Model Loading
# =============================================================================
print("\n" + "=" * 70)
print("  Cell A: Setup & Model Loading")
print("=" * 70)

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, os.path.abspath('..'))
from config import Config
from models.kan_acnet import KANACNet
from dataset.kvasir_dataset import get_val_transform


def infer_kan_blocks_from_state_dict(state_dict):
    encoder_kan_blocks = []
    for i in range(1, 5):
        if f'kan_imm{i}.kan_modulator.spline_weight' in state_dict:
            encoder_kan_blocks.append(i)
    decoder_kan_blocks = []
    for i in range(1, 5):
        if f'decoder{i}.kan_bam.kan_edge_enhance.kan_transform.spline_weight' in state_dict:
            decoder_kan_blocks.append(i)
    return encoder_kan_blocks, decoder_kan_blocks


def load_model(checkpoint_path, device, encoder_kan_blocks=None, decoder_kan_blocks=None):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if encoder_kan_blocks is None:
        encoder_kan_blocks = checkpoint.get('encoder_kan_blocks')
    if decoder_kan_blocks is None:
        decoder_kan_blocks = checkpoint.get('decoder_kan_blocks')
    if encoder_kan_blocks is None or decoder_kan_blocks is None:
        inferred_enc, inferred_dec = infer_kan_blocks_from_state_dict(checkpoint['model_state_dict'])
        if encoder_kan_blocks is None: encoder_kan_blocks = inferred_enc
        if decoder_kan_blocks is None: decoder_kan_blocks = inferred_dec
    model = KANACNet(
        in_channels=Config.in_channels, num_classes=Config.num_classes,
        base_channels=Config.base_channels,
        use_pretrained_backbone=Config.use_pretrained_backbone,
        use_texture_pathway=Config.use_texture_pathway,
        backbone=Config.backbone,
        encoder_kan_blocks=encoder_kan_blocks,
        decoder_kan_blocks=decoder_kan_blocks,
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE = (512, 512)
CHECKPOINT_PATH = os.path.abspath('../best_model.pth')
OUTPUT_DIR = './illumination_robustness_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

model, checkpoint = load_model(CHECKPOINT_PATH, DEVICE)
checkpoint_metrics = checkpoint.get('metrics', {})
THRESHOLD = checkpoint_metrics.get('threshold', 0.4)
total_params = sum(p.numel() for p in model.parameters())
print(f"Device:     {DEVICE}")
print(f"Checkpoint: {CHECKPOINT_PATH}")
print(f"Threshold:  {THRESHOLD}")
print(f"Parameters: {total_params:,}")
print(f"Output dir: {OUTPUT_DIR}")


# =============================================================================
# CELL B: Per-Image Inference + Illumination Stats
# =============================================================================
print("\n" + "=" * 70)
print("  Cell B: Define per_image_inference_with_illumination()")
print("=" * 70)


def _get_mask_path(img_name, mask_dir):
    base, ext = os.path.splitext(img_name)
    candidates = [img_name, base + '.png', base + '.jpg', base + '.bmp',
                  base.replace('_images_', '_masks_') + '_mask' + ext,
                  base.replace('_images_', '_masks_') + '_mask.png',
                  base + '_mask' + ext, base + '_mask.png']
    for c in candidates:
        p = os.path.join(mask_dir, c)
        if os.path.exists(p):
            return p
    return None


def per_image_inference_with_illumination(model, image_dir, mask_dir, device,
                                          threshold, image_size=(512, 512), desc=''):
    val_tf = get_val_transform(image_size)
    image_files = sorted([f for f in os.listdir(image_dir)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))])
    rows = []
    for img_name in tqdm(image_files, desc=desc, leave=False):
        img_path = os.path.join(image_dir, img_name)
        mask_path = _get_mask_path(img_name, mask_dir)
        if mask_path is None: continue
        raw = cv2.imread(img_path)
        if raw is None: continue
        raw_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        illum = compute_illumination_stats(raw_rgb)
        mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_raw is None: continue
        mask_bin = (mask_raw > 127).astype(np.uint8)
        aug = val_tf(image=raw_rgb, mask=mask_bin)
        img_t = aug['image'].unsqueeze(0).to(device)
        mask_t = aug['mask'].float()
        if mask_t.dim() == 2: mask_t = mask_t.unsqueeze(0)
        with torch.no_grad():
            out = model(img_t)
            if isinstance(out, (tuple, list)): out = out[0]
            prob = torch.sigmoid(out).squeeze().cpu().numpy()
        pred = (prob >= threshold).astype(np.float32)
        gt = mask_t.squeeze().numpy().astype(np.float32)
        intersection = (pred * gt).sum()
        pred_sum = pred.sum(); gt_sum = gt.sum(); union = pred_sum + gt_sum
        dice = (2.0 * intersection / union) if union > 0 else (1.0 if gt_sum == 0 else 0.0)
        iou = (intersection / (union - intersection)) if (union - intersection) > 0 else (1.0 if gt_sum == 0 else 0.0)
        precision = (intersection / pred_sum) if pred_sum > 0 else (1.0 if gt_sum == 0 else 0.0)
        recall = (intersection / gt_sum) if gt_sum > 0 else (1.0 if pred_sum == 0 else 0.0)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        row = {'filename': img_name, 'dice': dice, 'iou': iou,
               'precision': precision, 'recall': recall, 'f1': f1}
        row.update(illum)
        rows.append(row)
    return pd.DataFrame(rows)

print("  per_image_inference_with_illumination() defined")


# =============================================================================
# CELL C: Run Inference on All 4 Datasets
# =============================================================================
print("\n" + "=" * 70)
print("  Cell C: Running inference on all datasets")
print("=" * 70)

datasets = {
    'kvasir_sessile':    '../data_seen/kvasir-sessile',
    'CVC_ColonDB':       '../data_unseen/CVC-ColonDB',
    'data_C6':           '../data_unseen/data_C6',
    'ETIS_LaribPolypDB': '../data_unseen/ETIS-LaribPolypDB',
}

dfs = {}
for name, root in datasets.items():
    img_dir = os.path.join(root, 'images')
    mask_dir = os.path.join(root, 'masks')
    print(f"\n--- {name} ---")
    df = per_image_inference_with_illumination(
        model, img_dir, mask_dir, DEVICE, THRESHOLD,
        image_size=IMAGE_SIZE, desc=name)
    df['dataset'] = name
    dfs[name] = df
    csv_path = os.path.join(OUTPUT_DIR, f'{name}_per_image.csv')
    df.to_csv(csv_path, index=False)
    print(f"  {len(df)} images  |  mean Dice = {df['dice'].mean():.4f}  |  saved -> {csv_path}")

combined_df = pd.concat(dfs.values(), ignore_index=True)
combined_csv = os.path.join(OUTPUT_DIR, 'combined_per_image.csv')
combined_df.to_csv(combined_csv, index=False)
print(f"\nCombined: {len(combined_df)} images  |  mean Dice = {combined_df['dice'].mean():.4f}")
print(f"Saved -> {combined_csv}")


# =============================================================================
# CELL D: Natural Variation Analysis
# =============================================================================
print("\n" + "=" * 70)
print("  Cell D: Natural Variation Analysis")
print("=" * 70)

illumination_vars = ['michelson_contrast', 'mean_brightness', 'rms_contrast', 'std_brightness']
correlation_records = []
analysis_sets = {**dfs, 'COMBINED': combined_df}

for set_name, df in analysis_sets.items():
    print(f"\n{'='*70}")
    print(f"  Natural Variation Analysis  --  {set_name}  ({len(df)} images)")
    print(f"{'='*70}")
    for illum_var in illumination_vars:
        if illum_var not in df.columns: continue
        res = analyze_contrast_performance(df, metric='dice', contrast_col=illum_var)
        if np.isnan(res['pearson_r']):
            interpretation = 'CONSTANT (no variation)'
        elif abs(res['pearson_r']) < 0.1:
            interpretation = 'ROBUST (|r|<0.1)'
        elif abs(res['pearson_r']) < 0.3:
            interpretation = 'MODERATE (0.1<=|r|<0.3)'
        else:
            interpretation = 'DEPENDENT (|r|>=0.3)'
        print(f"  {illum_var:25s}  Pearson r={res['pearson_r']:+.4f} (p={res['pearson_p']:.4f})  "
              f"Spearman r={res['spearman_r']:+.4f} (p={res['spearman_p']:.4f})  --> {interpretation}")
        correlation_records.append({
            'dataset': set_name, 'illumination_var': illum_var,
            'pearson_r': res['pearson_r'], 'pearson_p': res['pearson_p'],
            'spearman_r': res['spearman_r'], 'spearman_p': res['spearman_p'],
            'interpretation': interpretation})

    # Plot for the first non-constant illumination variable
    for plot_var in illumination_vars:
        if plot_var in df.columns and df[plot_var].nunique() >= 2:
            plot_contrast_analysis(df, metric='dice', contrast_col=plot_var,
                                   model_name=set_name,
                                   save_path=os.path.join(OUTPUT_DIR, f'contrast_analysis_{set_name}.png'))
            break

corr_df = pd.DataFrame(correlation_records)
corr_csv = os.path.join(OUTPUT_DIR, 'natural_variation_correlations.csv')
corr_df.to_csv(corr_csv, index=False)
print(f"\nCorrelation summary saved -> {corr_csv}")


# =============================================================================
# CELL E: Mixed-Effects Regression
# =============================================================================
print("\n" + "=" * 70)
print("  Cell E: Mixed-Effects Regression")
print("=" * 70)
print("Fitting: dice ~ michelson_contrast + mean_brightness + rms_contrast | (1|dataset)")

me_result = mixed_effects_model(
    combined_df, dependent_var='dice',
    fixed_effects=['michelson_contrast', 'mean_brightness', 'rms_contrast'],
    groups='dataset')

print_model_summary(me_result)

if 'result' in me_result:
    plot_mixed_effects_diagnostics(me_result, combined_df, dependent_var='dice',
                                   save_path=os.path.join(OUTPUT_DIR, 'mixed_effects_diagnostics.png'))

print("\nInterpretation:")
pvals = {}
if 'pvalues' in me_result:
    pvals = me_result['pvalues']
elif 'result' in me_result and hasattr(me_result['result'], 'pvalues'):
    pvals = me_result['result'].pvalues.to_dict()

for var in ['michelson_contrast', 'mean_brightness', 'rms_contrast']:
    if var in pvals:
        p = pvals[var]
        sig = ("SIGNIFICANT predictor of Dice (illumination-sensitive)" if p < 0.05
               else "NOT significant -> model is ROBUST to this variable")
        print(f"  {var:25s}  p={p:.4f}  {sig}")


# =============================================================================
# CELL F: Synthetic Perturbation Functions
# =============================================================================
print("\n" + "=" * 70)
print("  Cell F: Define perturbation functions")
print("=" * 70)


def apply_illumination_perturbation(image_rgb, perturb_type, severity):
    if perturb_type == 'brightness':
        tf = A.RandomBrightnessContrast(brightness_limit=(severity, severity),
                                        contrast_limit=(0, 0), always_apply=True)
    elif perturb_type == 'gamma':
        gamma_val = max(20, min(200, int(100 + severity * 80)))
        tf = A.RandomGamma(gamma_limit=(gamma_val, gamma_val), always_apply=True)
    elif perturb_type == 'contrast':
        tf = A.RandomBrightnessContrast(brightness_limit=(0, 0),
                                        contrast_limit=(severity, severity), always_apply=True)
    else:
        raise ValueError(f"Unknown perturbation type: {perturb_type}")
    return tf(image=image_rgb)['image']


def run_perturbation_sweep(model, image_dir, mask_dir, device, threshold,
                           perturb_type='brightness', severities=None,
                           image_size=(512, 512)):
    if severities is None:
        severities = np.linspace(-0.5, 0.5, 11)
    val_tf = get_val_transform(image_size)
    image_files = sorted([f for f in os.listdir(image_dir)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))])
    records = []
    for sev in tqdm(severities, desc=f'{perturb_type} sweep'):
        dices = []
        for img_name in image_files:
            img_path = os.path.join(image_dir, img_name)
            mask_path = _get_mask_path(img_name, mask_dir)
            if mask_path is None: continue
            raw = cv2.imread(img_path)
            if raw is None: continue
            raw_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            perturbed = apply_illumination_perturbation(raw_rgb, perturb_type, sev)
            mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_raw is None: continue
            mask_bin = (mask_raw > 127).astype(np.uint8)
            aug = val_tf(image=perturbed, mask=mask_bin)
            img_t = aug['image'].unsqueeze(0).to(device)
            mask_t = aug['mask'].float()
            if mask_t.dim() == 2: mask_t = mask_t.unsqueeze(0)
            with torch.no_grad():
                out = model(img_t)
                if isinstance(out, (tuple, list)): out = out[0]
                prob = torch.sigmoid(out).squeeze().cpu().numpy()
            pred = (prob >= threshold).astype(np.float32)
            gt = mask_t.squeeze().numpy().astype(np.float32)
            inter = (pred * gt).sum(); total = pred.sum() + gt.sum()
            dice = (2 * inter / total) if total > 0 else (1.0 if gt.sum() == 0 else 0.0)
            dices.append(dice)
        records.append({'severity': float(sev), 'mean_dice': float(np.mean(dices)),
                        'std_dice': float(np.std(dices)), 'n_images': len(dices)})
    return pd.DataFrame(records)

print("  apply_illumination_perturbation() and run_perturbation_sweep() defined")


# =============================================================================
# CELL G: Run Synthetic Perturbation Sweeps
# =============================================================================
print("\n" + "=" * 70)
print("  Cell G: Running perturbation sweeps on kvasir-sessile (196 images)")
print("=" * 70)

PERTURB_IMG_DIR = '../data_seen/kvasir-sessile/images'
PERTURB_MASK_DIR = '../data_seen/kvasir-sessile/masks'
PERTURB_TYPES = ['brightness', 'gamma', 'contrast']
SEVERITIES = np.linspace(-0.5, 0.5, 11)

sweep_results = {}
for pt in PERTURB_TYPES:
    print(f"\nRunning {pt} sweep ...")
    sweep_df = run_perturbation_sweep(
        model, PERTURB_IMG_DIR, PERTURB_MASK_DIR, DEVICE, THRESHOLD,
        perturb_type=pt, severities=SEVERITIES, image_size=IMAGE_SIZE)
    sweep_results[pt] = sweep_df
    csv_path = os.path.join(OUTPUT_DIR, f'sweep_{pt}.csv')
    sweep_df.to_csv(csv_path, index=False)
    print(f"  saved -> {csv_path}")
    print(sweep_df.to_string(index=False))

print("\nAll perturbation sweeps complete.")


# =============================================================================
# CELL H: Perturbation Visualization + Robustness Scores
# =============================================================================
print("\n" + "=" * 70)
print("  Cell H: Perturbation Visualization")
print("=" * 70)

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
colors = {'brightness': '#3498db', 'gamma': '#e74c3c', 'contrast': '#2ecc71'}
robustness_scores = {}

for idx, pt in enumerate(PERTURB_TYPES):
    df = sweep_results[pt]; ax = axes[idx]
    ax.errorbar(df['severity'].values, df['mean_dice'].values, yerr=df['std_dice'].values,
                fmt='o-', color=colors[pt], capsize=4, linewidth=2, markersize=6)
    baseline_dice = df.loc[df['severity'].abs().idxmin(), 'mean_dice']
    ax.axhline(baseline_dice, color='gray', linestyle='--', alpha=0.6,
               label=f'baseline Dice={baseline_dice:.4f}')
    ax.axvline(0, color='gray', linestyle=':', alpha=0.4)
    ax.set_xlabel('Severity', fontsize=12)
    if idx == 0: ax.set_ylabel('Mean Dice', fontsize=12)
    ax.set_title(f'{pt.capitalize()} Sweep', fontsize=13)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    max_drop = baseline_dice - df['mean_dice'].min()
    max_drop_pct = 100 * max_drop / baseline_dice if baseline_dice > 0 else 0
    auc = np.trapz(df['mean_dice'], df['severity']) / (df['severity'].max() - df['severity'].min())
    norm_auc = auc / baseline_dice if baseline_dice > 0 else 0
    robustness_scores[pt] = {'baseline_dice': baseline_dice, 'min_dice': df['mean_dice'].min(),
                             'max_dice_drop': max_drop, 'max_dice_drop_pct': max_drop_pct,
                             'normalized_auc': norm_auc}

plt.suptitle('Synthetic Illumination Perturbation Sweeps (kvasir-sessile)', fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'perturbation_sweeps.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Plot saved -> {os.path.join(OUTPUT_DIR, 'perturbation_sweeps.png')}")

print(f"\n{'Perturbation':<15} {'Baseline':>10} {'Min Dice':>10} {'Max Drop':>10} {'Max Drop%':>10} {'Norm AUC':>10}")
print('-' * 65)
for pt, sc in robustness_scores.items():
    print(f"{pt:<15} {sc['baseline_dice']:>10.4f} {sc['min_dice']:>10.4f} "
          f"{sc['max_dice_drop']:>10.4f} {sc['max_dice_drop_pct']:>9.2f}% {sc['normalized_auc']:>10.4f}")


# =============================================================================
# CELL I: Qualitative Examples
# =============================================================================
print("\n" + "=" * 70)
print("  Cell I: Qualitative Examples (3 images x 3 conditions)")
print("=" * 70)

val_tf = get_val_transform(IMAGE_SIZE)
sample_files = sorted([f for f in os.listdir(PERTURB_IMG_DIR)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
np.random.seed(42)
sample_indices = np.random.choice(len(sample_files), size=min(3, len(sample_files)), replace=False)
sample_names = [sample_files[i] for i in sorted(sample_indices)]

conditions = [('Dark (sev=-0.4)', 'brightness', -0.4),
              ('Original', None, 0.0),
              ('Bright (sev=+0.4)', 'brightness', 0.4)]

fig, axes = plt.subplots(len(sample_names), len(conditions),
                          figsize=(4*len(conditions), 4*len(sample_names)))
if len(sample_names) == 1:
    axes = axes[np.newaxis, :]

for row, img_name in enumerate(sample_names):
    img_path = os.path.join(PERTURB_IMG_DIR, img_name)
    mask_path = _get_mask_path(img_name, PERTURB_MASK_DIR)
    raw = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask_bin = (mask_raw > 127).astype(np.uint8)

    for col, (label, pt, sev) in enumerate(conditions):
        ax = axes[row, col]
        disp = apply_illumination_perturbation(raw, pt, sev) if pt is not None else raw.copy()
        aug = val_tf(image=disp, mask=mask_bin)
        img_t = aug['image'].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(img_t)
            if isinstance(out, (tuple, list)): out = out[0]
            prob = torch.sigmoid(out).squeeze().cpu().numpy()
        pred = (prob >= THRESHOLD).astype(np.uint8)
        gt = aug['mask'].numpy().astype(np.float32)
        if gt.ndim == 3: gt = gt.squeeze(0)
        inter = (pred.astype(np.float32) * gt).sum()
        total = pred.sum() + gt.sum()
        dice_val = (2 * inter / total) if total > 0 else 1.0
        disp_resized = cv2.resize(disp, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
        ax.imshow(disp_resized)
        if pred.max() > 0:
            ax.contour(pred, levels=[0.5], colors='lime', linewidths=1.5)
        gt_display = cv2.resize(mask_bin, (IMAGE_SIZE[1], IMAGE_SIZE[0]),
                                interpolation=cv2.INTER_NEAREST)
        if gt_display.max() > 0:
            ax.contour(gt_display, levels=[0.5], colors='red', linewidths=1.0, linestyles='--')
        ax.set_title(f'{label}\nDice={dice_val:.3f}', fontsize=10)
        ax.axis('off')

from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color='lime', linewidth=2, label='Prediction'),
                   Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='Ground Truth')]
fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=11, frameon=True)
plt.suptitle('Qualitative Examples: Dark / Original / Bright', fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'qualitative_examples.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Plot saved -> {os.path.join(OUTPUT_DIR, 'qualitative_examples.png')}")


# =============================================================================
# CELL J: Summary Report
# =============================================================================
print("\n" + "=" * 80)
print("          ILLUMINATION ROBUSTNESS SUMMARY  --  best_model.pth")
print("=" * 80)

print("\n1. NATURAL VARIATION ANALYSIS (Pearson |r| vs Dice, combined images)")
print("-" * 80)
combined_corrs = corr_df[corr_df['dataset'] == 'COMBINED']
print(f"  {'Illumination Variable':25s} {'Pearson r':>12} {'p-value':>12} {'Interpretation':>25}")
for _, row in combined_corrs.iterrows():
    print(f"  {row['illumination_var']:25s} {row['pearson_r']:>+12.4f} {row['pearson_p']:>12.4f} {row['interpretation']:>25}")

print("\n2. MIXED-EFFECTS REGRESSION  (dice ~ illumination vars | dataset)")
print("-" * 80)
if pvals:
    for var in ['michelson_contrast', 'mean_brightness', 'rms_contrast']:
        if var in pvals:
            p = pvals[var]
            status = "NOT significant (robust)" if p >= 0.05 else "SIGNIFICANT (sensitive)"
            print(f"  {var:25s}  p = {p:.4f}  -->  {status}")
else:
    print("  (p-values not available from model output)")

print("\n3. SYNTHETIC PERTURBATION ROBUSTNESS (kvasir-sessile, 196 images)")
print("-" * 80)
print(f"  {'Perturbation':15s} {'Baseline Dice':>14} {'Worst Dice':>12} {'Max Drop %':>12} {'Norm AUC':>10}")
for pt, sc in robustness_scores.items():
    print(f"  {pt:15s} {sc['baseline_dice']:>14.4f} {sc['min_dice']:>12.4f} "
          f"{sc['max_dice_drop_pct']:>11.2f}% {sc['normalized_auc']:>10.4f}")

print("\n" + "=" * 80)
print("  OVERALL VERDICT")
print("=" * 80)
max_abs_r = combined_corrs['pearson_r'].abs().max()
nat_robust = max_abs_r < 0.15
max_drop = max(sc['max_dice_drop_pct'] for sc in robustness_scores.values())
syn_robust = max_drop < 5.0

if nat_robust and syn_robust:
    verdict = "The model handles illumination well regardless of conditions."
elif nat_robust:
    verdict = (f"Natural variation: ROBUST (max |r| = {max_abs_r:.3f}).  "
               f"Synthetic perturbation: MODERATE (max drop = {max_drop:.1f}%).")
elif syn_robust:
    verdict = (f"Synthetic perturbation: ROBUST (max drop = {max_drop:.1f}%).  "
               f"Natural variation: some correlation detected (max |r| = {max_abs_r:.3f}).")
else:
    verdict = (f"Illumination sensitivity detected.  "
               f"Max natural |r| = {max_abs_r:.3f}, max synthetic drop = {max_drop:.1f}%.")

print(f"\n  Max |Pearson r| across illumination vars (combined): {max_abs_r:.4f}  "
      f"{'< 0.15 (robust)' if nat_robust else '>= 0.15 (some sensitivity)'}")
print(f"  Max synthetic Dice drop:                             {max_drop:.2f}%  "
      f"{'< 5% (robust)' if syn_robust else '>= 5% (some sensitivity)'}")
print(f"\n  --> {verdict}")
print("=" * 80)

summary_path = os.path.join(OUTPUT_DIR, 'summary_report.txt')
with open(summary_path, 'w') as f:
    f.write("Illumination Robustness Summary -- best_model.pth\n")
    f.write(f"Max |Pearson r| (natural variation, combined): {max_abs_r:.4f}\n")
    f.write(f"Max synthetic Dice drop: {max_drop:.2f}%\n")
    f.write(f"Verdict: {verdict}\n")
print(f"\nSummary saved -> {summary_path}")
print("\nDone!")
