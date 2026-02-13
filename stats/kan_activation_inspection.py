#!/usr/bin/env python3
"""
KAN Activation Function Inspection & Similarity Testing

Provides tools for:
1. KAN activation function inspection - Visualizing learned activation functions
2. Similarity testing - Comparing KAN activations with standard activation functions
3. B-spline analysis - Understanding the basis function composition
4. Batch analysis of multiple checkpoints

Usage:
    python 02_kan_activation_inspection.py --checkpoint /path/to/best_model.pth
    python 02_kan_activation_inspection.py --checkpoint /path/to/best_model.pth --all-channels
    python 02_kan_activation_inspection.py --demo
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn


# =============================================================================
# STANDARD ACTIVATION FUNCTIONS
# =============================================================================

def standard_activations() -> dict:
    """Return dictionary of standard activation functions."""
    return {
        'ReLU': lambda x: np.maximum(0, x),
        'Sigmoid': lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500))),
        'Tanh': lambda x: np.tanh(x),
        'GELU': lambda x: 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))),
        'Swish': lambda x: x / (1 + np.exp(-np.clip(x, -500, 500))),
        'Mish': lambda x: x * np.tanh(np.log(1 + np.exp(np.clip(x, -20, 20)))),
        'LeakyReLU': lambda x: np.where(x > 0, x, 0.01 * x),
        'ELU': lambda x: np.where(x > 0, x, 0.1 * (np.exp(x) - 1)),
        'SiLU': lambda x: x / (1 + np.exp(-np.clip(x, -500, 500))),  # Same as Swish
    }


# =============================================================================
# KAN ACTIVATION INSPECTOR
# =============================================================================

class KANActivationInspector:
    """
    Inspect and compare KAN activation functions with standard activations.
    KAN uses B-spline based learnable activation functions.
    """

    KAN_PATTERNS = [
        'spline_weight', 'spline_scaler', 'kan_modulator',
        'kan_transform', 'kan_fusion', 'kan_texture', 'kan_edge_enhance'
    ]

    def __init__(self, checkpoint_path: str = None):
        self.checkpoint_path = checkpoint_path
        self.learned_activations = {}
        self.kan_layers = {}
        self.standard_activations = standard_activations()

        if checkpoint_path and Path(checkpoint_path).exists():
            self._load_kan_activations()

    def _load_kan_activations(self):
        """Load learned KAN activation parameters from checkpoint."""
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)

            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            for key, value in state_dict.items():
                if any(pattern in key for pattern in self.KAN_PATTERNS):
                    self.learned_activations[key] = value

            print(f"Loaded {len(self.learned_activations)} KAN activation parameters")
            if len(self.learned_activations) > 0:
                print("Sample keys:", list(self.learned_activations.keys())[:5])
                self._organize_kan_layers()
        except Exception as e:
            print(f"Could not load checkpoint: {e}")

    def _organize_kan_layers(self):
        """Organize KAN parameters by layer name."""
        layer_groups = {}

        for key in self.learned_activations.keys():
            parts = key.split('.')
            module_path = []
            for part in parts:
                module_path.append(part)
                if any(pattern in part for pattern in [
                    'kan_modulator', 'kan_transform', 'kan_fusion',
                    'kan_texture', 'kan_edge_enhance'
                ]):
                    break

            module_name = '.'.join(module_path)
            if module_name not in layer_groups:
                layer_groups[module_name] = []
            layer_groups[module_name].append(key)

        for module_name, keys in layer_groups.items():
            spline_key = None
            base_key = None
            grid_key = None

            for key in keys:
                if 'spline_weight' in key:
                    spline_key = key
                elif 'base_weight' in key:
                    base_key = key
                elif 'grid' in key and 'grid' not in key.replace('grid', '').replace('.', ''):
                    grid_key = key

            self.kan_layers[module_name] = {
                'spline_weight': self.learned_activations.get(spline_key),
                'base_weight': self.learned_activations.get(base_key),
                'grid': self.learned_activations.get(grid_key),
                'all_keys': keys
            }

    def list_kan_layers(self):
        """Print all available KAN layers in the checkpoint."""
        if not self.kan_layers:
            print("No KAN layers found. Make sure checkpoint contains KAN parameters.")
            return

        print(f"\nFound {len(self.kan_layers)} KAN modules:")
        for i, (name, params) in enumerate(sorted(self.kan_layers.items()), 1):
            spline_shape = params['spline_weight'].shape if params['spline_weight'] is not None else "N/A"
            if 'kan_imm' in name:
                layer_type = "Encoder Immediate"
            elif 'decoder' in name:
                layer_type = "Decoder"
            elif 'fusion' in name:
                layer_type = "Fusion"
            elif 'texture' in name:
                layer_type = "Texture"
            else:
                layer_type = "Unknown"
            print(f"  {i:2}. {layer_type:12} | {name}: {spline_shape}")

    def b_spline_basis(self, x: np.ndarray, grid: np.ndarray, order: int = 3) -> np.ndarray:
        """Compute B-spline basis functions."""
        x_clipped = np.clip(x, grid[0], grid[-1])

        basis = []
        for i in range(len(grid) - 1):
            b = np.zeros_like(x)
            mask = (x >= grid[i]) & (x < grid[i + 1])
            b[mask] = 1.0
            basis.append(b)

        while len(basis) < len(grid) + order:
            basis.append(np.zeros_like(x))

        return np.column_stack(basis)

    def simulate_kan_activation(self,
                                x: np.ndarray,
                                spline_coeffs: np.ndarray,
                                grid: np.ndarray = None,
                                base_weight: float = 0.0) -> np.ndarray:
        """Simulate KAN activation function given learned parameters."""
        if grid is None:
            grid_size = len(spline_coeffs) - 3
            grid = np.linspace(-1, 1, grid_size + 1)

        x_norm = np.tanh(x)
        basis = self.b_spline_basis(x_norm, grid)

        n_basis = basis.shape[1]
        if len(spline_coeffs) != n_basis:
            if len(spline_coeffs) < n_basis:
                padded_coeffs = np.zeros(n_basis)
                padded_coeffs[:len(spline_coeffs)] = spline_coeffs
                spline_coeffs = padded_coeffs
            else:
                spline_coeffs = spline_coeffs[:n_basis]

        spline_out = np.dot(basis, spline_coeffs)
        base_out = base_weight * x_norm

        return spline_out + base_out

    def get_layer_activation(self, layer_name: str, channel_idx: int = 0, input_idx: int = 0):
        """Extract and simulate activation for a specific KAN layer."""
        if layer_name not in self.kan_layers:
            print(f"Layer '{layer_name}' not found. Available layers:")
            self.list_kan_layers()
            return None, None

        params = self.kan_layers[layer_name]
        spline_weight = params['spline_weight']
        base_weight = params['base_weight']
        grid = params['grid']

        if spline_weight is None:
            print(f"No spline_weight found for layer '{layer_name}'")
            return None, None

        if spline_weight.ndim == 3:
            out_ch = min(channel_idx, spline_weight.shape[0] - 1)
            in_ch = min(input_idx, spline_weight.shape[1] - 1)
            coeffs = spline_weight[out_ch, in_ch].cpu().numpy()
            base = base_weight[out_ch, in_ch].cpu().numpy() if base_weight is not None else 0.0
        elif spline_weight.ndim == 2:
            out_ch = min(channel_idx, spline_weight.shape[0] - 1)
            coeffs = spline_weight[out_ch].cpu().numpy()
            base = base_weight[out_ch].cpu().numpy() if base_weight is not None else 0.0
        else:
            coeffs = spline_weight.flatten().cpu().numpy()
            base = 0.0

        grid_np = grid.cpu().numpy() if grid is not None else None

        x = np.linspace(-3, 3, 200)
        activation = self.simulate_kan_activation(x, coeffs, grid_np, base)

        return x, activation

    def compare_to_standard(self, kan_activation: np.ndarray, x: np.ndarray) -> dict:
        """Compare KAN activation to standard activation functions."""
        results = {}

        for name, func in self.standard_activations.items():
            standard_out = func(x)
            corr, p_value = pearsonr(kan_activation.flatten(), standard_out.flatten())
            mse = np.mean((kan_activation.flatten() - standard_out.flatten())**2)

            results[name] = {
                'correlation': corr,
                'p_value': p_value,
                'mse': mse
            }

        results = dict(sorted(results.items(),
                              key=lambda item: item[1]['correlation'],
                              reverse=True))
        return results

    def plot_comparison(self,
                        kan_activation: np.ndarray,
                        x: np.ndarray,
                        top_n: int = 3,
                        title: str = "KAN Activation vs Standard Activations",
                        save_path: str = None):
        """Plot KAN activation alongside most similar standard activations."""
        comparisons = self.compare_to_standard(kan_activation, x)
        top_matches = list(comparisons.items())[:top_n]

        fig, axes = plt.subplots(1, top_n + 1, figsize=(5 * (top_n + 1), 4))

        axes[0].plot(x, kan_activation, 'b-', linewidth=2, label='KAN Activation')
        axes[0].set_xlabel('Input')
        axes[0].set_ylabel('Output')
        axes[0].set_title('KAN Activation Function')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        for idx, (name, metrics) in enumerate(top_matches, 1):
            standard_out = self.standard_activations[name](x)
            axes[idx].plot(x, kan_activation, 'b-', linewidth=2, label='KAN', alpha=0.7)
            axes[idx].plot(x, standard_out, 'r--', linewidth=2, label=name)
            axes[idx].set_xlabel('Input')
            axes[idx].set_ylabel('Output')
            axes[idx].set_title(f'{name} (r={metrics["correlation"]:.3f})')
            axes[idx].grid(True, alpha=0.3)
            axes[idx].legend()

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

        return fig

    def analyze_activation_shape(self, activation: np.ndarray) -> dict:
        """Analyze the shape characteristics of an activation function."""
        return {
            'monotonic': bool(
                np.all(np.diff(activation) >= -1e-6) or np.all(np.diff(activation) <= 1e-6)
            ),
            'bounded_below': float(np.min(activation)) if np.min(activation) > -1e6 else -np.inf,
            'bounded_above': float(np.max(activation)) if np.max(activation) < 1e6 else np.inf,
            'zero_centered': abs(np.mean(activation)) < 0.1,
            'approx_linear': float(np.corrcoef(activation, np.linspace(0, 1, len(activation)))[0, 1]),
            'saturation_ratio': float(np.sum(np.abs(np.diff(activation)) < 1e-3) / len(activation))
        }

    def classify_activation_type(self, activation: np.ndarray, x: np.ndarray) -> tuple:
        """Classify the KAN activation as most similar to a standard type."""
        comparisons = self.compare_to_standard(activation, x)
        best_match = max(comparisons.items(), key=lambda item: item[1]['correlation'])
        return best_match[0], best_match[1]['correlation']


# =============================================================================
# COMPREHENSIVE SIMILARITY TESTING
# =============================================================================

def comprehensive_similarity_test(kan_activation: np.ndarray,
                                  std_activations: dict,
                                  x: np.ndarray) -> pd.DataFrame:
    """Perform comprehensive similarity testing between KAN and standard activations."""
    results = []

    for name, func in std_activations.items():
        standard_out = func(x)
        kan_flat = kan_activation.flatten()
        std_flat = standard_out.flatten()

        pearson_corr, pearson_p = pearsonr(kan_flat, std_flat)
        spearman_corr, spearman_p = spearmanr(kan_flat, std_flat)
        kendall_corr, kendall_p = kendalltau(kan_flat, std_flat)

        cosine_dist = cosine(kan_flat, std_flat)
        euclidean_dist = euclidean(kan_flat, std_flat)

        mse = mean_squared_error(std_flat, kan_flat)
        mae = mean_absolute_error(std_flat, kan_flat)

        normalized_pearson = (pearson_corr + 1) / 2
        normalized_cosine = 1 - cosine_dist

        results.append({
            'activation': name,
            'pearson_r': pearson_corr,
            'pearson_p': pearson_p,
            'spearman_r': spearman_corr,
            'spearman_p': spearman_p,
            'kendall_tau': kendall_corr,
            'kendall_p': kendall_p,
            'cosine_distance': cosine_dist,
            'euclidean_distance': euclidean_dist,
            'mse': mse,
            'mae': mae,
            'normalized_pearson': normalized_pearson,
            'normalized_cosine': normalized_cosine,
            'combined_score': (normalized_pearson + normalized_cosine) / 2
        })

    df = pd.DataFrame(results)
    df = df.sort_values('combined_score', ascending=False)
    return df


def plot_similarity_heatmap(similarity_df: pd.DataFrame,
                            metrics: list = None,
                            figsize: tuple = (12, 8),
                            save_path: str = None):
    """Plot heatmap of similarity metrics."""
    if metrics is None:
        metrics = ['pearson_r', 'spearman_r', 'kendall_tau', 'normalized_cosine', 'combined_score']

    plot_df = similarity_df.set_index('activation')[metrics]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(plot_df.values, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)

    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(plot_df.index)))
    ax.set_xticklabels(metrics)
    ax.set_yticklabels(plot_df.index)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Similarity Score')

    for i in range(len(plot_df.index)):
        for j in range(len(metrics)):
            ax.text(j, i, f'{plot_df.values[i, j]:.3f}',
                    ha="center", va="center", color="black", fontsize=9)

    ax.set_title('KAN Activation Similarity to Standard Activations')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return fig


def plot_activation_overlay(kan_activation: np.ndarray,
                            std_activations: dict,
                            x: np.ndarray,
                            top_n: int = 5,
                            save_path: str = None):
    """Plot KAN activation overlaid with top standard activations."""
    similarity_df = comprehensive_similarity_test(kan_activation, std_activations, x)
    top_activations = similarity_df.head(top_n)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, kan_activation, 'b-', linewidth=3, label='KAN Activation', alpha=0.8)

    colors = plt.cm.tab10(np.linspace(0, 1, top_n))
    for i, (idx, row) in enumerate(top_activations.iterrows()):
        name = row['activation']
        standard_out = std_activations[name](x)
        ax.plot(x, standard_out, '--', linewidth=2, color=colors[i],
                label=f"{name} (r={row['pearson_r']:.3f})")

    ax.set_xlabel('Input', fontsize=12)
    ax.set_ylabel('Output', fontsize=12)
    ax.set_title('KAN Activation vs Most Similar Standard Activations', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return fig


def generate_similarity_report(kan_activation: np.ndarray,
                               std_activations: dict,
                               x: np.ndarray) -> str:
    """Generate a text report of similarity analysis."""
    similarity_df = comprehensive_similarity_test(kan_activation, std_activations, x)

    report = []
    report.append("=" * 70)
    report.append("KAN ACTIVATION SIMILARITY REPORT")
    report.append("=" * 70)

    top = similarity_df.iloc[0]
    report.append(f"\nMost Similar Activation: {top['activation']}")
    report.append(f"  Pearson Correlation: {top['pearson_r']:.4f} (p={top['pearson_p']:.6f})")
    report.append(f"  Spearman Correlation: {top['spearman_r']:.4f} (p={top['spearman_p']:.6f})")
    report.append(f"  Combined Similarity Score: {top['combined_score']:.4f}")

    report.append(f"\nTop 5 Most Similar Activations:")
    for i, (idx, row) in enumerate(similarity_df.head(5).iterrows(), 1):
        report.append(f"  {i}. {row['activation']}: ")
        report.append(f"     Pearson r={row['pearson_r']:.4f}, MSE={row['mse']:.4f}")

    report.append(f"\nActivation Shape Characteristics:")
    report.append(f"  Range: [{kan_activation.min():.4f}, {kan_activation.max():.4f}]")
    report.append(f"  Mean: {kan_activation.mean():.4f}, Std: {kan_activation.std():.4f}")
    report.append(f"  Monotonic: {np.all(np.diff(kan_activation) >= 0) or np.all(np.diff(kan_activation) <= 0)}")
    report.append("=" * 70)

    return "\n".join(report)


# =============================================================================
# BATCH ANALYSIS
# =============================================================================

def analyze_all_kan_layers(checkpoint_path: str,
                           output_dir: str = './kan_analysis_results',
                           save_plots: bool = True,
                           save_json: bool = True,
                           sample_channels: bool = True):
    """Analyze all KAN layers in a checkpoint, save plots and similarity results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plots_dir = output_path / 'plots'
    json_dir = output_path / 'json'
    if save_plots:
        plots_dir.mkdir(exist_ok=True)
    if save_json:
        json_dir.mkdir(exist_ok=True)

    inspector = KANActivationInspector(checkpoint_path)

    if not inspector.kan_layers:
        print("No KAN layers found in checkpoint!")
        return None

    checkpoint_name = Path(checkpoint_path).parent.name
    print(f"\n{'=' * 70}")
    print(f"ANALYZING MODEL: {checkpoint_name}")
    print(f"{'=' * 70}")

    inspector.list_kan_layers()

    all_results = {
        'model_name': checkpoint_name,
        'checkpoint_path': str(checkpoint_path),
        'analysis_date': datetime.now().isoformat(),
        'num_kan_layers': len(inspector.kan_layers),
        'layers': {}
    }

    for layer_name in sorted(inspector.kan_layers.keys()):
        if 'kan_imm' in layer_name:
            layer_type = 'encoder'
            clean_name = layer_name.replace('.kan_modulator', '')
        elif 'decoder' in layer_name:
            layer_type = 'decoder'
            clean_name = layer_name.replace('.kan_bam.kan_edge_enhance', '')
        elif 'fusion' in layer_name:
            layer_type = 'fusion'
            clean_name = layer_name.replace('.kan_fusion', '')
        elif 'texture' in layer_name:
            layer_type = 'texture'
            clean_name = layer_name.replace('.kan_texture', '')
        else:
            layer_type = 'other'
            clean_name = layer_name

        print(f"\n{'=' * 70}")
        print(f"Processing Layer: {layer_name} ({layer_type})")
        print(f"{'=' * 70}")

        layer_params = inspector.kan_layers[layer_name]
        layer_results = {
            'layer_name': layer_name,
            'clean_name': clean_name,
            'layer_type': layer_type,
            'spline_weight_shape': list(layer_params['spline_weight'].shape) if layer_params['spline_weight'] is not None else None,
            'base_weight_shape': list(layer_params['base_weight'].shape) if layer_params['base_weight'] is not None else None,
            'grid_shape': list(layer_params['grid'].shape) if layer_params['grid'] is not None else None,
            'channels_analyzed': []
        }

        spline_weight = layer_params['spline_weight']
        if spline_weight is None:
            continue

        if spline_weight.ndim == 3:
            if sample_channels:
                n_out_channels = min(spline_weight.shape[0], 3)
                n_in_channels = min(spline_weight.shape[1], 2)
            else:
                n_out_channels = spline_weight.shape[0]
                n_in_channels = spline_weight.shape[1]
        elif spline_weight.ndim == 2:
            n_out_channels = min(spline_weight.shape[0], 3) if sample_channels else spline_weight.shape[0]
            n_in_channels = 1
        else:
            n_out_channels = 1
            n_in_channels = 1

        mode_str = "Sampling" if sample_channels else "Analyzing all"
        print(f"{mode_str} {n_out_channels} output channels x {n_in_channels} input channels")

        for out_ch in range(n_out_channels):
            for in_ch in range(n_in_channels):
                x, activation = inspector.get_layer_activation(layer_name, out_ch, in_ch)
                if x is None:
                    continue

                comparisons = inspector.compare_to_standard(activation, x)
                shape_info = inspector.analyze_activation_shape(activation)

                channel_results = {
                    'output_channel': out_ch,
                    'input_channel': in_ch,
                    'best_match': list(comparisons.keys())[0],
                    'best_correlation': float(comparisons[list(comparisons.keys())[0]]['correlation']),
                    'all_comparisons': {
                        name: {
                            'correlation': float(metrics['correlation']),
                            'p_value': float(metrics['p_value']),
                            'mse': float(metrics['mse'])
                        }
                        for name, metrics in comparisons.items()
                    },
                    'shape_analysis': {
                        k: float(v) if not isinstance(v, bool) else v
                        for k, v in shape_info.items()
                    },
                    'activation_stats': {
                        'min': float(activation.min()),
                        'max': float(activation.max()),
                        'mean': float(activation.mean()),
                        'std': float(activation.std())
                    }
                }
                layer_results['channels_analyzed'].append(channel_results)

                if save_plots:
                    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

                    axes[0].plot(x, activation, 'b-', linewidth=2)
                    axes[0].set_xlabel('Input')
                    axes[0].set_ylabel('Output')
                    axes[0].set_title(f'KAN: {clean_name}\n(ch={out_ch}, in={in_ch})', fontsize=10)
                    axes[0].grid(True, alpha=0.3)

                    top_matches = list(comparisons.items())[:3]
                    colors = ['red', 'green', 'orange']
                    for idx, (name, metrics) in enumerate(top_matches):
                        standard_out = inspector.standard_activations[name](x)
                        axes[idx + 1].plot(x, activation, 'b-', linewidth=2, label='KAN', alpha=0.7)
                        axes[idx + 1].plot(x, standard_out, '--', linewidth=2, color=colors[idx], label=name)
                        axes[idx + 1].set_xlabel('Input')
                        axes[idx + 1].set_ylabel('Output')
                        axes[idx + 1].set_title(f'{name} (r={metrics["correlation"]:.3f})', fontsize=10)
                        axes[idx + 1].grid(True, alpha=0.3)
                        axes[idx + 1].legend(fontsize=8)

                    plt.suptitle(f'{layer_type.upper()}: {clean_name}', fontsize=12, fontweight='bold')
                    plt.tight_layout()

                    safe_name = clean_name.replace('.', '_')
                    plot_path = plots_dir / f'{layer_type}_{safe_name}_ch{out_ch}_in{in_ch}.png'
                    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    print(f"  Saved: {plot_path.name}")

        all_results['layers'][layer_name] = layer_results

    # Summary statistics
    all_correlations = []
    all_best_matches = []
    correlations_by_type = {'encoder': [], 'decoder': [], 'fusion': [], 'texture': [], 'other': []}

    for layer_name, layer_data in all_results['layers'].items():
        lt = layer_data['layer_type']
        for ch in layer_data['channels_analyzed']:
            all_correlations.append(ch['best_correlation'])
            all_best_matches.append(ch['best_match'])
            if lt in correlations_by_type:
                correlations_by_type[lt].append(ch['best_correlation'])

    if all_correlations:
        all_results['summary'] = {
            'total_channels_analyzed': len(all_correlations),
            'mean_best_correlation': float(np.mean(all_correlations)),
            'std_best_correlation': float(np.std(all_correlations)),
            'min_best_correlation': float(np.min(all_correlations)),
            'max_best_correlation': float(np.max(all_correlations)),
            'most_common_activation': max(set(all_best_matches), key=all_best_matches.count),
            'activation_distribution': {name: all_best_matches.count(name) for name in set(all_best_matches)},
            'by_layer_type': {
                lt: {
                    'count': len(correlations_by_type[lt]),
                    'mean_correlation': float(np.mean(correlations_by_type[lt])) if correlations_by_type[lt] else 0.0,
                    'std_correlation': float(np.std(correlations_by_type[lt])) if len(correlations_by_type[lt]) > 1 else 0.0,
                }
                for lt in correlations_by_type.keys()
            }
        }

    if save_json:
        json_path = json_dir / f'{checkpoint_name}_kan_analysis.json'
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n{'=' * 70}")
        print(f"Results saved to: {json_path}")
        print(f"{'=' * 70}")

    return all_results


def batch_analyze_checkpoints(checkpoint_paths: list,
                              base_output_dir: str = './kan_analysis_results',
                              save_plots: bool = True,
                              save_json: bool = True):
    """Analyze multiple checkpoints and generate a comparative summary."""
    all_summaries = []

    for ckpt_path in checkpoint_paths:
        print(f"\n\n{'#' * 70}")
        print(f"# PROCESSING: {ckpt_path}")
        print(f"{'#' * 70}")

        ckpt_name = Path(ckpt_path).parent.name
        output_dir = Path(base_output_dir) / ckpt_name

        results = analyze_all_kan_layers(
            checkpoint_path=ckpt_path,
            output_dir=str(output_dir),
            save_plots=save_plots,
            save_json=save_json
        )

        if results and 'summary' in results:
            all_summaries.append({
                'model': ckpt_name,
                'checkpoint_path': str(ckpt_path),
                'num_kan_layers': results['num_kan_layers'],
                'total_channels_analyzed': results['summary']['total_channels_analyzed'],
                'mean_best_correlation': results['summary']['mean_best_correlation'],
                'most_common_activation': results['summary']['most_common_activation'],
                'activation_distribution': results['summary']['activation_distribution']
            })

    if all_summaries:
        summary_path = Path(base_output_dir) / 'combined_summary.json'
        with open(summary_path, 'w') as f:
            json.dump({
                'analysis_date': datetime.now().isoformat(),
                'num_models': len(all_summaries),
                'models': all_summaries
            }, f, indent=2)
        print(f"\n{'=' * 70}")
        print(f"Combined summary saved to: {summary_path}")
        print(f"{'=' * 70}")

        print(f"\n{'=' * 70}")
        print("MODEL COMPARISON")
        print(f"{'=' * 70}")
        print(f"{'Model':<40} {'Layers':<8} {'Channels':<10} {'Mean Corr':<12} {'Top Activation':<15}")
        print(f"{'-' * 70}")
        for s in all_summaries:
            print(f"{s['model']:<40} {s['num_kan_layers']:<8} {s['total_channels_analyzed']:<10} "
                  f"{s['mean_best_correlation']:<12.4f} {s['most_common_activation']:<15}")
        print(f"{'=' * 70}")

    return all_summaries


# =============================================================================
# DEMO
# =============================================================================

def demo_kan_activation_inspection():
    """Demonstrate KAN activation inspection with simulated data."""
    print("KAN Activation Function Inspection Demo")
    print("=" * 60)

    inspector = KANActivationInspector()
    x = np.linspace(-3, 3, 200)

    grid_size = 7
    grid = np.linspace(-1, 1, grid_size + 1)

    # Example 1: ReLU-like
    relu_like_coeffs = np.zeros(10)
    relu_like_coeffs[3:] = np.linspace(0, 2, 7)
    relu_kan = inspector.simulate_kan_activation(x, relu_like_coeffs, grid, base_weight=0.1)

    print("\nExample 1: ReLU-like KAN Activation")
    best_type, corr = inspector.classify_activation_type(relu_kan, x)
    print(f"Most similar to: {best_type} (correlation: {corr:.4f})")
    shape_analysis = inspector.analyze_activation_shape(relu_kan)
    print(f"Shape analysis: {shape_analysis}")
    inspector.plot_comparison(relu_kan, x, title="ReLU-like KAN Activation")

    # Example 2: Sigmoid-like
    sigmoid_like_coeffs = np.linspace(0, 1, 10)
    sigmoid_kan = inspector.simulate_kan_activation(x, sigmoid_like_coeffs, grid, base_weight=0.0)

    print("\nExample 2: Sigmoid-like KAN Activation")
    best_type, corr = inspector.classify_activation_type(sigmoid_kan, x)
    print(f"Most similar to: {best_type} (correlation: {corr:.4f})")
    shape_analysis = inspector.analyze_activation_shape(sigmoid_kan)
    print(f"Shape analysis: {shape_analysis}")
    inspector.plot_comparison(sigmoid_kan, x, title="Sigmoid-like KAN Activation")

    # Example 3: Custom
    custom_coeffs = np.sin(np.linspace(0, 2 * np.pi, 10)) * 0.5 + 0.5
    custom_kan = inspector.simulate_kan_activation(x, custom_coeffs, grid, base_weight=0.2)

    print("\nExample 3: Custom KAN Activation")
    best_type, corr = inspector.classify_activation_type(custom_kan, x)
    print(f"Most similar to: {best_type} (correlation: {corr:.4f})")
    shape_analysis = inspector.analyze_activation_shape(custom_kan)
    print(f"Shape analysis: {shape_analysis}")
    inspector.plot_comparison(custom_kan, x, title="Custom KAN Activation")

    return inspector


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="KAN Activation Function Inspection & Similarity Testing"
    )
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--output-dir', type=str, default='./kan_analysis_results',
                        help='Output directory for results (default: ./kan_analysis_results)')
    parser.add_argument('--layer', type=str, default=None,
                        help='Specific layer to inspect (default: all layers)')
    parser.add_argument('--channel', type=int, default=0,
                        help='Output channel index (default: 0)')
    parser.add_argument('--input-channel', type=int, default=0,
                        help='Input channel index (default: 0)')
    parser.add_argument('--all-channels', action='store_true',
                        help='Analyze all channels (slower)')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip saving plots')
    parser.add_argument('--no-json', action='store_true',
                        help='Skip saving JSON results')
    parser.add_argument('--demo', action='store_true',
                        help='Run demo with simulated data')
    parser.add_argument('--batch', nargs='+', type=str, default=None,
                        help='Multiple checkpoint paths for batch analysis')
    args = parser.parse_args()

    if args.demo:
        demo_kan_activation_inspection()
        return

    if args.batch:
        batch_analyze_checkpoints(
            checkpoint_paths=args.batch,
            base_output_dir=args.output_dir,
            save_plots=not args.no_plots,
            save_json=not args.no_json
        )
        return

    if args.checkpoint is None:
        parser.print_help()
        print("\nError: --checkpoint or --demo is required.")
        return

    if args.layer:
        # Single layer inspection
        inspector = KANActivationInspector(args.checkpoint)
        if not inspector.kan_layers:
            print("No KAN layers found in checkpoint.")
            return

        x, activation = inspector.get_layer_activation(args.layer, args.channel, args.input_channel)
        if x is None:
            return

        print(f"\nAnalyzing Layer: {args.layer}")
        print(f"Channel: {args.channel}, Input: {args.input_channel}")

        best_type, corr = inspector.classify_activation_type(activation, x)
        print(f"Most similar to: {best_type} (correlation: {corr:.4f})")

        shape_analysis = inspector.analyze_activation_shape(activation)
        print(f"Shape analysis: {shape_analysis}")

        report = generate_similarity_report(activation, standard_activations(), x)
        print(report)

        inspector.plot_comparison(activation, x,
                                  title=f"KAN Activation: {args.layer} (ch={args.channel}, in={args.input_channel})")
    else:
        # Full analysis of all layers
        results = analyze_all_kan_layers(
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            save_plots=not args.no_plots,
            save_json=not args.no_json,
            sample_channels=not args.all_channels
        )

        if results and 'summary' in results:
            print(f"\n{'=' * 70}")
            print("ANALYSIS SUMMARY")
            print(f"{'=' * 70}")
            print(f"Model: {results['model_name']}")
            print(f"KAN Modules Found: {results['num_kan_layers']}")
            print(f"Total Channels Analyzed: {results['summary']['total_channels_analyzed']}")
            print(f"Overall Mean Best Correlation: {results['summary']['mean_best_correlation']:.4f}")
            print(f"Most Common Activation Type: {results['summary']['most_common_activation']}")

            print(f"\nBy Layer Type:")
            for layer_type, stats in results['summary']['by_layer_type'].items():
                if stats['count'] > 0:
                    print(f"  {layer_type.capitalize():10} | Count: {stats['count']:2} | Mean Corr: {stats['mean_correlation']:.4f}")

            print(f"\nActivation Distribution:")
            for act, count in results['summary']['activation_distribution'].items():
                print(f"  {act}: {count}")
            print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
