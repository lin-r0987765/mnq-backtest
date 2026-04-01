"""
Auto-Optimizer Module
======================
Iteratively tests parameter combinations to find optimal settings.
Used by the hourly scheduled task for continuous improvement.
"""

import numpy as np
import pandas as pd
import json
import os
import itertools
from datetime import datetime
from config import PARAM_RANGES, VERSION


def generate_param_grid(ranges=None):
    """Generate parameter combinations from ranges."""
    if ranges is None:
        ranges = PARAM_RANGES

    param_names = []
    param_values = []

    for name, (low, high, step) in ranges.items():
        values = np.arange(low, high + step/2, step).tolist()
        param_names.append(name)
        param_values.append(values)

    combinations = list(itertools.product(*param_values))
    return param_names, combinations


def run_optimization(df, run_single_backtest_fn, param_names, combinations, max_trials=50):
    """
    Run optimization over parameter combinations.
    run_single_backtest_fn(df, params_dict) -> metrics_dict

    Uses random sampling if combinations > max_trials.
    """
    results = []

    if len(combinations) > max_trials:
        indices = np.random.choice(len(combinations), max_trials, replace=False)
        sample = [combinations[i] for i in indices]
    else:
        sample = combinations

    for combo in sample:
        params = dict(zip(param_names, combo))
        try:
            metrics = run_single_backtest_fn(df, params)
            result = {**params, **metrics}
            results.append(result)
        except Exception as e:
            continue

    if not results:
        return None, None

    results_df = pd.DataFrame(results)

    # Composite score: weighted combination of key metrics
    results_df['composite_score'] = (
        results_df.get('sharpe_ratio', 0) * 0.3 +
        results_df.get('profit_factor', 0) * 0.2 +
        results_df.get('win_rate', 0) / 100 * 0.2 +
        (100 + results_df.get('max_drawdown_pct', 0)) / 100 * 0.15 +  # less drawdown = better
        results_df.get('total_return_pct', 0) / 100 * 0.15
    )

    best_idx = results_df['composite_score'].idxmax()
    best_params = {name: results_df.loc[best_idx, name] for name in param_names}
    best_metrics = results_df.loc[best_idx].to_dict()

    return best_params, results_df


def save_iteration_log(iteration, params, metrics, report_dir='iteration_logs'):
    """Save iteration results for tracking improvement over time."""
    os.makedirs(report_dir, exist_ok=True)

    log_entry = {
        'iteration': iteration,
        'timestamp': datetime.now().isoformat(),
        'version': VERSION,
        'parameters': params,
        'metrics': {k: v for k, v in metrics.items()
                    if isinstance(v, (int, float, str))},
    }

    log_file = os.path.join(report_dir, 'optimization_history.jsonl')
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry, default=str) + '\n')

    return log_file


def update_config_file(best_params, config_path='config.py'):
    """Update config.py with optimized parameters."""
    with open(config_path, 'r') as f:
        content = f.read()

    for param, value in best_params.items():
        import re
        if isinstance(value, float):
            pattern = rf'^{param}\s*=\s*[\d.]+\s*'
            replacement = f'{param} = {value}'
        else:
            pattern = rf'^{param}\s*=\s*\d+\s*'
            replacement = f'{param} = {int(value)}'
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

    # Update iteration counter
    match = re.search(r"ITERATION\s*=\s*(\d+)", content)
    new_iter = int(match.group(1)) + 1 if match else 1
    content = re.sub(
        r'^ITERATION\s*=\s*\d+',
        f'ITERATION = {new_iter}',
        content,
        flags=re.MULTILINE
    )

    with open(config_path, 'w') as f:
        f.write(content)

    return content


def compare_iterations(log_dir='iteration_logs'):
    """Compare performance across iterations."""
    log_file = os.path.join(log_dir, 'optimization_history.jsonl')
    if not os.path.exists(log_file):
        return None

    entries = []
    with open(log_file, 'r') as f:
        for line in f:
            entries.append(json.loads(line))

    if len(entries) < 2:
        return None

    comparison = {
        'total_iterations': len(entries),
        'first': entries[0],
        'latest': entries[-1],
        'improvements': {}
    }

    first_m = entries[0]['metrics']
    latest_m = entries[-1]['metrics']

    for key in ['sharpe_ratio', 'profit_factor', 'win_rate', 'max_drawdown_pct', 'total_return_pct']:
        if key in first_m and key in latest_m:
            comparison['improvements'][key] = {
                'first': first_m[key],
                'latest': latest_m[key],
                'change': latest_m[key] - first_m[key]
            }

    return comparison