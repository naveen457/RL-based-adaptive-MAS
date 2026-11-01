"""
Experimental reproducibility, multi-seed aggregation, and table generation utilities
for research-grade reporting and verification.
"""

from __future__ import annotations

import datetime
import math
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence


# Critical values of Student's t-distribution for two-sided 95% confidence interval
_T_95_CRITICAL_VALUES: Dict[int, float] = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


@dataclass(frozen=True)
class SummaryStatistics:
    """Statistical summary across multiple experimental runs/seeds."""

    sample_count: int
    mean: float
    std: float
    ci95_low: float
    ci95_high: float
    ci95_margin: float
    min: float
    max: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_summary_statistics(values: Sequence[float]) -> SummaryStatistics:
    """Compute mean, sample std, min, max, and 95% confidence interval for a metric."""
    clean = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    n = len(clean)
    if n == 0:
        return SummaryStatistics(
            sample_count=0,
            mean=0.0,
            std=0.0,
            ci95_low=0.0,
            ci95_high=0.0,
            ci95_margin=0.0,
            min=0.0,
            max=0.0,
        )

    mean_val = float(sum(clean) / n)
    if n > 1:
        var = float(sum((x - mean_val) ** 2 for x in clean) / (n - 1))
        std_val = math.sqrt(max(0.0, var))
    else:
        std_val = 0.0

    min_val = min(clean)
    max_val = max(clean)

    if n > 1 and std_val > 0.0:
        df = n - 1
        t_crit = _T_95_CRITICAL_VALUES.get(df, 1.960 if df > 30 else 2.042)
        sem = std_val / math.sqrt(n)
        ci_margin = float(t_crit * sem)
    else:
        ci_margin = 0.0

    return SummaryStatistics(
        sample_count=n,
        mean=round(mean_val, 6),
        std=round(std_val, 6),
        ci95_low=round(mean_val - ci_margin, 6),
        ci95_high=round(mean_val + ci_margin, 6),
        ci95_margin=round(ci_margin, 6),
        min=round(min_val, 6),
        max=round(max_val, 6),
    )


def collect_experiment_metadata(
    seed: Optional[int] = None,
    hyperparameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Collect comprehensive system and environment metadata for experimental reproducibility."""
    # 1. Git Commit Hash
    git_hash = "unknown"
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            git_hash = proc.stdout.strip()
    except Exception:
        pass

    # 2. Framework Versions
    framework_versions: Dict[str, str] = {}
    for mod_name in ["torch", "langgraph", "langchain", "pydantic", "numpy"]:
        try:
            m = __import__(mod_name)
            framework_versions[mod_name] = getattr(m, "__version__", "installed")
        except ImportError:
            framework_versions[mod_name] = "not installed"

    # 3. Hardware Summary
    cuda_available = False
    cuda_device = "none"
    try:
        import torch
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            cuda_device = torch.cuda.get_device_name(0)
    except Exception:
        pass

    hardware_summary = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count() or 1,
        "cuda_available": cuda_available,
        "cuda_device_name": cuda_device,
    }

    return {
        "git_commit_hash": git_hash,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "seed": seed,
        "python_version": sys.version.split()[0],
        "framework_versions": framework_versions,
        "hardware_summary": hardware_summary,
        "hyperparameters": hyperparameters or {},
    }


def generate_markdown_table(headers: List[str], rows: List[List[Any]]) -> str:
    """Format headers and row data into a GitHub Flavored Markdown table."""
    if not headers:
        return ""
    col_widths = [len(str(h)) for h in headers]
    for row in rows:
        for idx, val in enumerate(row):
            if idx < len(col_widths):
                col_widths[idx] = max(col_widths[idx], len(str(val)))

    header_line = "| " + " | ".join(str(h).ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    separator_line = "| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |"
    data_lines = [
        "| " + " | ".join(str(val).ljust(col_widths[i]) for i, val in enumerate(row)) + " |"
        for row in rows
    ]
    return "\n".join([header_line, separator_line] + data_lines)


def generate_latex_table(
    headers: List[str],
    rows: List[List[Any]],
    caption: str = "Experimental Results",
    label: str = "tab:results",
) -> str:
    """Format headers and row data into a LaTeX booktabs table for research paper inclusion."""
    cols_spec = "l" + "c" * (len(headers) - 1)
    header_str = " & ".join(headers) + " \\\\"
    row_strs = [" & ".join(str(x) for x in r) + " \\\\" for r in rows]

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{cols_spec}}}",
        "\\toprule",
        header_str,
        "\\midrule",
    ]
    lines.extend(row_strs)
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    return "\n".join(lines)


def format_baseline_comparison_markdown(results: Dict[str, Any]) -> str:
    """Format baseline comparison summary statistics into a clean Markdown table."""
    headers = [
        "Policy Method",
        "Sample Count (N)",
        "Mean Final Reward",
        "Std Final Reward",
        "Mean Adaptation Gain",
        "Std Adaptation Gain",
    ]
    summary_stats = results.get("summary_statistics", {})
    rows = []
    display_names = {
        "shared_meta_policy": "Shared Neural Policy (Ours)",
        "fresh_scratch_policy": "Fresh Neural Policy (Scratch)",
        "deterministic_baseline": "Deterministic Static Baseline",
        "random_policy": "Random Policy",
    }
    for key, stats in summary_stats.items():
        name = display_names.get(key, key)
        rows.append([
            name,
            stats.get("sample_count", 0),
            f"{stats.get('mean_final_performance', 0.0):.4f}",
            f"{stats.get('std_final_performance', 0.0):.4f}",
            f"{stats.get('mean_adaptation_gain', 0.0):+.4f}",
            f"{stats.get('std_adaptation_gain', 0.0):.4f}",
        ])
    return generate_markdown_table(headers, rows)


def format_baseline_comparison_latex(
    results: Dict[str, Any],
    caption: str = "Cross-Task Adaptation Baseline Comparison on Unseen Tasks",
    label: str = "tab:baseline_comparison",
) -> str:
    """Format baseline comparison summary statistics into a LaTeX booktabs table."""
    headers = ["Policy Method", "N", "Final Perf (Mean $\\pm$ Std)", "Adaptation Gain (Mean $\\pm$ Std)"]
    summary_stats = results.get("summary_statistics", {})
    rows = []
    display_names = {
        "shared_meta_policy": "Shared Neural Policy (Ours)",
        "fresh_scratch_policy": "Fresh Neural Policy (Scratch)",
        "deterministic_baseline": "Deterministic Static Baseline",
        "random_policy": "Random Policy",
    }
    for key, stats in summary_stats.items():
        name = display_names.get(key, key)
        n = stats.get("sample_count", 0)
        mean_p = stats.get("mean_final_performance", 0.0)
        std_p = stats.get("std_final_performance", 0.0)
        mean_g = stats.get("mean_adaptation_gain", 0.0)
        std_g = stats.get("std_adaptation_gain", 0.0)
        rows.append([
            name,
            str(n),
            f"${mean_p:.4f} \\pm {std_p:.4f}$",
            f"${mean_g:+.4f} \\pm {std_g:.4f}$",
        ])
    return generate_latex_table(headers, rows, caption=caption, label=label)

