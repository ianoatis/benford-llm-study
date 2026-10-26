#!/usr/bin/env python3
"""First-digit Benford analysis: 50 datasets of 500 amounts from each model.

Install: python -m pip install numpy pandas matplotlib
Run:     python analyze_benford_concise.py --data-dir "/path/to/AI Datasets"

Writes dataset results, one model summary, a duplicate report, two figures,
captions, and reproducibility metadata. Original files and repeated outputs
are retained. See README.md for the analysis assumptions and output definitions.
"""
import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import io
import json
from pathlib import Path
import platform
import re

import matplotlib
matplotlib.use("Agg")  # Save figures without opening windows.
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


# Study settings: edit here only when needed to match your documented protocol.
MODELS = [
    # Folder, filename prefix, researcher-reported model label, figure color
    ("Gemini", "Gemini", "Gemini 3.6 Flash", "#0072B2"),
    ("ChatGPT", "GPT", "GPT-6 Luna", "#D55E00"),
    ("Claude", "Claude", "Claude Sonnet 5", "#009E73"),
]
FILES_PER_MODEL = 50
AMOUNTS_PER_FILE = 500
SIMULATIONS = 100_000
BOOTSTRAPS = 20_000
SEED = 20261016
ALPHA = 0.05
DIGITS = np.arange(1, 10)
BENFORD = np.log10(1 + 1 / DIGITS)
COUNT_COLUMNS = [f"count_{d}" for d in DIGITS]
HASH_COLUMNS = ["file_sha256", "sequence_sha256", "collection_sha256"]


def rng(purpose, model_index=0):
    """Separate reproducible streams for simulation, bootstrap, and plot jitter."""
    return np.random.default_rng(np.random.SeedSequence([SEED, purpose, model_index]))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def parse_amount(text):
    """Accept positive amounts with one or two decimals; avoid float rounding."""
    text = text.strip()
    if not re.fullmatch(r"[0-9]+\.[0-9]{1,2}", text) or Decimal(text) <= 0:
        raise ValueError("Amount must be positive with one or two decimal places")
    amount = Decimal(text)
    return amount.as_tuple().digits[0], format(amount, ".2f")


def valid_date(text):
    """Date problems are warnings, not exclusions from first-digit analysis."""
    for pattern in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return 2024 <= datetime.strptime(text.strip(), pattern).year <= 2025
        except ValueError:
            pass
    return False


def read_dataset(path):
    """Reject a whole file on its first structural/amount error; never delete rows."""
    record = dict(status="missing", reason="File not found", rows=np.nan,
                  warnings="", one_decimal_amounts=np.nan)
    record.update({key: "" for key in HASH_COLUMNS})
    record.update({key: np.nan for key in COUNT_COLUMNS})
    if not path.is_file():
        return record
    try:
        raw = path.read_bytes()
        record["file_sha256"] = digest(raw)
        rows = list(csv.reader(io.StringIO(raw.decode("utf-8-sig"), newline=""), strict=True))
        if not rows:
            raise ValueError("Empty CSV")
        headers = [h.strip().lower() for h in rows[0]]
        record["rows"] = len(rows) - 1
        if not all(headers) or len(headers) != len(set(headers)) or "amount" not in headers:
            raise ValueError("Missing amount column, or empty/duplicate column names")
        if record["rows"] != AMOUNTS_PER_FILE:
            raise ValueError(f"Expected {AMOUNTS_PER_FILE} rows; found {record['rows']}")
        amount_index = headers.index("amount")
        date_index = headers.index("date") if "date" in headers else None
        warnings = [] if set(headers) == {"date", "amount"} else ["Expected only date and amount columns"]
        digits, amounts, bad_dates, one_decimal = [], [], 0, 0
        for row_number, row in enumerate(rows[1:], start=1):
            if len(row) != len(headers):
                raise ValueError(f"Data row {row_number}: inconsistent field count")
            try:
                digit, canonical = parse_amount(row[amount_index])
            except ValueError as error:
                raise ValueError(f"Data row {row_number}: {error}") from error
            digits.append(digit)
            amounts.append(canonical)
            one_decimal += len(row[amount_index].strip().split(".")[1]) == 1
            if date_index is not None and not valid_date(row[date_index]):
                bad_dates += 1
        if bad_dates:
            warnings.append(f"{bad_dates} unrecognized dates or dates outside 2024–2025")
        record.update(status="valid", reason="", warnings="; ".join(warnings),
                      one_decimal_amounts=one_decimal)
        record.update(zip(COUNT_COLUMNS, np.bincount(digits, minlength=10)[1:10]))
        record["sequence_sha256"] = digest("\n".join(amounts).encode())
        # Sorting canonical strings is sufficient for equality of collections.
        # Repeated amounts remain repeated; dates and row order are ignored.
        record["collection_sha256"] = digest("\n".join(sorted(amounts)).encode())
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        record.update(status="invalid", reason=str(error))
    return record


def scores(counts):
    """TV = half the absolute proportion difference; Pearson X² uses counts."""
    counts = np.asarray(counts, dtype=float)
    expected = counts.sum(axis=-1, keepdims=True) * BENFORD
    tv = 0.5 * np.abs(counts / counts.sum(axis=-1, keepdims=True) - BENFORD).sum(axis=-1)
    chi_square = ((counts - expected) ** 2 / expected).sum(axis=-1)
    return tv, chi_square


def mean_interval(values, generator):
    """95% percentile bootstrap CI, resampling generation attempts within model."""
    if len(values) < 2:
        return np.nan, np.nan
    indices = generator.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
    means = values[indices].mean(axis=1)
    return np.quantile(means, [ALPHA / 2, 1 - ALPHA / 2])


def summarize(results):
    summary = []
    for index, (folder, _, label, _) in enumerate(MODELS):
        inventory = results[results.model_id == folder]
        valid = inventory[inventory.status == "valid"]
        values = valid.tv.to_numpy(dtype=float)
        low, high = mean_interval(values, rng(2, index))
        summary.append(dict(
            model=label, valid_datasets=len(valid),
            missing_files=int((inventory.status == "missing").sum()),
            invalid_files=int((inventory.status == "invalid").sum()),
            mean_tv=values.mean() if len(values) else np.nan,
            mean_tv_ci_low=low, mean_tv_ci_high=high,
            flagged_datasets=int(valid.flagged.sum()),
            flagged_percent=100 * valid.flagged.mean() if len(valid) else np.nan,
            distinct_amount_collections=valid.collection_sha256.nunique(),
        ))
    return pd.DataFrame(summary)


def duplicate_report(results):
    """Report each kind of repetition within each model; retain every attempt."""
    rows = []
    for kind in HASH_COLUMNS:
        available = results[results[kind] != ""]
        for (model, fingerprint), group in available.groupby(["model_id", kind]):
            if len(group) > 1:
                for filename in group.filename:
                    rows.append([model, kind, fingerprint, filename, len(group)])
    return pd.DataFrame(rows, columns=["model_id", "kind", "hash", "filename", "group_size"])


def save_figure(fig, output, name):
    for extension in ("pdf", "svg", "png"):
        fig.savefig(output / f"{name}.{extension}", dpi=600, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def make_figures(results, summary, output):
    """Two paper figures: mean digit profiles and all dataset-level TV scores."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    valid = results[results.status == "valid"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True, layout="constrained")
    for index, (folder, _, label, color) in enumerate(MODELS):
        ax = axes[index]
        data = valid[valid.model_id == folder]
        ax.plot(DIGITS, BENFORD, "D--", color="#333333", markersize=3, label="Benford")
        if len(data):
            proportions = data[COUNT_COLUMNS].mean().to_numpy() / AMOUNTS_PER_FILE
            ax.plot(DIGITS, proportions, "o-", color=color, markersize=4, label="Model mean")
        else:
            ax.text(0.5, 0.5, "No valid datasets", ha="center", transform=ax.transAxes)
        ax.set(title=f"{chr(97 + index)}) {label}\n(n = {len(data)})",
               xlabel="First significant digit", xticks=DIGITS)
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Mean proportion")
    axes[0].set_ylim(bottom=0)
    axes[-1].legend(frameon=False, fontsize=8)
    save_figure(fig, output, "figure_1_digit_profiles")

    fig, ax = plt.subplots(figsize=(6.6, 3.8), layout="constrained")
    for index, (folder, _, label, color) in enumerate(MODELS):
        values = valid.loc[valid.model_id == folder, "tv"].to_numpy(dtype=float)
        if len(values):
            ax.boxplot([values], positions=[index], widths=0.4, showfliers=False,
                       manage_ticks=False, medianprops={"color": color, "linewidth": 1.5})
            ax.scatter(index + rng(3, index).uniform(-0.15, 0.15, len(values)), values,
                       s=18, color=color, alpha=0.65, zorder=3)
            row = summary.iloc[index]
            ax.plot(index + 0.3, row.mean_tv, "D", color="#222222", markersize=4)
            ax.vlines(index + 0.3, row.mean_tv_ci_low, row.mean_tv_ci_high, color="#222222")
        else:
            ax.text(index, 0.01, "No valid data", ha="center", fontsize=8)
    labels = [f"{model[2]}\n(n = {summary.iloc[i].valid_datasets})" for i, model in enumerate(MODELS)]
    ax.set(xticks=range(3), xticklabels=labels, xlim=(-0.45, 2.6),
           ylabel="Total variation distance", title="First-digit conformity across datasets")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.2)
    save_figure(fig, output, "figure_2_dataset_conformity")


def analyze(data_dir, output):
    """Read inputs, calculate outcomes, and save a new analysis without overwrites."""
    if not any((data_dir / model[0]).is_dir() for model in MODELS):
        raise ValueError("Select the parent folder containing Gemini, ChatGPT, and Claude")
    if output == data_dir or any((data_dir / m[0]) == output or (data_dir / m[0]) in output.parents for m in MODELS):
        raise ValueError("Results must be saved outside the input model folders")
    output.mkdir(parents=True, exist_ok=False)
    reference_counts = rng(1).multinomial(AMOUNTS_PER_FILE, BENFORD, size=SIMULATIONS)
    reference_chi = np.sort(scores(reference_counts)[1])
    records = []
    for folder, prefix, label, _ in MODELS:
        print(f"Reading {folder}: {FILES_PER_MODEL} expected files...", flush=True)
        for number in range(1, FILES_PER_MODEL + 1):
            filename = f"{folder}/{prefix}{number}.csv"
            row = dict(model_id=folder, model=label, filename=filename)
            row.update(read_dataset(data_dir / filename))
            row.update(tv=np.nan, chi_square=np.nan, p_mc=np.nan, flagged=np.nan)
            if row["status"] == "valid":
                tv, chi = scores([row[column] for column in COUNT_COLUMNS])
                exceedances = SIMULATIONS - np.searchsorted(reference_chi, chi, side="left")
                p = (1 + exceedances) / (SIMULATIONS + 1)  # Include ties; never p = 0.
                row.update(tv=float(tv), chi_square=float(chi), p_mc=float(p), flagged=int(p < ALPHA))
            records.append(row)
    results = pd.DataFrame(records)
    summary = summarize(results)
    print("Saving results and two figures...", flush=True)
    for name, table in [("dataset_results", results), ("model_summary", summary),
                        ("duplicate_files", duplicate_report(results))]:
        table.to_csv(output / f"{name}.csv", index=False, float_format="%.12g")
    make_figures(results, summary, output)
    expected = set(results.filename)
    extras = [str(p.relative_to(data_dir)) for m in MODELS for p in sorted((data_dir / m[0]).glob("*.csv"))
              if str(p.relative_to(data_dir)) not in expected]
    metadata = dict(created_utc=datetime.now(timezone.utc).isoformat(), data_dir=str(data_dir),
                    models=MODELS, files_per_model=FILES_PER_MODEL, amounts_per_file=AMOUNTS_PER_FILE,
                    seed=SEED, simulations=SIMULATIONS, bootstraps=BOOTSTRAPS, alpha=ALPHA,
                    python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                    matplotlib=matplotlib.__version__, script_sha256=digest(Path(__file__).read_bytes()),
                    benford_probabilities=BENFORD.tolist(), ignored_csv_files=extras)
    (output / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (output / "figure_captions.txt").write_text(
        f"Figure 1. Mean first-digit proportions across valid generation attempts compared with Benford probabilities. "
        f"Each dataset contains {AMOUNTS_PER_FILE} amounts. This figure is descriptive.\n\n"
        f"Figure 2. Each colored point represents one valid generation attempt, including repeated outputs. "
        f"Boxes show quartiles and medians; whiskers extend to the most extreme observations within 1.5 IQR. "
        f"Black diamonds show means, with {100 * (1 - ALPHA):g}% percentile-bootstrap intervals "
        f"({BOOTSTRAPS:,} resamples of whole datasets). Lower TV indicates closer agreement. "
        f"Intervals assume independent generation attempts.\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"\nSaved to: {output}")
    return results, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, help="New results folder; must not already exist")
    args = parser.parse_args()
    root = args.data_dir.expanduser().resolve()
    destination = args.output or root / "benford_results" / datetime.now(timezone.utc).strftime("concise_%Y%m%dT%H%M%S_%fZ")
    try:
        analyze(root, destination.expanduser().resolve())
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")
