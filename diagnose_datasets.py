# -*- coding: utf-8 -*-
"""Dataset-level diagnostics for cross-station solar text fusion.

The report is intentionally model-light: it measures target scale, daylight
volatility, ramp frequency, and text-template diversity before expensive
training.  Use it to decide which stations are text-informative and which
stations should rely mostly on the numerical foundation plus safe fallback.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


TARGET = "OT"
DATE_CANDIDATES = ("date", "timestamp", "time", "datetime")
TEXT1_FIELDS = ("state_prompt", "recent_trend_prompt", "statistical_variability_prompt")
TEXT2_FIELDS = ("low_frequency_trend_prompt", "high_frequency_component_prompt")


def _split_names(value: str | None, dataset_root: Path) -> List[str]:
    if value:
        return [item.strip() for item in value.split(",") if item.strip()]
    return sorted(path.name for path in dataset_root.iterdir() if (path / "solar.csv").exists())


def _read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _date_col(df: pd.DataFrame) -> str | None:
    for name in DATE_CANDIDATES:
        if name in df.columns:
            return name
    return None


def _text_stats(rows: List[dict], fields: Iterable[str]) -> Dict[str, float]:
    stats: Dict[str, float] = {"rows": float(len(rows))}
    all_templates = set()
    for field in fields:
        values = [str(row.get(field, "")).strip() for row in rows]
        nonempty = [value for value in values if value]
        unique = set(nonempty)
        all_templates.update((field, value) for value in unique)
        stats[f"{field}_unique"] = float(len(unique))
        stats[f"{field}_nonempty_rate"] = float(len(nonempty) / max(len(values), 1))
    stats["all_field_templates_unique"] = float(len(all_templates))
    stats["unique_per_100_rows"] = float(len(all_templates) / max(len(rows), 1) * 100.0)
    return stats


def _ramp_rate(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 3:
        return 0.0
    diffs = np.abs(np.diff(values))
    threshold = np.nanpercentile(diffs, 90.0)
    return float(np.mean(diffs >= max(float(threshold), 1e-8)))


def diagnose_dataset(dataset_root: Path, name: str) -> dict:
    folder = dataset_root / name
    solar_path = folder / "solar.csv"
    df = pd.read_csv(solar_path)
    if TARGET not in df.columns:
        raise ValueError(f"{solar_path} missing target column {TARGET!r}")
    target = pd.to_numeric(df[TARGET], errors="coerce")
    date_name = _date_col(df)
    if date_name is not None:
        dates = pd.to_datetime(df[date_name], errors="coerce")
        hour = dates.dt.hour + dates.dt.minute / 60.0
        daylight_mask = (hour >= 5.0) & (hour <= 18.0)
    else:
        daylight_mask = target > 0
    positive_mask = target > 1e-8
    daylight_values = target[daylight_mask].to_numpy(dtype=np.float64)
    positive_values = target[positive_mask].to_numpy(dtype=np.float64)
    text1_rows = _read_jsonl(folder / "rt_text1.jsonl")
    text2_rows = _read_jsonl(folder / "rt_text2.jsonl")
    text1 = _text_stats(text1_rows, TEXT1_FIELDS)
    text2 = _text_stats(text2_rows, TEXT2_FIELDS)
    row = {
        "dataset": name,
        "rows": int(len(df)),
        "text1_rows": int(len(text1_rows)),
        "text2_rows": int(len(text2_rows)),
        "ot_mean": float(np.nanmean(target)),
        "ot_std": float(np.nanstd(target)),
        "ot_zero_rate": float(np.mean(np.nan_to_num(target.to_numpy(dtype=np.float64), nan=0.0) <= 1e-8)),
        "daylight_rate": float(np.mean(daylight_mask)),
        "daylight_ot_mean": float(np.nanmean(daylight_values)) if daylight_values.size else 0.0,
        "daylight_ot_std": float(np.nanstd(daylight_values)) if daylight_values.size else 0.0,
        "positive_ot_std": float(np.nanstd(positive_values)) if positive_values.size else 0.0,
        "ramp_rate_all": _ramp_rate(target.to_numpy(dtype=np.float64)),
        "ramp_rate_daylight": _ramp_rate(daylight_values),
        "text1_templates_unique": int(text1["all_field_templates_unique"]),
        "text1_unique_per_100_rows": text1["unique_per_100_rows"],
        "text1_state_unique": int(text1.get("state_prompt_unique", 0.0)),
        "text1_trend_unique": int(text1.get("recent_trend_prompt_unique", 0.0)),
        "text1_variability_unique": int(text1.get("statistical_variability_prompt_unique", 0.0)),
        "text2_templates_unique": int(text2["all_field_templates_unique"]),
        "text2_unique_per_100_rows": text2["unique_per_100_rows"],
        "text2_low_unique": int(text2.get("low_frequency_trend_prompt_unique", 0.0)),
        "text2_high_unique": int(text2.get("high_frequency_component_prompt_unique", 0.0)),
    }
    text_diversity = (
        0.25 * min(row["text1_trend_unique"] / 18.0, 1.0)
        + 0.20 * min(row["text1_variability_unique"] / 4.0, 1.0)
        + 0.25 * min(row["text2_low_unique"] / 12.0, 1.0)
        + 0.30 * min(row["text2_high_unique"] / 5.0, 1.0)
    )
    ramp_strength = min(row["daylight_ot_std"] / max(row["ot_std"], 1e-6), 2.0) / 2.0
    row["text_diversity_score"] = float(text_diversity)
    row["text_opportunity_score"] = float(0.58 * text_diversity + 0.42 * ramp_strength)
    if row["text_opportunity_score"] >= 0.62 and row["text2_high_unique"] >= 4:
        row["text_station_type"] = "A_text_informative"
    elif row["text_opportunity_score"] >= 0.45 and row["text2_high_unique"] >= 2:
        row["text_station_type"] = "B_weak_text"
    else:
        row["text_station_type"] = "C_text_redundant_or_low_diversity"
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="./datasets")
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--out", default="./results/dataset_diagnostics.csv")
    args = parser.parse_args()
    dataset_root = Path(args.dataset_root)
    names = _split_names(args.datasets, dataset_root)
    rows = [diagnose_dataset(dataset_root, name) for name in names]
    report = pd.DataFrame(rows).sort_values(
        ["text_station_type", "text_opportunity_score", "daylight_ot_std"],
        ascending=[True, False, False],
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_path, index=False)
    print(report.to_string(index=False))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
