# -*- coding: utf-8 -*-
"""Merge dataset diagnostics with existing experiment metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def _load_dataset_from_metadata(path: Path) -> str:
    metadata_path = path.parent / "metadata.json"
    if not metadata_path.exists():
        return _infer_dataset_from_path(path)
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    names = metadata.get("dataset_names")
    if isinstance(names, list) and names:
        return ",".join(str(item) for item in names)
    args = metadata.get("args", {})
    names = args.get("dataset_names")
    if isinstance(names, list) and names:
        return ",".join(str(item) for item in names)
    dataset = str(args.get("dataset_name", "") or "")
    return dataset or _infer_dataset_from_path(path)


def _infer_dataset_from_path(path: Path) -> str:
    text = str(path)
    if "station05" in text or "hebei_station05" in text:
        return "hebei_station05"
    if (
        "numeric_baseline_best" in text
        or "exp6_text1_best" in text
        or "FRTC_BER_paper_mainline_full_v39_event_switch" in text
    ):
        return "kongzhaopu"
    return ""


def _metric_role(path: Path) -> str:
    parts = path.parts
    joined = "/".join(parts)
    if "text1" in joined or "realtime" in joined:
        return "text1"
    if "FRTC_BER" in joined or "text2" in joined:
        return "text2"
    if "baseline" in joined or "numerical" in joined or "numeric" in joined:
        return "baseline"
    return "unknown"


def collect_metrics(results_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for path in sorted(results_root.glob("**/test_metrics_accepted.csv")):
        try:
            metrics = pd.read_csv(path).iloc[0].to_dict()
        except Exception:
            continue
        rows.append(
            {
                "dataset": _load_dataset_from_metadata(path),
                "role": _metric_role(path),
                "result_dir": str(path.parent),
                "real_MSE": metrics.get("real_MSE", np.nan),
                "real_MAE": metrics.get("real_MAE", np.nan),
                "real_RMSE": metrics.get("real_RMSE", np.nan),
                "real_R2": metrics.get("real_R2", np.nan),
                "stage1_real_MSE": metrics.get("stage1_real_MSE", np.nan),
                "stage1_text_real_MSE": metrics.get("stage1_text_real_MSE", np.nan),
                "numeric_to_text1_gain_MSE": metrics.get("numeric_to_text1_gain_MSE", np.nan),
                "text1_to_final_gain_MSE": metrics.get("text1_to_final_gain_MSE", np.nan),
                "extreme_union_text2_residual_improve_MSE": metrics.get(
                    "extreme_union_text2_residual_improve_MSE", np.nan
                ),
                "normal_daylight_degrade_MSE": metrics.get("normal_daylight_degrade_MSE", np.nan),
                "residual_calibration_variant": metrics.get("residual_calibration_variant", ""),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics", default="./results/dataset_diagnostics.csv")
    parser.add_argument("--results_root", default="./results")
    parser.add_argument("--out", default="./results/result_diagnostics_summary.csv")
    args = parser.parse_args()

    diagnostics = pd.read_csv(args.diagnostics)
    metrics = collect_metrics(Path(args.results_root))
    if metrics.empty:
        print("no metrics found")
        return
    merged = metrics.merge(diagnostics, on="dataset", how="left")
    merged["rmse_over_ot_std"] = merged["real_RMSE"] / merged["ot_std"].replace(0, np.nan)
    merged["rmse_over_daylight_std"] = merged["real_RMSE"] / merged["daylight_ot_std"].replace(0, np.nan)
    merged["mse_over_ot_var"] = merged["real_MSE"] / (merged["ot_std"] ** 2).replace(0, np.nan)
    merged["mse_over_daylight_var"] = merged["real_MSE"] / (merged["daylight_ot_std"] ** 2).replace(0, np.nan)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    display_cols = [
        "dataset",
        "role",
        "real_MSE",
        "real_RMSE",
        "rmse_over_ot_std",
        "rmse_over_daylight_std",
        "real_R2",
        "numeric_to_text1_gain_MSE",
        "text1_to_final_gain_MSE",
        "extreme_union_text2_residual_improve_MSE",
        "normal_daylight_degrade_MSE",
        "text2_high_unique",
        "text_opportunity_score",
        "result_dir",
    ]
    print(merged[display_cols].sort_values(["dataset", "role", "real_MSE"]).to_string(index=False))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
