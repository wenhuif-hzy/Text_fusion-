# -*- coding: utf-8 -*-
"""
FRTC-BER: fuzzy residual text correction with budgeted extreme release.

The script supports a single station or multiple independent station datasets.
It trains a numerical baseline with an internal residual chain:
OT first predicts the horizon, auxiliary meteorology corrects the OT residual,
historical NWP corrects the remaining residual, and future NWP corrects the
final numerical residual.  It then trains the final model that injects realtime
text and forecast-time available Text2 as residual innovations over that
baseline.  The paper-facing execution path is a single mainline method:
numerical foundation -> realtime textual residual intervention -> forecast-time
fuzzy residual release.  Ablations and extra baselines are intentionally left as
explicit modes instead of being part of the default run.

Text is hourly, while OT, auxiliary meteorology, and NWP are sampled every
15 minutes.  The final model is FRTC-BER and uses:
1. Stepwise numerical residual correction over OT, auxiliary meteorology,
   historical NWP, and future NWP.
2. NWP-regime encoding over 15-minute past/future weather fields.
3. Observed numerical response encoding over 15-minute OT and auxiliary
   meteorology.
4. Field-aware realtime text innovation from historical state/trend/variability
   descriptions through regime-conditioned deformable asynchronous
   cross-attention.
5. Forecast-time Text2 decomposition into long-horizon trend information and
   short-term extreme-weather risk information.
6. NWP/response-conditioned entropic optimal transport that aligns future hourly
   text mass to 15-minute forecast horizons under weather, response, semantic,
   and temporal costs.
7. Orthogonal semantic innovation factorization that removes text components
   explainable by NWP and recent numerical response.
   For Text2, this is upgraded to conditioned deconfounding: text
   components predictable from NWP regime, recent PV response, and horizon
   timing are stripped before residual intervention.
8. Forecast-time Text2 is used as residual evidence, not as a forced multimodal
   forecast head.  A budgeted residual-release policy decides how much
   counterfactual Text2 treatment can be released at each horizon; it does not
   add a second residual path.
9. Future extreme-weather semantic prototype memory and fuzzy semantic
   extraction:
   hourly high-risk text is residualized against low-frequency trend text,
   selects learnable weather-risk prototypes, extracts fuzzy memberships for
   ambiguous extreme-weather risk, predicts onset/intensity/duration, and
   integrates shock derivatives into 15-minute residual evidence.
10. Text-guided residual release: calibrated numerical hypotheses expose
    disagreement and opportunity, while text semantics decide whether the Text2
    residual proposal can be safely released.
11. Residual-opportunity triggering: internal disagreement among OT, auxiliary
    meteorology, historical NWP, and future NWP predictions is exposed to the
    text2 corrector, so forecast-time text corrects only plausible residual
    regions.
12. Validation-calibrated residual intervention: after training, validation
    residuals fit fixed horizon-wise weights over text residual evidence
    channels.  Test/inference never uses true future residuals or post-event
    text.
13. Extreme-weather-first evidence release: forecast-time Text2 does not try to
    improve every horizon uniformly.  It is preferentially released on
    daylight high-ramp/high-residual regimes where fuzzy risk semantics,
    high-frequency shock prototypes, realtime text evidence, and NWP
    disagreement point to the same residual direction.
14. Observable extreme-regime mixture calibration: the final residual calibrator
    does not use one global set of weights.  It exposes forecast-time observable
    normal-weather and extreme-weather residual bases, plus a factual-vs-neutral
    Text2 counterfactual risk basis.  Validation learns fixed horizon-wise
    mixture weights; test/inference only sees observable text/NWP/fuzzy risk
    signals.
15. Dataset-style adaptive release: validation learns a fixed text-release scale
    and temporal stability profile for each station or station mix.  Rich
    text/event datasets can release more residual correction, while stations
    with weak or redundant Text2 evidence automatically shrink to the Text1
    forecast instead of being harmed by forced text release.

No random seed is set.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from contextlib import nullcontext
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_interface import (
    SOLAR_AUX_COLS,
    SOLAR_NWP_COLS,
    TARGET,
    build_text1_cache,
    build_text2_cache,
    ensure_dir,
    load_and_merge_numeric,
    make_loaders,
    make_multi_loaders,
    move_batch,
    regression_metrics,
    TEXT2_SCALAR_DIM,
)
from enhanced_itransformer import EnhancedITransformer


CODE_VERSION = "frtc_ber_v54_general_event_sampling"
METHOD_NAME = "FRTC-BER"
METHOD_FULL_NAME = "Fuzzy Residual Text Correction with Budgeted Extreme-aware Release"
METHOD_CORE_CLAIM = (
    "Forecast-time textual evidence should not directly replace the numerical "
    "forecast. It should propose residual interventions and release them under "
    "observable extreme-weather, ramp, NWP-disagreement, and counterfactual "
    "text-treatment evidence."
)
METHOD_PIPELINE = (
    "iTransformer numerical candidate-pool foundation",
    "realtime Text1 asynchronous residual injection",
    "forecast-time Text2 counterfactual residual proposal",
    "fuzzy extreme-weather and ramp semantic evidence",
    "validation-fitted bi-criteria extreme-stable residual release without test-time residual leakage",
)
PAPER_MAINLINE_NAME = "FRTC_BER_ForecastTimeResidualIntervention"
PAPER_MAIN_CONTRIBUTIONS = (
    {
        "name": "Forecast-time textual residual intervention",
        "description": (
            "Text is not used as a direct forecasting feature. Realtime and "
            "forecast-time text produce residual intervention evidence over a "
            "stable numerical baseline."
        ),
    },
    {
        "name": "Asynchronous and deconfounded text-to-horizon alignment",
        "description": (
            "Hourly text is aligned to 15-minute horizons through asynchronous "
            "attention or regime-conditioned transport, and text components "
            "explainable by NWP/recent numerical response are separated from "
            "textual innovation."
        ),
    },
    {
        "name": "Fuzzy extreme-aware Regime-Pareto residual release",
        "description": (
            "Forecast-time Text2 is decomposed into normal trend and extreme "
            "ramp/shock residual bases. An observable regime mixture allocates "
            "residual release capacity between normal-weather stability and "
            "extreme-weather correction."
        ),
    },
)
PAPER_METHOD_FORMULAS = {
    "numerical_foundation": "y_num = f_num(OT, aux_met, past_NWP, future_NWP)",
    "realtime_text1_intervention": "y_text1 = y_num + Delta_rt",
    "text2_counterfactual_evidence": "Delta_treat = Delta_text2(factual) - Delta_text2(neutral)",
    "bicriteria_extreme_stable_release": (
        "Delta_final = m_event * B_event * Delta_event + "
        "(1 - m_event) * B_normal * Delta_normal"
    ),
    "final_forecast": "y_final = y_text1 + Delta_final",
}
PAPER_INFERENCE_CONSTRAINTS = (
    "Text2 must be available at forecast issue time; post-event observed text is not allowed.",
    "Validation residuals are used only to fit fixed release weights/profiles.",
    "Test-time release uses only observable text, NWP disagreement, fuzzy risk, ramp, shock, and coverage signals.",
    "Text residuals are released as bounded interventions over y_text1, never as an unconstrained replacement forecast.",
)
METHOD_CARD = {
    "method_name": METHOD_NAME,
    "method_full_name": METHOD_FULL_NAME,
    "paper_mainline_name": PAPER_MAINLINE_NAME,
    "core_claim": METHOD_CORE_CLAIM,
    "pipeline": list(METHOD_PIPELINE),
    "main_contributions": list(PAPER_MAIN_CONTRIBUTIONS),
    "formulas": PAPER_METHOD_FORMULAS,
    "inference_constraints": list(PAPER_INFERENCE_CONSTRAINTS),
}
PAPER_CORE_METRIC_KEYS = (
    "metric_space",
    "real_MSE",
    "real_MAE",
    "stage1_real_MSE",
    "stage1_real_MAE",
    "stage1_text_real_MSE",
    "stage1_text_real_MAE",
    "numeric_to_text1_gain_MSE",
    "text1_to_final_gain_MSE",
    "numeric_to_final_gain_MSE",
    "daylight_text2_residual_improve_MSE",
    "high_residual_text2_residual_improve_MSE",
    "ramp_event_text2_residual_improve_MSE",
    "extreme_union_text2_residual_improve_MSE",
    "normal_daylight_degrade_MSE",
    "released_residual_coverage",
    "released_residual_abs_scaled",
    "released_residual_rms_scaled",
    "extreme_release_coverage",
    "normal_release_coverage",
    "release_extreme_share",
    "observable_release_budget_mean",
    "observable_release_budget_coverage",
    "observable_release_budget_rms",
    "observable_release_budget_profile",
    "observable_normal_budget_profile",
    "residual_calibration_variant",
    "residual_calibration_name",
    "residual_calibration_cap",
    "residual_calibration_val_event_score",
    "residual_calibration_val_normal_degrade_MSE",
    "residual_calibration_val_normal_improve_MSE",
    "observable_event_mix_mean",
    "observable_event_mix_rms",
    "observable_event_mix_event_mean",
    "observable_event_mix_normal_mean",
)
ACCEPTED_CHECKPOINT_NAME = "best_accepted.pt"
ROUTER_CANDIDATE_NAMES = (
    "stage1_text",
    "future_nwp_chain",
    "nwp_physical_prior",
    "periodic_prior",
)
NUMERICAL_CANDIDATE_NAMES = (
    "numeric_chain_final",
    "ot_only",
    "history",
    "aux_corrected",
    "past_nwp_corrected",
    "future_nwp_chain",
    "nwp_physical_prior",
    "periodic_prior",
)
RESIDUAL_CALIBRATION_FEATURES = (
    "text2_intervention",
    "released_text2_treatment",
    "proposal_release_delta",
    "text2_treatment_effect",
    "treatment_augmented_proposal",
    "counterfactual_gap_delta",
    "treatment_low_delta",
    "treatment_high_delta",
    "event_treatment_delta",
    "trend_treatment_delta",
    "dataset_evidence_delta",
    "dataset_evidence_high_delta",
    "dataset_evidence_low_delta",
    "dataset_evidence_release_gate",
    "text_scalar_direction_delta",
    "text_scalar_risk_delta",
    "realtime_evidence_delta",
    "nwp_disagreement_delta",
    "extreme_weather_evidence_delta",
    "fuzzy_ramp_evidence_delta",
    "shock_direction_delta",
    "semantic_risk_release_gate",
    "semantic_signed_risk_gate",
    "ramp_signed_risk_gate",
    "shock_signed_risk_gate",
    "text_scalar_signed_risk_gate",
    "extreme_counterfactual_delta",
    "extreme_proposal_delta",
    "extreme_dataset_delta",
    "extreme_realtime_delta",
    "extreme_nwp_disagreement_delta",
    "extreme_ramp_prior_delta",
    "event_counterfactual_basis_delta",
    "event_proposal_basis_delta",
    "event_dataset_basis_delta",
    "event_ramp_basis_delta",
    "event_realtime_basis_delta",
    "event_nwp_disagreement_basis_delta",
    "normal_trend_basis_delta",
    "stable_proposal_delta",
    "stable_dataset_delta",
    "observable_extreme_gate",
    "observable_stable_gate",
    "competitive_event_gate",
    "competitive_stable_gate",
    "competitive_event_treatment_delta",
    "competitive_trend_treatment_delta",
    "event_stability_margin_gate",
    "future_daylight_gate",
    "text2_residual_proposal",
    "neutral_text2_proposal",
    "raw_text2_correction",
    "scaled_text2_correction",
    "route_delta",
    "route_prediction_delta",
    "correction",
    "low_delta",
    "high_delta",
    "text_ramp_prior",
    "residual_prior_correction",
    "shock_delta",
    "residual_local",
    "residual_spectral",
)
RESIDUAL_CALIBRATION_SPECS = {
    "single_intervention": ("text2_intervention",),
    "frtc_ber": (
        "route_prediction_delta",
        "route_delta",
        "text2_intervention",
        "proposal_release_delta",
        "text2_residual_proposal",
        "raw_text2_correction",
        "scaled_text2_correction",
        "treatment_augmented_proposal",
        "released_text2_treatment",
        "text2_treatment_effect",
        "counterfactual_gap_delta",
        "treatment_low_delta",
        "treatment_high_delta",
        "low_delta",
        "high_delta",
        "event_treatment_delta",
        "trend_treatment_delta",
        "dataset_evidence_delta",
        "dataset_evidence_high_delta",
        "dataset_evidence_low_delta",
        "dataset_evidence_release_gate",
        "text_scalar_direction_delta",
        "text_scalar_risk_delta",
        "realtime_evidence_delta",
        "nwp_disagreement_delta",
        "event_counterfactual_basis_delta",
        "event_proposal_basis_delta",
        "event_dataset_basis_delta",
        "event_ramp_basis_delta",
        "event_realtime_basis_delta",
        "event_nwp_disagreement_basis_delta",
        "normal_trend_basis_delta",
        "competitive_event_treatment_delta",
        "competitive_trend_treatment_delta",
        "event_stability_margin_gate",
        "text_ramp_prior",
        "residual_prior_correction",
        "shock_delta",
    ),
    "causal_treatment_safe": (
        "route_prediction_delta",
        "route_delta",
        "text2_intervention",
        "proposal_release_delta",
        "text2_residual_proposal",
        "raw_text2_correction",
        "scaled_text2_correction",
        "treatment_augmented_proposal",
        "released_text2_treatment",
        "text2_treatment_effect",
        "counterfactual_gap_delta",
        "treatment_low_delta",
        "treatment_high_delta",
        "low_delta",
        "high_delta",
        "event_treatment_delta",
        "trend_treatment_delta",
        "dataset_evidence_delta",
        "dataset_evidence_high_delta",
        "dataset_evidence_low_delta",
        "dataset_evidence_release_gate",
        "text_scalar_direction_delta",
        "text_scalar_risk_delta",
        "realtime_evidence_delta",
        "nwp_disagreement_delta",
        "event_counterfactual_basis_delta",
        "event_proposal_basis_delta",
        "event_dataset_basis_delta",
        "event_ramp_basis_delta",
        "event_realtime_basis_delta",
        "event_nwp_disagreement_basis_delta",
        "normal_trend_basis_delta",
        "competitive_event_treatment_delta",
        "competitive_trend_treatment_delta",
        "event_stability_margin_gate",
        "text_ramp_prior",
        "residual_prior_correction",
        "shock_delta",
    ),
    "extreme_weather_first": (
        "proposal_release_delta",
        "text2_residual_proposal",
        "released_text2_treatment",
        "treatment_augmented_proposal",
        "counterfactual_gap_delta",
        "treatment_high_delta",
        "event_treatment_delta",
        "competitive_event_treatment_delta",
        "event_stability_margin_gate",
        "dataset_evidence_high_delta",
        "text_scalar_risk_delta",
        "nwp_disagreement_delta",
        "extreme_weather_evidence_delta",
        "fuzzy_ramp_evidence_delta",
        "shock_direction_delta",
        "semantic_risk_release_gate",
        "semantic_signed_risk_gate",
        "ramp_signed_risk_gate",
        "shock_signed_risk_gate",
        "text_scalar_signed_risk_gate",
        "text_ramp_prior",
        "shock_delta",
    ),
    "event_causal_treatment": (
        "proposal_release_delta",
        "text2_residual_proposal",
        "treatment_augmented_proposal",
        "released_text2_treatment",
        "counterfactual_gap_delta",
        "treatment_high_delta",
        "event_treatment_delta",
        "competitive_event_treatment_delta",
        "event_stability_margin_gate",
        "dataset_evidence_delta",
        "dataset_evidence_high_delta",
        "text_scalar_risk_delta",
        "extreme_weather_evidence_delta",
        "fuzzy_ramp_evidence_delta",
        "shock_direction_delta",
        "semantic_signed_risk_gate",
        "ramp_signed_risk_gate",
        "shock_signed_risk_gate",
        "text_scalar_signed_risk_gate",
        "text_ramp_prior",
        "shock_delta",
    ),
    "observable_extreme_moe": (
        "route_prediction_delta",
        "route_delta",
        "text2_intervention",
        "proposal_release_delta",
        "text2_residual_proposal",
        "raw_text2_correction",
        "scaled_text2_correction",
        "released_text2_treatment",
        "text2_treatment_effect",
        "counterfactual_gap_delta",
        "treatment_low_delta",
        "treatment_high_delta",
        "event_treatment_delta",
        "trend_treatment_delta",
        "dataset_evidence_delta",
        "dataset_evidence_high_delta",
        "dataset_evidence_low_delta",
        "text_scalar_direction_delta",
        "text_scalar_risk_delta",
        "realtime_evidence_delta",
        "nwp_disagreement_delta",
        "extreme_counterfactual_delta",
        "extreme_proposal_delta",
        "extreme_dataset_delta",
        "extreme_realtime_delta",
        "extreme_nwp_disagreement_delta",
        "extreme_ramp_prior_delta",
        "event_counterfactual_basis_delta",
        "event_proposal_basis_delta",
        "event_dataset_basis_delta",
        "event_ramp_basis_delta",
        "event_realtime_basis_delta",
        "event_nwp_disagreement_basis_delta",
        "normal_trend_basis_delta",
        "stable_proposal_delta",
        "stable_dataset_delta",
        "text_ramp_prior",
        "residual_prior_correction",
        "shock_delta",
    ),
    "event_adaptive_basis": (
        "route_prediction_delta",
        "route_delta",
        "event_counterfactual_basis_delta",
        "event_proposal_basis_delta",
        "event_dataset_basis_delta",
        "event_ramp_basis_delta",
        "event_realtime_basis_delta",
        "event_nwp_disagreement_basis_delta",
        "normal_trend_basis_delta",
        "text2_residual_proposal",
        "proposal_release_delta",
        "text2_treatment_effect",
        "counterfactual_gap_delta",
        "released_text2_treatment",
        "dataset_evidence_high_delta",
        "text_scalar_risk_delta",
        "nwp_disagreement_delta",
        "text_ramp_prior",
        "shock_delta",
    ),
    "pareto_regime_basis": (
        "route_prediction_delta",
        "route_delta",
        "normal_trend_basis_delta",
        "competitive_trend_treatment_delta",
        "stable_proposal_delta",
        "stable_dataset_delta",
        "event_counterfactual_basis_delta",
        "event_proposal_basis_delta",
        "event_dataset_basis_delta",
        "event_ramp_basis_delta",
        "event_realtime_basis_delta",
        "event_nwp_disagreement_basis_delta",
        "competitive_event_treatment_delta",
        "event_stability_margin_gate",
        "extreme_weather_evidence_delta",
        "fuzzy_ramp_evidence_delta",
        "shock_direction_delta",
        "text2_residual_proposal",
        "proposal_release_delta",
        "text2_treatment_effect",
        "counterfactual_gap_delta",
        "dataset_evidence_delta",
        "dataset_evidence_low_delta",
        "dataset_evidence_high_delta",
        "text_scalar_direction_delta",
        "text_scalar_risk_delta",
        "realtime_evidence_delta",
        "nwp_disagreement_delta",
        "text_ramp_prior",
        "shock_delta",
    ),
    "competitive_treatment_basis": (
        "route_prediction_delta",
        "route_delta",
        "competitive_event_treatment_delta",
        "competitive_trend_treatment_delta",
        "event_stability_margin_gate",
        "competitive_event_gate",
        "competitive_stable_gate",
        "event_counterfactual_basis_delta",
        "event_proposal_basis_delta",
        "event_ramp_basis_delta",
        "event_realtime_basis_delta",
        "event_nwp_disagreement_basis_delta",
        "normal_trend_basis_delta",
        "extreme_weather_evidence_delta",
        "fuzzy_ramp_evidence_delta",
        "shock_direction_delta",
        "text_scalar_risk_delta",
        "nwp_disagreement_delta",
        "text_ramp_prior",
    ),
    "extreme_counterfactual_basis": (
        "route_prediction_delta",
        "route_delta",
        "event_counterfactual_basis_delta",
        "event_proposal_basis_delta",
        "event_ramp_basis_delta",
        "event_realtime_basis_delta",
        "event_nwp_disagreement_basis_delta",
        "extreme_weather_evidence_delta",
        "fuzzy_ramp_evidence_delta",
        "shock_direction_delta",
        "semantic_signed_risk_gate",
        "ramp_signed_risk_gate",
        "shock_signed_risk_gate",
        "text_scalar_signed_risk_gate",
    ),
    "semantic_route_candidate": (
        "route_prediction_delta",
        "route_delta",
        "text2_intervention",
        "text2_residual_proposal",
        "proposal_release_delta",
        "released_text2_treatment",
        "text2_treatment_effect",
        "counterfactual_gap_delta",
        "event_counterfactual_basis_delta",
        "event_proposal_basis_delta",
        "event_dataset_basis_delta",
        "event_ramp_basis_delta",
        "normal_trend_basis_delta",
        "dataset_evidence_delta",
        "text_scalar_risk_delta",
        "nwp_disagreement_delta",
        "text_ramp_prior",
        "shock_delta",
    ),
    "evidential_proposal": (
        "route_prediction_delta",
        "route_delta",
        "text2_intervention",
        "text2_residual_proposal",
        "raw_text2_correction",
        "residual_prior_correction",
        "shock_delta",
    ),
    "spectral_safe": (
        "text2_residual_proposal",
        "low_delta",
        "high_delta",
        "text_ramp_prior",
    ),
    "paper_safe": (
        "text2_intervention",
        "text2_residual_proposal",
        "raw_text2_correction",
        "scaled_text2_correction",
        "low_delta",
        "high_delta",
        "text_ramp_prior",
        "residual_prior_correction",
        "shock_delta",
    ),
    "internal_probe": (
        "route_delta",
        "low_delta",
        "high_delta",
        "text_ramp_prior",
        "residual_prior_correction",
        "shock_delta",
        "residual_local",
        "residual_spectral",
    ),
}


class AttrDict:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _json_scalar(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return value


def _safe_float(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(value):
        return float(default)
    return float(value)


def _atanh_scalar(value: float) -> torch.Tensor:
    value = float(np.clip(value, -0.999, 0.999))
    return torch.tensor(0.5 * np.log((1.0 + value) / (1.0 - value)), dtype=torch.float32)


def _assert_safe_text2_path(path: Optional[str]) -> None:
    """Reject text2 sources that are not forecast-time available."""
    if not path:
        return
    name = os.path.basename(str(path)).lower()
    unsafe_markers = ("residual_oriented", "leaked", "future_observed", "label")
    if any(marker in name for marker in unsafe_markers):
        raise ValueError(
            "unsafe text2 source for paper run: text2 must be forecast-time "
            f"available and must not encode future observed residuals. Got: {path}"
        )


def _split_dataset_names(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    names: List[str] = []
    for chunk in str(value).replace(";", ",").split(","):
        name = chunk.strip()
        if name:
            names.append(name)
    return names


def _dataset_dir(args, dataset_name: str) -> str:
    return os.path.join(str(args.dataset_root), str(dataset_name))


def _dataset_file(args, dataset_name: str, filename: str) -> str:
    return os.path.join(_dataset_dir(args, dataset_name), filename)


def _optional_dataset_file(args, dataset_name: str, filename: str) -> Optional[str]:
    path = _dataset_file(args, dataset_name, filename)
    return path if os.path.exists(path) else None


def _optional_sibling_file(path: Optional[str], filename: str) -> Optional[str]:
    if not path:
        return None
    candidate = os.path.join(os.path.dirname(os.path.abspath(str(path))), filename)
    return candidate if os.path.exists(candidate) else None


def _resolve_dataset_specs(args) -> List[dict]:
    if (
        str(getattr(args, "dataset_root", "./datasets")) == "./datasets"
        and not os.path.isdir("./datasets")
        and os.path.isdir("/home/fengwh/datasets")
    ):
        args.dataset_root = "/home/fengwh/datasets"
    names = _split_dataset_names(getattr(args, "datasets", None))
    if not names:
        names = [str(getattr(args, "dataset_name", "kongzhaopu"))]
    specs: List[dict] = []
    for name in names:
        specs.append(
            {
                "name": name,
                "solar_csv": _dataset_file(args, name, "solar.csv"),
                "context_csv": _optional_dataset_file(args, name, "context_full_day.csv"),
                "solar_aux_csv": None,
                "solar_nwp_csv": None,
                "text1_path": _dataset_file(args, name, "rt_text1.jsonl"),
                "text2_path": _dataset_file(args, name, "rt_text2.jsonl"),
            }
        )
    return specs


def _is_default_flat_path(path: Optional[str], default_name: str) -> bool:
    if path is None:
        return True
    normalized = os.path.normpath(str(path))
    return normalized == os.path.normpath(os.path.join("./data", default_name))


def _infer_single_dataset_name(args, solar_path: Optional[str], fallback: str) -> str:
    if _arg_was_provided("dataset_name"):
        return str(getattr(args, "dataset_name", fallback))
    if solar_path:
        parent = os.path.basename(os.path.dirname(os.path.abspath(str(solar_path))))
        if parent:
            return parent
    return str(fallback)


def _apply_dataset_defaults(args) -> None:
    specs = _resolve_dataset_specs(args)
    args.dataset_specs = specs
    args.dataset_names = [spec["name"] for spec in specs]
    args.multi_dataset = len(specs) > 1
    first = specs[0]
    if args.multi_dataset:
        return
    solar_is_default = _is_default_flat_path(args.solar_csv, "solar.csv")
    if solar_is_default:
        args.solar_csv = first["solar_csv"]
    if _is_default_flat_path(getattr(args, "context_csv", None), "context_full_day.csv"):
        args.context_csv = first.get("context_csv") if solar_is_default else _optional_sibling_file(
            args.solar_csv,
            "context_full_day.csv",
        )
    if _is_default_flat_path(args.text1_path, "rt_text1.jsonl"):
        args.text1_path = first["text1_path"] if solar_is_default else _optional_sibling_file(
            args.solar_csv,
            "rt_text1.jsonl",
        )
    if _is_default_flat_path(args.text2_path, "rt_text2.jsonl"):
        args.text2_path = first["text2_path"] if solar_is_default else _optional_sibling_file(
            args.solar_csv,
            "rt_text2.jsonl",
        )
    single_name = _infer_single_dataset_name(args, args.solar_csv, str(first["name"]))
    args.dataset_name = single_name
    args.dataset_names = [single_name]
    args.dataset_specs = [
        {
            "name": single_name,
            "solar_csv": args.solar_csv,
            "context_csv": args.context_csv,
            "solar_aux_csv": args.solar_aux_csv,
            "solar_nwp_csv": args.solar_nwp_csv,
            "text1_path": args.text1_path,
            "text2_path": args.text2_path,
        }
    ]


def _arg_was_provided(name: str) -> bool:
    flag = f"--{name}"
    no_flag = f"--no-{name}"
    return any(item == flag or item == no_flag for item in sys.argv[1:])


def _apply_numeric_foundation_profile(args) -> None:
    """Select a robust numerical foundation before adding text residuals.

    The compact profile preserves the conservative residual chain.  The robust
    profile adds history, NWP-physical, and periodic candidates and opens their
    release gates.  Auto now uses the same robust profile for every dataset, so
    the numerical foundation is not selected by station name.
    """
    profile = str(getattr(args, "numeric_foundation_profile", "auto") or "auto").lower()
    if profile not in {"auto", "compact", "robust"}:
        raise ValueError(
            "--numeric_foundation_profile must be one of: auto, compact, robust"
        )
    if profile == "auto":
        profile = "robust"
    args.numeric_foundation_profile_effective = profile

    if profile != "robust":
        return

    robust_bool_defaults = {
        "use_history_backbone": True,
        "use_nwp_prior": True,
        "use_periodic_prior": True,
        "use_aux_residual": True,
        "use_past_nwp_residual": True,
        "use_future_nwp_residual": True,
    }
    for key, value in robust_bool_defaults.items():
        if not _arg_was_provided(key):
            setattr(args, key, value)

    robust_float_defaults = {
        "aux_gate_bias": -2.2,
        "past_nwp_gate_bias": -3.0,
        "future_nwp_gate_bias": -3.0,
        "aux_gate_max": 0.80,
        "past_nwp_gate_max": 0.70,
        "future_nwp_gate_max": 0.70,
        "w_history_prior": 0.10,
        "w_history_trust": 0.015,
        "w_nwp_prior": 0.28,
        "w_nwp_trust": 0.025,
        "w_periodic_prior": 0.20,
        "w_periodic_trust": 0.025,
        "w_branch": 0.12,
        "w_residual": 0.20,
    }
    for key, value in robust_float_defaults.items():
        if not _arg_was_provided(key):
            setattr(args, key, value)
    robust_general_defaults = {
        "use_dataset_conditioning": bool(getattr(args, "multi_dataset", False)),
        "balanced_dataset_sampling": True,
        "event_balanced_sampling": True,
        "w_numeric_event": 0.58,
        "w_non_degradation": 3.8,
        "w_candidate_trust_sparse": 0.03,
        "ema_decay": 0.985,
    }
    for key, value in robust_general_defaults.items():
        if not _arg_was_provided(key):
            setattr(args, key, value)
    if bool(getattr(args, "multi_dataset", False)):
        if not _arg_was_provided("epochs"):
            args.epochs = max(int(getattr(args, "epochs", 40)), 50)
        if not _arg_was_provided("patience"):
            args.patience = max(int(getattr(args, "patience", 8)), 10)


def _build_text1_caches(args, device: torch.device) -> Dict[str, dict]:
    caches: Dict[str, dict] = {}
    specs = getattr(args, "dataset_specs", None) or _resolve_dataset_specs(args)
    for spec in specs:
        name = str(spec["name"])
        caches[name] = build_text1_cache(
            str(spec["text1_path"]),
            args.text_model,
            str(device),
            args.cache_dir,
            batch_size=args.text_batch_size,
            max_tokens=args.text_max_tokens,
        )
    return caches


def _build_text2_caches(args, device: torch.device) -> Dict[str, dict]:
    caches: Dict[str, dict] = {}
    specs = getattr(args, "dataset_specs", None) or _resolve_dataset_specs(args)
    for spec in specs:
        name = str(spec["name"])
        _assert_safe_text2_path(str(spec["text2_path"]))
        caches[name] = build_text2_cache(
            str(spec["text2_path"]),
            args.text_model,
            str(device),
            args.cache_dir,
            batch_size=args.text_batch_size,
        )
    return caches


def _first_cache(caches: Dict[str, dict]) -> dict:
    if not caches:
        raise ValueError("cache dictionary is empty")
    return next(iter(caches.values()))


def _single_or_multi_path_metadata(args) -> dict:
    specs = getattr(args, "dataset_specs", None) or _resolve_dataset_specs(args)
    if len(specs) == 1:
        return {
            "dataset_names": [specs[0]["name"]],
            "text1_path": specs[0]["text1_path"],
            "text2_path": specs[0]["text2_path"],
        }
    return {
        "dataset_names": [spec["name"] for spec in specs],
        "text1_path": {spec["name"]: spec["text1_path"] for spec in specs},
        "text2_path": {spec["name"]: spec["text2_path"] for spec in specs},
    }


def _current_dataset_signature(args) -> Tuple[str, ...]:
    return tuple(str(name) for name in getattr(args, "dataset_names", []) or [])


def _metadata_value(metadata: dict, saved_args: dict, key: str, default=None):
    if key in metadata:
        return metadata.get(key)
    return saved_args.get(key, default)


def _float_matches(saved, current, tol: float = 1e-6) -> bool:
    if saved is None or current is None:
        return saved is None and current is None
    try:
        return abs(float(saved) - float(current)) <= tol
    except (TypeError, ValueError):
        return False


def _checkpoint_prediction_protocol_matches(metadata: dict, saved_args: dict, args) -> bool:
    for key in ("seq_len", "pred_len"):
        saved = _metadata_value(metadata, saved_args, key)
        current = getattr(args, key, None)
        if saved is None:
            if key == "pred_len" and int(current) != 96:
                return False
            continue
        try:
            if int(saved) != int(current):
                return False
        except (TypeError, ValueError):
            return False

    saved_window = metadata.get("prediction_window", {}) if isinstance(metadata, dict) else {}
    saved_start = saved_window.get("start_hour", saved_args.get("prediction_start_hour"))
    saved_end = saved_window.get("end_hour", saved_args.get("prediction_end_hour"))
    saved_include = saved_window.get("include_end", saved_args.get("prediction_include_end"))
    current_start = getattr(args, "prediction_start_hour", None)
    current_end = getattr(args, "prediction_end_hour", None)
    current_include = bool(getattr(args, "prediction_include_end", False))
    if current_start is not None or current_end is not None:
        if saved_start is None or saved_end is None:
            return False
        if not _float_matches(saved_start, current_start):
            return False
        if not _float_matches(saved_end, current_end):
            return False
        if bool(saved_include) != current_include:
            return False

    sample_minutes = _metadata_value(metadata, saved_args, "sample_minutes", 15)
    try:
        return int(sample_minutes) == 15
    except (TypeError, ValueError):
        return False


def _normalize_source_path(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    if not text:
        return None
    return os.path.abspath(os.path.expanduser(text))


def _current_source_signature(args) -> Tuple[object, object]:
    specs = getattr(args, "dataset_specs", None) or _resolve_dataset_specs(args)
    names = [str(spec.get("name")) for spec in specs]
    context = {
        name: _normalize_source_path(spec.get("context_csv") or spec.get("solar_csv"))
        for name, spec in zip(names, specs)
    }
    target = {
        name: _normalize_source_path(spec.get("solar_csv"))
        for name, spec in zip(names, specs)
    }
    if len(names) == 1:
        name = names[0]
        return context[name], target[name]
    return context, target


def _normalize_saved_source(value):
    if isinstance(value, dict):
        return {str(key): _normalize_source_path(item) for key, item in value.items()}
    return _normalize_source_path(value)


def _checkpoint_sources_match(metadata: dict, args) -> bool:
    saved_context = metadata.get("context_source")
    saved_target = metadata.get("target_source")
    if saved_context is None or saved_target is None:
        return False
    current_context, current_target = _current_source_signature(args)
    return (
        _normalize_saved_source(saved_context) == current_context
        and _normalize_saved_source(saved_target) == current_target
    )


def _checkpoint_matches_current_data(path: str, args) -> bool:
    if not os.path.exists(path):
        return False
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    metadata = checkpoint.get("metadata", {}) if isinstance(checkpoint, dict) else {}
    saved_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    if not _checkpoint_prediction_protocol_matches(metadata, saved_args, args):
        return False
    if not _checkpoint_sources_match(metadata, args):
        return False
    saved = tuple(str(name) for name in metadata.get("dataset_names", []) or [])
    current = _current_dataset_signature(args)
    if saved and current:
        return saved == current
    saved_datasets = saved_args.get("datasets")
    if saved_datasets:
        return tuple(_split_dataset_names(saved_datasets)) == current
    saved_name = saved_args.get("dataset_name")
    if saved_name:
        return (str(saved_name),) == current
    return not bool(getattr(args, "multi_dataset", False))


def _require_checkpoint_matches_current_data(path: Optional[str], args, label: str) -> None:
    if not path:
        return
    if not _checkpoint_matches_current_data(path, args):
        raise ValueError(
            f"{label} checkpoint is incompatible with the current dataset or "
            f"prediction protocol: {path}"
        )


def _json_safe(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _args_metadata(args) -> dict:
    return {
        str(key): _json_safe(value)
        for key, value in vars(args).items()
    }


def _daylight_flag(hour: int) -> bool:
    return 5 <= int(hour) <= 18


def _trend_word(delta: float, abs_threshold: float = 0.15) -> str:
    if delta >= abs_threshold:
        return "increase"
    if delta <= -abs_threshold:
        return "decrease"
    return "remain stable"


def _risk_word(short_delta: float, long_delta: float, abs_threshold: float = 0.25) -> str:
    if abs(short_delta) >= abs_threshold and abs(short_delta) >= abs(long_delta):
        return "sharp ramp"
    if abs(long_delta) >= abs_threshold:
        return "sustained drift"
    return "weak fluctuation"


def _confidence_token(confidence: str) -> str:
    confidence = str(confidence).strip().lower()
    if confidence == "high":
        return "high-confidence"
    if confidence == "medium":
        return "medium-confidence"
    return "low-confidence"


def _read_time_indexed_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError(f"{path} is missing date column")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    return df


def _solar_residual_prompt_rows(
    df: pd.DataFrame,
) -> List[dict]:
    rows: List[dict] = []
    if df.empty:
        return rows
    df = df.copy()
    df = df[df["date"].dt.minute == 0].reset_index(drop=True)
    df["hour"] = df["date"].dt.hour
    n = len(df)
    for idx, row in df.iterrows():
        date = row["date"]
        hour = int(row["hour"])
        daylight = _daylight_flag(hour)
        nwp_sw = _safe_float(row.get("nwp_shortwaveirrad", 0.0))
        nwp_scatter = _safe_float(row.get("nwp_scatterirrad", 0.0))
        nwp_direct = _safe_float(row.get("nwp_directirrad", 0.0))
        nwp_tmp = _safe_float(row.get("nwp_tmp", 0.0))
        nwp_rh = _safe_float(row.get("nwp_rh", 0.0))
        ot = _safe_float(row.get("OT", 0.0))
        temp = _safe_float(row.get("temp", 0.0))
        pressure = _safe_float(row.get("pressure", 0.0))
        wind = _safe_float(row.get("wind_speed", 0.0))
        sun = _safe_float(row.get("sun_radiation", 0.0))
        sca = _safe_float(row.get("sca_radiation", 0.0))

        past_1h_idx = max(idx - 1, 0)
        past_4h_idx = max(idx - 4, 0)
        future_1h_idx = min(idx + 1, n - 1)
        future_4h_idx = min(idx + 4, n - 1)

        ot_past_1h = _safe_float(df.iloc[past_1h_idx]["OT"], ot)
        ot_past_4h = _safe_float(df.iloc[past_4h_idx]["OT"], ot)
        sw_now = nwp_sw
        sw_1h = _safe_float(df.iloc[future_1h_idx]["nwp_shortwaveirrad"], sw_now)
        sw_4h = _safe_float(df.iloc[future_4h_idx]["nwp_shortwaveirrad"], sw_now)
        direct_1h = _safe_float(df.iloc[future_1h_idx]["nwp_directirrad"], nwp_direct)
        scatter_1h = _safe_float(df.iloc[future_1h_idx]["nwp_scatterirrad"], nwp_scatter)

        short_delta = sw_1h - sw_now
        long_delta = sw_4h - sw_now
        past_short_delta = ot - ot_past_1h
        past_long_delta = ot - ot_past_4h
        nwp_consistency = abs(short_delta) + 0.35 * abs(long_delta)
        drift_word = _trend_word(long_delta, abs_threshold=45.0)
        risk_word = _risk_word(short_delta, long_delta, abs_threshold=65.0)

        risky = abs(short_delta) >= 100.0 or abs(long_delta) >= 180.0
        if daylight:
            if sw_now <= 5.0 and sw_1h <= 5.0:
                regime = "nighttime background"
            elif short_delta <= -120.0 and direct_1h <= nwp_direct * 0.9:
                regime = "cloud-shadow weakening"
            elif short_delta >= 120.0 and direct_1h >= nwp_direct * 1.05:
                regime = "rapid irradiance rise"
            else:
                regime = "daytime radiation transition"
        else:
            regime = "nighttime background"

        if past_short_delta < -0.15 and short_delta < -40.0:
            baseline_bias = "overestimate"
        elif past_short_delta > 0.15 and short_delta > 40.0:
            baseline_bias = "underestimate"
        else:
            baseline_bias = "track closely"
        confidence = "high" if nwp_consistency >= 180.0 else "medium" if nwp_consistency >= 80.0 else "low"
        confidence_token = _confidence_token(confidence)
        if abs(short_delta) >= 120.0 and abs(past_short_delta) >= 0.20:
            direction = "downward correction"
        elif abs(short_delta) <= 40.0 and abs(past_short_delta) <= 0.10:
            direction = "minimal correction"
        else:
            direction = "moderate correction"
        if risky and short_delta <= -100.0:
            short_risk = "high-frequency drop risk"
        elif risky and short_delta >= 100.0:
            short_risk = "high-frequency rise risk"
        else:
            short_risk = "short-term ambiguity"

        low_prompt = (
            f"Residual-oriented background: {regime}. "
            f"The numerical baseline may {baseline_bias} the PV level over the next hour. "
            f"Low-frequency NWP trend is expected to {drift_word}, with radiation regime driven by shortwave {nwp_sw:.1f}, "
            f"direct {nwp_direct:.1f}, diffuse {nwp_scatter:.1f}. "
            f"Humidity {nwp_rh:.1f}, temperature {nwp_tmp:.1f}, wind {wind:.1f}, observed sun radiation {sun:.1f}, scatter radiation {sca:.1f}. "
            f"Expected correction direction is {direction}."
        )
        if risky:
            high_prompt = (
                f"Fuzzy risk view: {short_risk}. "
                f"The coming horizon contains {risk_word} and may show cloud-edge timing uncertainty or a brief reversal. "
                f"Short-term NWP change over 1h is {short_delta:.1f}, over 4h is {long_delta:.1f}; "
                f"this suggests a {confidence_token} residual opportunity. "
                f"Use the text as a semantic routing cue for selecting numerical hypotheses and only then as a bounded residual cue."
            )
        else:
            high_prompt = (
                f"Stable residual view: the horizon appears near a weak fluctuation regime. "
                f"Short-term NWP change over 1h is {short_delta:.1f}, over 4h is {long_delta:.1f}; "
                f"this suggests a {confidence_token} minimal correction. "
                f"Use the text mainly to keep the reliable numerical hypothesis rather than to force residual change."
            )
        rows.append(
            {
                "timestamp": date.strftime("%Y-%m-%d %H:%M:%S"),
                "low_frequency_trend_prompt": low_prompt,
                "high_frequency_component_prompt": high_prompt,
            }
        )
    return rows


def build_residual_oriented_text2(args):
    specs = getattr(args, "dataset_specs", None) or _resolve_dataset_specs(args)
    source_csv = getattr(args, "solar_csv", None) or specs[0]["solar_csv"]
    out_path = getattr(args, "output_path", None) or os.path.join(
        str(args.dataset_root), str(specs[0]["name"]), "rt_text2_residual_oriented.jsonl"
    )
    df = _read_time_indexed_csv(source_csv)
    if getattr(args, "build_neutral_text2", False):
        hourly = df[df["date"].dt.minute == 0].reset_index(drop=True)
        rows = [
            {
                "timestamp": value.strftime("%Y-%m-%d %H:%M:%S"),
                "low_frequency_trend_prompt": "Neutral future background: no reliable residual direction is provided.",
                "high_frequency_component_prompt": "Neutral future risk view: no reliable high-frequency residual evidence or expert-routing cue is provided.",
            }
            for value in hourly["date"]
        ]
    else:
        rows = _solar_residual_prompt_rows(df)
    ensure_dir(os.path.dirname(out_path) or ".")
    with open(out_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    kind = "neutral" if getattr(args, "build_neutral_text2", False) else "residual-oriented"
    print(f"wrote {len(rows)} {kind} rows to {out_path}")
    return {"output_path": out_path, "rows": len(rows)}


class DeepProjection(nn.Module):
    def __init__(self, input_dim: int, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def daylight_weight(future_time_features: torch.Tensor, strength: float) -> torch.Tensor:
    if strength <= 1.0:
        return torch.ones_like(future_time_features[..., 0])
    sin_hour = future_time_features[..., 0]
    cos_hour = future_time_features[..., 1]
    hour = torch.atan2(sin_hour, cos_hour) * 24.0 / (2.0 * torch.pi)
    hour = torch.where(hour < 0.0, hour + 24.0, hour)
    daylight = torch.sigmoid((hour - 5.5) / 0.7) * torch.sigmoid((19.0 - hour) / 0.7)
    return 1.0 + (strength - 1.0) * daylight


def forecasting_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    future_time_features: torch.Tensor,
    daytime_weight: float,
) -> torch.Tensor:
    weight = daylight_weight(future_time_features, daytime_weight)
    mse = torch.mean(weight * (prediction - target).square())
    huber = torch.mean(
        weight * F.smooth_l1_loss(prediction, target, reduction="none", beta=0.5)
    )
    return 0.8 * mse + 0.2 * huber


def target_event_focus(target: torch.Tensor, future_time_features: torch.Tensor) -> torch.Tensor:
    daylight = (daylight_weight(future_time_features, 2.0) - 1.0).clamp(0.0, 1.0)
    level_abs = target.abs()
    level_scale = level_abs.mean(dim=1, keepdim=True).detach().clamp_min(1e-3)
    level_focus = torch.sigmoid((level_abs - 1.10 * level_scale) / 0.12)
    ramp = torch.zeros_like(target)
    if target.shape[1] > 1:
        ramp[:, 1:] = target[:, 1:] - target[:, :-1]
    ramp_abs = ramp.abs()
    scale = ramp_abs.mean(dim=1, keepdim=True).detach().clamp_min(1e-3)
    ramp_focus = torch.sigmoid((ramp_abs - 0.70 * scale) / 0.05)
    curvature = torch.zeros_like(target)
    if target.shape[1] > 2:
        curvature[:, 2:] = target[:, 2:] - 2.0 * target[:, 1:-1] + target[:, :-2]
    curvature_abs = curvature.abs()
    curvature_scale = curvature_abs.mean(dim=1, keepdim=True).detach().clamp_min(1e-3)
    curvature_focus = torch.sigmoid((curvature_abs - 0.75 * curvature_scale) / 0.05)
    focus = torch.maximum(
        0.45 * level_focus + 0.40 * ramp_focus + 0.15 * curvature_focus,
        ramp_focus,
    )
    return (focus * (0.20 + 0.80 * daylight)).detach()


def target_ramp_focus(target: torch.Tensor, future_time_features: torch.Tensor) -> torch.Tensor:
    return target_event_focus(target, future_time_features)


def event_weighted_forecasting_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    future_time_features: torch.Tensor,
    daytime_weight: float,
    event_strength: float,
) -> torch.Tensor:
    base = forecasting_loss(prediction, target, future_time_features, daytime_weight)
    if event_strength <= 0.0:
        return base
    focus = target_event_focus(target, future_time_features)
    weight = 1.0 + float(event_strength) * focus
    mse = torch.mean(weight * (prediction - target).square())
    ramp_pred = torch.zeros_like(prediction)
    ramp_target = torch.zeros_like(target)
    if prediction.shape[1] > 1:
        ramp_pred[:, 1:] = prediction[:, 1:] - prediction[:, :-1]
        ramp_target[:, 1:] = target[:, 1:] - target[:, :-1]
    ramp_loss = torch.mean(weight * F.smooth_l1_loss(ramp_pred, ramp_target, reduction="none", beta=0.25))
    return 0.70 * base + 0.24 * mse + 0.06 * ramp_loss


def safe_probability_bce(probability: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = probability.float().clamp(1e-5, 1.0 - 1e-5)
    target = target.float()
    return -(
        target * torch.log(probability)
        + (1.0 - target) * torch.log1p(-probability)
    ).mean()


def _time_features(offset: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            offset / 24.0,
            torch.sin(2.0 * torch.pi * offset / 24.0),
            torch.cos(2.0 * torch.pi * offset / 24.0),
            torch.sign(offset) * torch.log1p(offset.abs()),
            torch.exp(-offset.abs() / 6.0),
        ],
        dim=-1,
    )


class MaskedAttentivePool(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, max(d_model // 2, 1)),
            nn.Tanh(),
            nn.Linear(max(d_model // 2, 1), 1, bias=False),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.score(x).squeeze(-1)
        safe = mask.any(dim=1, keepdim=True)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        logits = torch.where(safe, logits, torch.zeros_like(logits))
        weights = torch.softmax(logits, dim=1)
        weights = torch.where(mask, weights, torch.zeros_like(weights))
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        pooled = torch.sum(weights.unsqueeze(-1) * x, dim=1)
        return torch.where(safe, pooled, torch.zeros_like(pooled))


class SmoothDCTBands(nn.Module):
    def __init__(self, horizon: int, cutoff_index: float = 8.0, sharpness: float = 4.0):
        super().__init__()
        n = torch.arange(horizon, dtype=torch.float32)
        k = torch.arange(horizon, dtype=torch.float32).unsqueeze(1)
        basis = torch.cos(torch.pi / horizon * (n + 0.5) * k)
        basis[0] *= 1.0 / np.sqrt(horizon)
        if horizon > 1:
            basis[1:] *= np.sqrt(2.0 / horizon)
        frequency = torch.arange(horizon, dtype=torch.float32)
        low_weight = torch.exp(-((frequency / max(cutoff_index, 1e-3)) ** sharpness))
        high_weight = 1.0 - low_weight
        self.register_buffer("basis", basis, persistent=True)
        self.register_buffer("low_weight", low_weight, persistent=True)
        self.register_buffer("high_weight", high_weight, persistent=True)

    def project(self, signal: torch.Tensor, band: str) -> torch.Tensor:
        coeff = signal @ self.basis.t()
        weight = self.low_weight if band == "low" else self.high_weight
        return (coeff * weight) @ self.basis

    def split(self, signal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.project(signal, "low"), self.project(signal, "high")


class NWPRegimeEncoder(nn.Module):
    """Encodes 15-minute NWP into horizon-wise weather regimes."""

    def __init__(
        self,
        nwp_dim: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        layers: int,
        dropout: float,
    ):
        super().__init__()
        self.value_projection = DeepProjection(max(nwp_dim, 1), d_model, d_ff, dropout)
        self.time_projection = nn.Sequential(
            nn.Linear(5, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.segment_embedding = nn.Parameter(torch.empty(1, 2, d_model))
        nn.init.trunc_normal_(self.segment_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.pool = MaskedAttentivePool(d_model)
        self.horizon_fusion = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        horizon_state: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        past = self.value_projection(batch["past_nwp"])
        future = self.value_projection(batch["future_nwp"])
        past = past + self.time_projection(_time_features(batch["past_offset_hours"]))
        future = future + self.time_projection(_time_features(batch["future_offset_hours"]))
        past = past + self.segment_embedding[:, :1]
        future = future + self.segment_embedding[:, 1:2]
        memory = torch.cat([past, future], dim=1)
        mask = torch.cat([batch["past_nwp_mask"], batch["future_nwp_mask"]], dim=1).bool()
        safe_mask = mask.clone()
        empty = ~safe_mask.any(dim=1)
        if empty.any():
            safe_mask[empty, 0] = True
        memory = self.encoder(memory, src_key_padding_mask=~safe_mask)
        memory = self.norm(memory).masked_fill(~mask.unsqueeze(-1), 0.0)
        past_state = self.pool(memory[:, : past.shape[1]], batch["past_nwp_mask"].bool())
        future_state = memory[:, past.shape[1] :]
        future_mask = batch["future_nwp_mask"].bool()
        future_global = self.pool(future_state, future_mask)
        past_expand = past_state.unsqueeze(1).expand_as(horizon_state)
        horizon_regime = self.horizon_fusion(
            torch.cat([horizon_state, future_state, past_expand], dim=-1)
        )
        horizon_regime = torch.where(
            future_mask.unsqueeze(-1), horizon_regime, torch.zeros_like(horizon_regime)
        )
        global_regime = 0.5 * (future_global + past_state)
        return horizon_regime, global_regime


class ObservedNumericalResponseEncoder(nn.Module):
    """Encodes 15-minute OT and auxiliary meteorology into horizon response states."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        layers: int,
        dropout: float,
    ):
        super().__init__()
        self.value_projection = DeepProjection(max(input_dim, 1), d_model, d_ff, dropout)
        self.history_time_projection = nn.Sequential(
            nn.Linear(5, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.horizon_time_projection = nn.Sequential(
            nn.Linear(5, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.trend_norm = nn.LayerNorm(d_model)
        self.output = nn.Sequential(
            nn.LayerNorm(d_model * 4),
            nn.Linear(d_model * 4, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.LayerNorm(d_model),
        )

    @staticmethod
    def _masked_weights(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        safe = mask.any(dim=1, keepdim=True)
        masked = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        masked = torch.where(safe, masked, torch.zeros_like(masked))
        weights = torch.softmax(masked, dim=1)
        weights = torch.where(mask, weights, torch.zeros_like(weights))
        return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

    @staticmethod
    def _pool(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.sum(weights.unsqueeze(-1) * x, dim=1)

    def forward(
        self,
        horizon_state: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = torch.ones(
            batch["x"].shape[:2],
            dtype=torch.bool,
            device=batch["x"].device,
        )
        history = self.value_projection(batch["x"])
        history = history + self.history_time_projection(
            _time_features(batch["past_offset_hours"])
        )
        history = self.encoder(history, src_key_padding_mask=~mask)
        history = self.norm(history)

        recent_logits = -torch.relu(-batch["past_offset_hours"]) / 6.0
        slow_logits = -(batch["past_offset_hours"] + 12.0).abs() / 12.0
        recent = self._pool(history, self._masked_weights(recent_logits, mask))
        slow = self._pool(history, self._masked_weights(slow_logits, mask))
        trend = self.trend_norm(recent - slow)
        global_state = 0.5 * (recent + slow)

        horizon_time = self.horizon_time_projection(
            _time_features(batch["future_offset_hours"])
        )
        recent = recent.unsqueeze(1).expand(-1, horizon_state.shape[1], -1)
        trend = trend.unsqueeze(1).expand(-1, horizon_state.shape[1], -1)
        response = self.output(
            torch.cat([horizon_state, recent, trend, horizon_time], dim=-1)
        )
        return response, global_state


class HistoricalSignalEncoder(nn.Module):
    """Encodes one historical numerical source for residual correction."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        layers: int,
        dropout: float,
    ):
        super().__init__()
        self.value_projection = DeepProjection(max(input_dim, 1), d_model, d_ff, dropout)
        self.history_time_projection = nn.Sequential(
            nn.Linear(5, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.horizon_time_projection = nn.Sequential(
            nn.Linear(5, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.trend_norm = nn.LayerNorm(d_model)
        self.output = nn.Sequential(
            nn.LayerNorm(d_model * 4),
            nn.Linear(d_model * 4, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.LayerNorm(d_model),
        )

    @staticmethod
    def _masked_weights(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        safe = mask.any(dim=1, keepdim=True)
        masked = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        masked = torch.where(safe, masked, torch.zeros_like(masked))
        weights = torch.softmax(masked, dim=1)
        weights = torch.where(mask, weights, torch.zeros_like(weights))
        return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

    @staticmethod
    def _pool(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.sum(weights.unsqueeze(-1) * x, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        past_offset_hours: torch.Tensor,
        future_offset_hours: torch.Tensor,
        horizon_state: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        encoded = self.value_projection(x)
        encoded = encoded + self.history_time_projection(_time_features(past_offset_hours))
        safe_mask = mask.bool().clone()
        empty = ~safe_mask.any(dim=1)
        if empty.any():
            safe_mask[empty, 0] = True
        encoded = self.encoder(encoded, src_key_padding_mask=~safe_mask)
        encoded = self.norm(encoded).masked_fill(~mask.unsqueeze(-1), 0.0)

        recent_logits = -torch.relu(-past_offset_hours) / 6.0
        slow_logits = -(past_offset_hours + 12.0).abs() / 12.0
        recent = self._pool(encoded, self._masked_weights(recent_logits, mask))
        slow = self._pool(encoded, self._masked_weights(slow_logits, mask))
        trend = self.trend_norm(recent - slow)
        global_state = 0.5 * (recent + slow)

        horizon_time = self.horizon_time_projection(_time_features(future_offset_hours))
        context = self.output(
            torch.cat(
                [
                    horizon_state,
                    recent.unsqueeze(1).expand_as(horizon_state),
                    trend.unsqueeze(1).expand_as(horizon_state),
                    horizon_time,
                ],
                dim=-1,
            )
        )
        return context, global_state


class FutureSignalEncoder(nn.Module):
    """Encodes future NWP for horizon-specific residual correction."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        layers: int,
        dropout: float,
    ):
        super().__init__()
        self.value_projection = DeepProjection(max(input_dim, 1), d_model, d_ff, dropout)
        self.time_projection = nn.Sequential(
            nn.Linear(5, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.pool = MaskedAttentivePool(d_model)
        self.output = nn.Sequential(
            nn.LayerNorm(d_model * 4),
            nn.Linear(d_model * 4, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        future_offset_hours: torch.Tensor,
        horizon_state: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.value_projection(x)
        horizon_time = self.time_projection(_time_features(future_offset_hours))
        encoded = encoded + horizon_time
        safe_mask = mask.bool().clone()
        empty = ~safe_mask.any(dim=1)
        if empty.any():
            safe_mask[empty, 0] = True
        encoded = self.encoder(encoded, src_key_padding_mask=~safe_mask)
        encoded = self.norm(encoded).masked_fill(~mask.unsqueeze(-1), 0.0)
        global_state = self.pool(encoded, mask.bool())
        context = self.output(
            torch.cat(
                [
                    horizon_state,
                    encoded,
                    global_state.unsqueeze(1).expand_as(horizon_state),
                    horizon_time,
                ],
                dim=-1,
            )
        )
        context = torch.where(mask.unsqueeze(-1), context, torch.zeros_like(context))
        return context, global_state


class ResidualDeltaHead(nn.Module):
    """Horizon-wise bounded residual head with exact-zero initialization."""

    def __init__(self, d_model: int, d_ff: int, dropout: float, max_delta: float):
        super().__init__()
        self.max_delta = float(max_delta)
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_ff // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff // 2, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        raw = self.net(context).squeeze(-1)
        if self.max_delta > 0:
            return self.max_delta * torch.tanh(raw / self.max_delta)
        return raw


class SafeResidualGate(nn.Module):
    """Conservative opportunity gate for residual corrections.

    A new numerical source should start as an almost-no-op and only change the
    forecast where its residual candidate is useful.  The gate sees the current
    horizon state, the source context, the predecessor prediction, and the
    proposed residual magnitude.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float,
        gate_bias: float = -3.0,
        max_gate: float = 0.75,
    ):
        super().__init__()
        self.max_gate = float(max_gate)
        self.net = nn.Sequential(
            nn.LayerNorm(d_model * 2 + 3),
            nn.Linear(d_model * 2 + 3, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, max(d_ff // 2, 1)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(d_ff // 2, 1), 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, float(gate_bias))

    def forward(
        self,
        state: torch.Tensor,
        context: torch.Tensor,
        previous_prediction: torch.Tensor,
        candidate_delta: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        features = torch.cat(
            [
                state,
                context,
                previous_prediction.unsqueeze(-1),
                candidate_delta.unsqueeze(-1),
                candidate_delta.detach().abs().unsqueeze(-1),
            ],
            dim=-1,
        )
        gate = self.max_gate * torch.sigmoid(self.net(features).squeeze(-1))
        if valid_mask is not None:
            gate = gate * valid_mask.to(gate.dtype)
        return gate


class ResidualStateUpdate(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, state: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, context], dim=-1))


class NWPPhysicalPriorHead(nn.Module):
    """Calibrates future NWP into a direct solar-output prior.

    The numerical baseline should not treat NWP as only another residual source:
    future irradiance, humidity, and temperature are exogenous forecasts for the
    target horizon.  This head learns a scale-calibrated physical candidate and
    a trust score that blends that candidate with the residual chain.
    """

    def __init__(
        self,
        nwp_dim: int,
        d_model: int,
        d_ff: int,
        dropout: float,
        time_dim: int = 6,
    ):
        super().__init__()
        input_dim = max(nwp_dim, 1) + time_dim + d_model * 2
        self.candidate = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, max(d_ff // 2, 1)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(d_ff // 2, 1), 1),
        )
        self.trust = nn.Sequential(
            nn.LayerNorm(input_dim + 3),
            nn.Linear(input_dim + 3, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, max(d_ff // 2, 1)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(d_ff // 2, 1), 1),
        )
        nn.init.zeros_(self.trust[-1].weight)
        nn.init.constant_(self.trust[-1].bias, -1.0)

    def forward(
        self,
        future_nwp: torch.Tensor,
        future_time_features: torch.Tensor,
        future_context: torch.Tensor,
        past_state: torch.Tensor,
        chain_prediction: torch.Tensor,
        future_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat(
            [future_nwp, future_time_features, future_context, past_state],
            dim=-1,
        )
        candidate = self.candidate(x).squeeze(-1)
        disagreement = torch.stack(
            [
                chain_prediction.detach(),
                candidate.detach(),
                (candidate.detach() - chain_prediction.detach()).abs(),
            ],
            dim=-1,
        )
        trust = torch.sigmoid(self.trust(torch.cat([x, disagreement], dim=-1)).squeeze(-1))
        trust = trust * future_mask.to(trust.dtype)
        candidate = candidate * future_mask.to(candidate.dtype)
        return candidate, trust


class PeriodicResidualPriorHead(nn.Module):
    """Uses the previous-day OT curve as a calibrated periodic candidate."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float,
        time_dim: int = 6,
    ):
        super().__init__()
        input_dim = 2 + time_dim + d_model * 2
        self.correction = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, max(d_ff // 2, 1)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(d_ff // 2, 1), 1),
        )
        self.trust = nn.Sequential(
            nn.LayerNorm(input_dim + 3),
            nn.Linear(input_dim + 3, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, max(d_ff // 2, 1)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(d_ff // 2, 1), 1),
        )
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)
        nn.init.zeros_(self.trust[-1].weight)
        nn.init.constant_(self.trust[-1].bias, -1.2)

    def forward(
        self,
        periodic_value: torch.Tensor,
        future_time_features: torch.Tensor,
        future_context: torch.Tensor,
        response_state: torch.Tensor,
        reference_prediction: torch.Tensor,
        future_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        periodic_value = periodic_value * future_mask.to(periodic_value.dtype)
        x = torch.cat(
            [
                periodic_value.unsqueeze(-1),
                (periodic_value - reference_prediction.detach()).unsqueeze(-1),
                future_time_features,
                future_context,
                response_state,
            ],
            dim=-1,
        )
        candidate = periodic_value + self.correction(x).squeeze(-1)
        disagreement = torch.stack(
            [
                reference_prediction.detach(),
                candidate.detach(),
                (candidate.detach() - reference_prediction.detach()).abs(),
            ],
            dim=-1,
        )
        trust = torch.sigmoid(self.trust(torch.cat([x, disagreement], dim=-1)).squeeze(-1))
        trust = trust * future_mask.to(trust.dtype)
        candidate = candidate * future_mask.to(candidate.dtype)
        return candidate, trust


class NumericalSolarBaseline(nn.Module):
    """Stepwise numerical residual baseline.

    The model is still trained as one clean checkpoint, but internally it follows
    the experimental logic: OT gives the base forecast, auxiliary meteorology
    corrects the OT residual, historical NWP corrects the remaining residual,
    and future NWP corrects the final numerical residual.  Text is never used
    here.
    """

    def __init__(
        self,
        input_dim: int,
        nwp_dim: int,
        seq_len: int,
        pred_len: int,
        d_model: int,
        n_heads: int,
        e_layers: int,
        d_layers: int,
        d_ff: int,
        dropout: float,
        nwp_layers: int,
        ma_kernel: int,
        max_delta: float,
        use_history_backbone: bool = False,
        use_nwp_prior: bool = False,
        use_periodic_prior: bool = False,
        use_aux_residual: bool = True,
        use_past_nwp_residual: bool = True,
        use_future_nwp_residual: bool = True,
        aux_gate_bias: float = -2.2,
        past_nwp_gate_bias: float = -3.0,
        future_nwp_gate_bias: float = -3.0,
        aux_gate_max: float = 0.80,
        past_nwp_gate_max: float = 0.70,
        future_nwp_gate_max: float = 0.70,
        dataset_count: int = 1,
        use_dataset_conditioning: bool = False,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.input_dim = int(input_dim)
        self.nwp_dim = int(nwp_dim)
        self.d_model = int(d_model)
        self.aux_dim = max(self.input_dim - 1, 1)
        self.max_delta = float(max_delta)
        self.use_history_backbone = bool(use_history_backbone)
        self.use_nwp_prior = bool(use_nwp_prior)
        self.use_periodic_prior = bool(use_periodic_prior)
        self.use_aux_residual = bool(use_aux_residual)
        self.use_past_nwp_residual = bool(use_past_nwp_residual)
        self.use_future_nwp_residual = bool(use_future_nwp_residual)
        self.dataset_count = int(max(dataset_count, 1))
        self.use_dataset_conditioning = bool(use_dataset_conditioning and self.dataset_count > 1)
        self.register_buffer(
            "candidate_pool_weights",
            torch.zeros(self.pred_len, len(NUMERICAL_CANDIDATE_NAMES)),
        )
        self.register_buffer(
            "dataset_candidate_pool_weights",
            torch.zeros(self.dataset_count, self.pred_len, len(NUMERICAL_CANDIDATE_NAMES)),
        )
        self.register_buffer("candidate_pool_enabled", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("dataset_candidate_pool_enabled", torch.tensor(False, dtype=torch.bool))
        self.backbone = EnhancedITransformer(
            seq_len=seq_len,
            pred_len=pred_len,
            input_dim=1,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_layers=d_layers,
            d_ff=d_ff,
            dropout=dropout,
            ma_kernel=ma_kernel,
        )
        self.history_backbone = EnhancedITransformer(
            seq_len=seq_len,
            pred_len=pred_len,
            input_dim=input_dim,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_layers=d_layers,
            d_ff=d_ff,
            dropout=dropout,
            ma_kernel=ma_kernel,
        )
        self.history_trust = nn.Sequential(
            nn.LayerNorm(d_model * 2 + 3),
            nn.Linear(d_model * 2 + 3, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, max(d_ff // 2, 1)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(d_ff // 2, 1), 1),
        )
        nn.init.zeros_(self.history_trust[-1].weight)
        nn.init.constant_(self.history_trust[-1].bias, -2.0)
        self.dataset_embedding = nn.Embedding(self.dataset_count, d_model)
        self.dataset_horizon_bias = nn.Sequential(
            nn.LayerNorm(d_model + 6),
            nn.Linear(d_model + 6, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, pred_len),
            nn.Tanh(),
        )
        self.dataset_gate_adapter = nn.Sequential(
            nn.LayerNorm(d_model + 6),
            nn.Linear(d_model + 6, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, pred_len * 5),
            nn.Tanh(),
        )
        self.aux_encoder = HistoricalSignalEncoder(
            self.aux_dim, d_model, n_heads, d_ff, max(nwp_layers, 1), dropout
        )
        self.past_nwp_encoder = HistoricalSignalEncoder(
            nwp_dim, d_model, n_heads, d_ff, max(nwp_layers, 1), dropout
        )
        self.future_nwp_encoder = FutureSignalEncoder(
            nwp_dim, d_model, n_heads, d_ff, max(nwp_layers, 1), dropout
        )
        self.aux_delta = ResidualDeltaHead(d_model, d_ff, dropout, max_delta)
        self.past_nwp_delta = ResidualDeltaHead(d_model, d_ff, dropout, max_delta)
        self.future_nwp_delta = ResidualDeltaHead(d_model, d_ff, dropout, max_delta)
        self.aux_gate = SafeResidualGate(
            d_model, d_ff, dropout, gate_bias=aux_gate_bias, max_gate=aux_gate_max
        )
        self.past_nwp_gate = SafeResidualGate(
            d_model, d_ff, dropout, gate_bias=past_nwp_gate_bias, max_gate=past_nwp_gate_max
        )
        self.future_nwp_gate = SafeResidualGate(
            d_model, d_ff, dropout, gate_bias=future_nwp_gate_bias, max_gate=future_nwp_gate_max
        )
        self.aux_update = ResidualStateUpdate(d_model, d_ff, dropout)
        self.past_nwp_update = ResidualStateUpdate(d_model, d_ff, dropout)
        self.future_nwp_update = ResidualStateUpdate(d_model, d_ff, dropout)
        self.nwp_physical_prior = NWPPhysicalPriorHead(
            nwp_dim=nwp_dim,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            time_dim=6,
        )
        self.periodic_prior = PeriodicResidualPriorHead(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            time_dim=6,
        )
        if not self.use_history_backbone:
            for parameter in self.history_backbone.parameters():
                parameter.requires_grad = False
            for parameter in self.history_trust.parameters():
                parameter.requires_grad = False
        if not self.use_nwp_prior:
            for parameter in self.nwp_physical_prior.parameters():
                parameter.requires_grad = False
        if not self.use_periodic_prior:
            for parameter in self.periodic_prior.parameters():
                parameter.requires_grad = False
        if not self.use_aux_residual:
            for module in [self.aux_delta, self.aux_gate]:
                for parameter in module.parameters():
                    parameter.requires_grad = False
        if not self.use_past_nwp_residual:
            for module in [self.past_nwp_delta, self.past_nwp_gate]:
                for parameter in module.parameters():
                    parameter.requires_grad = False
        if not self.use_future_nwp_residual:
            for module in [self.future_nwp_delta, self.future_nwp_gate]:
                for parameter in module.parameters():
                    parameter.requires_grad = False
        self.global_fusion = nn.Sequential(
            nn.LayerNorm(d_model * 4),
            nn.Linear(d_model * 4, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.LayerNorm(d_model),
        )
        if not self.use_dataset_conditioning:
            for parameter in self.dataset_embedding.parameters():
                parameter.requires_grad = False
            for parameter in self.dataset_horizon_bias.parameters():
                parameter.requires_grad = False
            for parameter in self.dataset_gate_adapter.parameters():
                parameter.requires_grad = False

    def _dataset_context(
        self,
        batch: Dict[str, torch.Tensor],
        reference: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = int(reference.shape[0])
        if "dataset_index" in batch:
            dataset_index = batch["dataset_index"].long().view(-1)
        else:
            dataset_index = torch.zeros(batch_size, dtype=torch.long, device=reference.device)
        dataset_index = dataset_index.clamp(0, self.dataset_count - 1)
        if not self.use_dataset_conditioning:
            dataset_context = torch.zeros(batch_size, self.d_model, dtype=reference.dtype, device=reference.device)
            dataset_bias = torch.zeros(batch_size, self.pred_len, dtype=reference.dtype, device=reference.device)
            dataset_gate_scale = torch.ones(batch_size, self.pred_len, 5, dtype=reference.dtype, device=reference.device)
            return dataset_index, dataset_context, dataset_bias, dataset_gate_scale
        dataset_context = self.dataset_embedding(dataset_index).to(dtype=reference.dtype)
        temporal_summary = batch["future_time_features"].to(reference.dtype).mean(dim=1)
        dataset_summary = torch.cat([dataset_context, temporal_summary], dim=-1)
        dataset_bias = self.dataset_horizon_bias(dataset_summary).to(dtype=reference.dtype)
        dataset_bias = dataset_bias * (0.05 * self.max_delta)
        dataset_gate_scale = 1.0 + 0.30 * self.dataset_gate_adapter(dataset_summary).view(
            batch_size, self.pred_len, 5
        ).to(dtype=reference.dtype)
        return dataset_index, dataset_context, dataset_bias, dataset_gate_scale

    def forward(self, batch: Dict[str, torch.Tensor], return_hidden: bool = False):
        x_ot = batch["x"][..., :1]
        dataset_index, dataset_context, dataset_bias, dataset_gate_scale = self._dataset_context(batch, x_ot)
        backbone_prediction, hidden = self.backbone(
            x_ot,
            batch["future_time_features"],
            return_hidden=True,
        )
        ot_only_prediction = backbone_prediction
        ot_state = hidden["horizon_state"]
        if self.use_history_backbone:
            history_prediction, history_hidden = self.history_backbone(
                batch["x"],
                batch["future_time_features"],
                return_hidden=True,
            )
            history_state = history_hidden["horizon_state"]
            trust_input = torch.cat(
                [
                    ot_state,
                    history_state,
                    backbone_prediction.unsqueeze(-1),
                    history_prediction.unsqueeze(-1),
                    (history_prediction - backbone_prediction).abs().unsqueeze(-1),
                ],
                dim=-1,
            )
            history_trust = 0.35 * torch.sigmoid(self.history_trust(trust_input).squeeze(-1))
            backbone_prediction = (
                (1.0 - history_trust) * backbone_prediction
                + history_trust * history_prediction
            )
            ot_state = (1.0 - history_trust.unsqueeze(-1)) * ot_state + history_trust.unsqueeze(-1) * history_state
        else:
            history_prediction = backbone_prediction.detach()
            history_trust = torch.zeros_like(backbone_prediction)
        hidden["horizon_state"] = ot_state

        aux_x = batch["x"][..., 1:]
        if aux_x.shape[-1] == 0:
            aux_x = torch.zeros(*batch["x"].shape[:2], 1, device=batch["x"].device)
        aux_context, aux_global = self.aux_encoder(
            aux_x,
            batch["past_offset_hours"],
            batch["future_offset_hours"],
            ot_state,
        )
        aux_raw_delta = self.aux_delta(aux_context)
        aux_gate = self.aux_gate(
            ot_state,
            aux_context,
            backbone_prediction,
            aux_raw_delta,
        )
        if self.use_dataset_conditioning:
            aux_gate = (aux_gate * dataset_gate_scale[..., 0]).clamp(0.0, 1.0)
        if not self.use_aux_residual:
            aux_gate = torch.zeros_like(aux_gate)
        aux_delta = aux_gate * aux_raw_delta
        aux_prediction = backbone_prediction + aux_delta
        aux_state = self.aux_update(ot_state, aux_context)

        past_context, past_global = self.past_nwp_encoder(
            batch["past_nwp"],
            batch["past_offset_hours"],
            batch["future_offset_hours"],
            aux_state,
            batch["past_nwp_mask"].bool(),
        )
        past_raw_delta = self.past_nwp_delta(past_context)
        past_gate = self.past_nwp_gate(
            aux_state,
            past_context,
            aux_prediction,
            past_raw_delta,
            batch["past_nwp_mask"].bool().any(dim=1, keepdim=True).expand(-1, self.pred_len),
        )
        if self.use_dataset_conditioning:
            past_gate = (past_gate * dataset_gate_scale[..., 1]).clamp(0.0, 1.0)
        if not self.use_past_nwp_residual:
            past_gate = torch.zeros_like(past_gate)
        past_delta = past_gate * past_raw_delta
        past_prediction = aux_prediction + past_delta
        past_state = self.past_nwp_update(aux_state, past_context)

        future_context, future_global = self.future_nwp_encoder(
            batch["future_nwp"],
            batch["future_offset_hours"],
            past_state,
            batch["future_nwp_mask"].bool(),
        )
        future_raw_delta = self.future_nwp_delta(future_context)
        future_gate = self.future_nwp_gate(
            past_state,
            future_context,
            past_prediction,
            future_raw_delta,
            batch["future_nwp_mask"].bool(),
        )
        if self.use_dataset_conditioning:
            future_gate = (future_gate * dataset_gate_scale[..., 2]).clamp(0.0, 1.0)
        if not self.use_future_nwp_residual:
            future_gate = torch.zeros_like(future_gate)
        future_delta = future_gate * future_raw_delta
        chain_prediction = past_prediction + future_delta
        fused = self.future_nwp_update(past_state, future_context)
        if self.use_nwp_prior:
            nwp_prior_prediction, nwp_prior_trust = self.nwp_physical_prior(
                batch["future_nwp"],
                batch["future_time_features"],
                future_context,
                past_state,
                chain_prediction,
                batch["future_nwp_mask"].bool(),
            )
        else:
            nwp_prior_prediction = chain_prediction.detach()
            nwp_prior_trust = torch.zeros_like(chain_prediction)
        if self.use_dataset_conditioning:
            nwp_prior_trust = (nwp_prior_trust * dataset_gate_scale[..., 3]).clamp(0.0, 1.0)
        prediction = (
            (1.0 - nwp_prior_trust) * chain_prediction
            + nwp_prior_trust * nwp_prior_prediction
        )
        if x_ot.shape[1] >= self.pred_len:
            periodic_value = x_ot[:, -self.pred_len :, 0]
        else:
            periodic_value = x_ot[:, -1:, 0].expand(-1, self.pred_len)
        if self.use_periodic_prior:
            periodic_prediction, periodic_trust = self.periodic_prior(
                periodic_value,
                batch["future_time_features"],
                future_context,
                aux_context,
                prediction,
                batch["future_nwp_mask"].bool(),
            )
        else:
            periodic_prediction = prediction.detach()
            periodic_trust = torch.zeros_like(prediction)
        if self.use_dataset_conditioning:
            periodic_trust = (periodic_trust * dataset_gate_scale[..., 4]).clamp(0.0, 1.0)
        pre_candidate_pool_prediction = (
            (1.0 - periodic_trust) * prediction
            + periodic_trust * periodic_prediction
        )
        if self.use_dataset_conditioning:
            pre_candidate_pool_prediction = pre_candidate_pool_prediction + dataset_bias
        candidate_pool = torch.stack(
            [
                pre_candidate_pool_prediction,
                ot_only_prediction,
                history_prediction,
                aux_prediction,
                past_prediction,
                chain_prediction,
                nwp_prior_prediction,
                periodic_prediction,
            ],
            dim=-1,
        )
        if bool(self.dataset_candidate_pool_enabled.item()):
            pool_weights = self.dataset_candidate_pool_weights.to(candidate_pool.dtype)
            selected_weights = pool_weights[dataset_index].view(-1, self.pred_len, len(NUMERICAL_CANDIDATE_NAMES))
            prediction = torch.sum(candidate_pool * selected_weights, dim=-1)
        elif bool(self.candidate_pool_enabled.item()):
            pool_weights = self.candidate_pool_weights.to(candidate_pool.dtype)
            prediction = torch.sum(candidate_pool * pool_weights.view(1, self.pred_len, -1), dim=-1)
        else:
            prediction = pre_candidate_pool_prediction
        numerical_global = self.global_fusion(
            torch.cat(
                [
                    ot_state.mean(dim=1),
                    aux_global,
                    past_global,
                    future_global,
                ],
                dim=-1,
            )
        )
        if return_hidden:
            hidden.update({
                "stage1_prediction": prediction,
                "pre_text_prediction": prediction,
                "ot_prediction": backbone_prediction,
                "ot_only_prediction": ot_only_prediction,
                "history_prediction": history_prediction,
                "history_trust": history_trust,
                "aux_prediction": aux_prediction,
                "past_nwp_prediction": past_prediction,
                "chain_prediction": chain_prediction,
                "pre_periodic_prediction": (
                    (1.0 - nwp_prior_trust) * chain_prediction
                    + nwp_prior_trust * nwp_prior_prediction
                ),
                "nwp_prior_prediction": nwp_prior_prediction,
                "nwp_prior_trust": nwp_prior_trust,
                "periodic_prediction": periodic_prediction,
                "pre_candidate_pool_prediction": pre_candidate_pool_prediction,
                "numerical_candidate_pool": candidate_pool,
                "numerical_candidate_pool_enabled": torch.full_like(
                    prediction,
                    float(self.candidate_pool_enabled.item() or self.dataset_candidate_pool_enabled.item())
                ),
                "dataset_index": dataset_index,
                "dataset_bias": dataset_bias,
                "dataset_gate_scale": dataset_gate_scale,
                "periodic_trust": periodic_trust,
                "periodic_value": periodic_value,
                "backbone_prediction": backbone_prediction,
                "aux_raw_delta": aux_raw_delta,
                "aux_delta": aux_delta,
                "aux_gate": aux_gate,
                "past_nwp_raw_delta": past_raw_delta,
                "past_nwp_delta": past_delta,
                "past_nwp_gate": past_gate,
                "future_nwp_raw_delta": future_raw_delta,
                "future_nwp_delta": future_delta,
                "future_nwp_gate": future_gate,
                "nwp_prior_delta": nwp_prior_prediction - chain_prediction,
                "periodic_delta": periodic_prediction - chain_prediction,
                "numerical_delta": prediction - backbone_prediction,
                "horizon_state": ot_state,
                "fused_horizon_state": fused,
                "nwp_fused_horizon_state": fused,
                "aux_context": aux_context,
                "past_nwp_context": past_context,
                "future_nwp_context": future_context,
                "nwp_regime": future_context,
                "nwp_global": 0.5 * (past_global + future_global),
                "response_state": aux_context,
                "response_global": numerical_global,
            })
            return prediction, hidden
        return prediction

    def objective(
        self,
        batch: Dict[str, torch.Tensor],
        prediction: torch.Tensor,
        parts: Dict[str, torch.Tensor],
        weights,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        main = event_weighted_forecasting_loss(
            prediction,
            batch["y"],
            batch["future_time_features"],
            weights.daytime_weight,
            weights.numeric_event,
        )
        target = batch["y"]
        aux_target = target - parts["ot_prediction"].detach()
        past_target = target - parts["aux_prediction"].detach()
        future_target = target - parts["past_nwp_prediction"].detach()
        aux_candidate_prediction = parts["ot_prediction"].detach() + parts["aux_raw_delta"]
        past_candidate_prediction = parts["aux_prediction"].detach() + parts["past_nwp_raw_delta"]
        future_candidate_prediction = parts["past_nwp_prediction"].detach() + parts["future_nwp_raw_delta"]
        residual_terms = []
        if self.use_aux_residual:
            residual_terms.append(F.smooth_l1_loss(parts["aux_raw_delta"], aux_target, beta=0.25))
        if self.use_past_nwp_residual:
            residual_terms.append(F.smooth_l1_loss(parts["past_nwp_raw_delta"], past_target, beta=0.25))
        if self.use_future_nwp_residual:
            residual_terms.append(F.smooth_l1_loss(parts["future_nwp_raw_delta"], future_target, beta=0.25))
        residual = (
            torch.stack(residual_terms).sum()
            if residual_terms
            else prediction.new_tensor(0.0)
        )
        history_prior = event_weighted_forecasting_loss(
            parts["history_prediction"],
            target,
            batch["future_time_features"],
            weights.daytime_weight,
            0.50 * weights.numeric_event,
        )
        if not self.use_history_backbone:
            history_prior = history_prior * 0.0
        nwp_prior = event_weighted_forecasting_loss(
            parts["nwp_prior_prediction"],
            target,
            batch["future_time_features"],
            weights.daytime_weight,
            0.50 * weights.numeric_event,
        )
        if not self.use_nwp_prior:
            nwp_prior = nwp_prior * 0.0
        periodic_prior = event_weighted_forecasting_loss(
            parts["periodic_prediction"],
            target,
            batch["future_time_features"],
            weights.daytime_weight,
            0.50 * weights.numeric_event,
        )
        if not self.use_periodic_prior:
            periodic_prior = periodic_prior * 0.0
        ot_error = F.mse_loss(parts["ot_prediction"].detach(), target)
        ot_only_error = F.mse_loss(parts["ot_only_prediction"], target)
        history_error = F.mse_loss(parts["history_prediction"], target)
        aux_error = F.mse_loss(parts["aux_prediction"], target)
        past_error = F.mse_loss(parts["past_nwp_prediction"], target)
        chain_error = F.mse_loss(parts["chain_prediction"], target)
        aux_candidate_error = F.mse_loss(aux_candidate_prediction, target)
        past_candidate_error = F.mse_loss(past_candidate_prediction, target)
        future_candidate_error = F.mse_loss(future_candidate_prediction, target)
        prior_error = F.mse_loss(parts["nwp_prior_prediction"], target)
        pre_periodic_error = F.mse_loss(parts["pre_periodic_prediction"], target)
        periodic_error = F.mse_loss(parts["periodic_prediction"], target)
        final_error = F.mse_loss(prediction, target)
        non_degradation = (
            F.relu(aux_error - ot_error + weights.improvement_margin)
            + F.relu(past_error - aux_error.detach() + weights.improvement_margin)
            + F.relu(chain_error - past_error.detach() + weights.improvement_margin)
            + F.relu(pre_periodic_error - torch.minimum(chain_error, prior_error).detach() + weights.improvement_margin)
            + F.relu(final_error - torch.minimum(pre_periodic_error, periodic_error).detach() + weights.improvement_margin)
        )
        delta = parts.get("numerical_delta", torch.zeros_like(prediction))
        history_trust = parts["history_trust"].mean()
        history_trust_target = (
            (parts["history_prediction"].detach() - target).square()
            < (parts["ot_only_prediction"].detach() - target).square()
        ).to(prediction.dtype)
        history_trust_regular = safe_probability_bce(
            parts["history_trust"], history_trust_target
        )
        if not self.use_history_backbone:
            history_trust_regular = history_trust_regular * 0.0
        prior_trust = parts["nwp_prior_trust"].mean()
        trust_target = (
            (parts["nwp_prior_prediction"].detach() - target).square()
            < (parts["chain_prediction"].detach() - target).square()
        ).to(prediction.dtype)
        trust_regular = safe_probability_bce(parts["nwp_prior_trust"], trust_target)
        if not self.use_nwp_prior:
            trust_regular = trust_regular * 0.0
        periodic_trust = parts["periodic_trust"].mean()
        periodic_trust_target = (
            (parts["periodic_prediction"].detach() - target).square()
            < (parts["pre_periodic_prediction"].detach() - target).square()
        ).to(prediction.dtype)
        periodic_trust_regular = safe_probability_bce(
            parts["periodic_trust"], periodic_trust_target
        )
        if not self.use_periodic_prior:
            periodic_trust_regular = periodic_trust_regular * 0.0
        aux_gate_target = (
            (aux_candidate_prediction.detach() - target).square()
            < (parts["ot_prediction"].detach() - target).square()
        ).to(prediction.dtype)
        past_gate_target = (
            (past_candidate_prediction.detach() - target).square()
            < (parts["aux_prediction"].detach() - target).square()
        ).to(prediction.dtype)
        future_gate_target = (
            (future_candidate_prediction.detach() - target).square()
            < (parts["past_nwp_prediction"].detach() - target).square()
        ).to(prediction.dtype)
        branch_gate_terms = []
        if self.use_aux_residual:
            branch_gate_terms.append(safe_probability_bce(parts["aux_gate"], aux_gate_target))
        if self.use_past_nwp_residual:
            branch_gate_terms.append(safe_probability_bce(parts["past_nwp_gate"], past_gate_target))
        if self.use_future_nwp_residual:
            branch_gate_terms.append(safe_probability_bce(parts["future_nwp_gate"], future_gate_target))
        branch_gate_regular = (
            torch.stack(branch_gate_terms).sum()
            if branch_gate_terms
            else prediction.new_tensor(0.0)
        )
        candidate_trust_sparse = (
            parts["history_trust"].mean()
            + parts["nwp_prior_trust"].mean()
            + parts["periodic_trust"].mean()
            + 0.5
            * (
                parts["aux_gate"].mean()
                + parts["past_nwp_gate"].mean()
                + parts["future_nwp_gate"].mean()
            )
        )
        energy = (
            parts["aux_delta"].square().mean()
            + parts["past_nwp_delta"].square().mean()
            + parts["future_nwp_delta"].square().mean()
            + 0.25 * parts["nwp_prior_delta"].square().mean()
            + 0.25 * parts["periodic_delta"].square().mean()
        )
        smooth = (
            (parts["aux_delta"][:, 1:] - parts["aux_delta"][:, :-1]).square().mean()
            + (parts["past_nwp_delta"][:, 1:] - parts["past_nwp_delta"][:, :-1]).square().mean()
            + (parts["future_nwp_delta"][:, 1:] - parts["future_nwp_delta"][:, :-1]).square().mean()
            + 0.25
            * (
                parts["nwp_prior_prediction"][:, 1:]
                - parts["nwp_prior_prediction"][:, :-1]
            ).square().mean()
        )
        total = (
            main
            + weights.residual * residual
            + weights.history_prior * history_prior
            + weights.history_trust * history_trust_regular
            + weights.nwp_prior * nwp_prior
            + weights.nwp_trust * trust_regular
            + weights.periodic_prior * periodic_prior
            + weights.periodic_trust * periodic_trust_regular
            + weights.candidate_trust_sparse * candidate_trust_sparse
            + weights.opportunity * branch_gate_regular
            + weights.non_degradation * non_degradation
            + weights.energy * energy
            + weights.smooth * smooth
        )
        terms = {
            "main": float(main.detach()),
            "residual": float(residual.detach()),
            "history_prior": float(history_prior.detach()),
            "history_trust": float(history_trust.detach()),
            "history_trust_regular": float(history_trust_regular.detach()),
            "nwp_prior": float(nwp_prior.detach()),
            "periodic_prior": float(periodic_prior.detach()),
            "nwp_trust": float(prior_trust.detach()),
            "nwp_trust_regular": float(trust_regular.detach()),
            "periodic_trust": float(periodic_trust.detach()),
            "periodic_trust_regular": float(periodic_trust_regular.detach()),
            "candidate_trust_sparse": float(candidate_trust_sparse.detach()),
            "ot_only_error": float(ot_only_error.detach()),
            "history_error": float(history_error.detach()),
            "aux_error": float(aux_error.detach()),
            "past_nwp_error": float(past_error.detach()),
            "chain_error": float(chain_error.detach()),
            "aux_candidate_error": float(aux_candidate_error.detach()),
            "past_nwp_candidate_error": float(past_candidate_error.detach()),
            "future_nwp_candidate_error": float(future_candidate_error.detach()),
            "prior_error": float(prior_error.detach()),
            "periodic_error": float(periodic_error.detach()),
            "branch_gate_regular": float(branch_gate_regular.detach()),
            "aux_gate": float(parts["aux_gate"].detach().mean()),
            "past_nwp_gate": float(parts["past_nwp_gate"].detach().mean()),
            "future_nwp_gate": float(parts["future_nwp_gate"].detach().mean()),
            "dataset_gate_scale": float(parts.get("dataset_gate_scale", torch.ones_like(prediction).unsqueeze(-1)).detach().mean()),
            "non_degradation": float(non_degradation.detach()),
            "energy": float(energy.detach()),
            "smooth": float(smooth.detach()),
            "delta_abs": float(delta.detach().abs().mean()),
        }
        return total, terms


class HourlyFutureTextEncoder(nn.Module):
    """Jointly encodes low/high forecast-time Text2 streams hourly."""

    def __init__(
        self,
        text_dim: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        layers: int,
        dropout: float,
    ):
        super().__init__()
        self.low_projection = DeepProjection(text_dim, d_model, d_ff, dropout)
        self.high_projection = DeepProjection(text_dim, d_model, d_ff, dropout)
        self.time_projection = nn.Sequential(
            nn.Linear(5, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.field_embedding = nn.Parameter(torch.empty(1, 2, d_model))
        nn.init.trunc_normal_(self.field_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.pool = MaskedAttentivePool(d_model)

    def forward(
        self,
        low_embedding: torch.Tensor,
        high_embedding: torch.Tensor,
        mask: torch.Tensor,
        offset_hours: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        low = self.low_projection(low_embedding)
        high = self.high_projection(high_embedding)
        time = self.time_projection(_time_features(offset_hours))
        low = low + time + self.field_embedding[:, :1]
        high = high + time + self.field_embedding[:, 1:2]
        tokens = torch.stack([low, high], dim=2)
        batch, hours, fields, dim = tokens.shape
        flat = tokens.reshape(batch, hours * fields, dim)
        flat_mask = mask.unsqueeze(-1).expand(-1, -1, fields).reshape(batch, hours * fields)
        safe_mask = flat_mask.clone()
        empty = ~safe_mask.any(dim=1)
        if empty.any():
            safe_mask[empty, 0] = True
        flat = self.encoder(flat, src_key_padding_mask=~safe_mask)
        flat = self.norm(flat).masked_fill(~flat_mask.unsqueeze(-1), 0.0)
        tokens = flat.reshape(batch, hours, fields, dim)
        low_token = tokens[:, :, 0]
        high_token = tokens[:, :, 1]
        global_text = self.pool(flat, flat_mask)
        return low_token, high_token, global_text


class RealtimeTextInnovationEncoder(nn.Module):
    """Encodes realtime historical text as field-aware semantic memory."""

    FIELD_COUNT = 3

    def __init__(
        self,
        text_dim: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        layers: int,
        dropout: float,
    ):
        super().__init__()
        self.projections = nn.ModuleList(
            [DeepProjection(text_dim, d_model, d_ff, dropout) for _ in range(self.FIELD_COUNT)]
        )
        self.time_projection = nn.Sequential(
            nn.Linear(5, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.field_embedding = nn.Parameter(torch.empty(1, 1, self.FIELD_COUNT, d_model))
        nn.init.trunc_normal_(self.field_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.pool = MaskedAttentivePool(d_model)

    def forward(
        self,
        state_sentence: torch.Tensor,
        trend_sentence: torch.Tensor,
        var_sentence: torch.Tensor,
        mask: torch.Tensor,
        offset_hours: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fields = [state_sentence, trend_sentence, var_sentence]
        projected = [
            projection(value) for projection, value in zip(self.projections, fields)
        ]
        tokens = torch.stack(projected, dim=2)
        tokens = tokens + self.time_projection(_time_features(offset_hours)).unsqueeze(2)
        tokens = tokens + self.field_embedding
        batch, hours, fields_count, dim = tokens.shape
        flat = tokens.reshape(batch, hours * fields_count, dim)
        flat_mask = mask.unsqueeze(-1).expand(-1, -1, fields_count).reshape(
            batch, hours * fields_count
        )
        safe_mask = flat_mask.clone()
        empty = ~safe_mask.any(dim=1)
        if empty.any():
            safe_mask[empty, 0] = True
        flat = self.encoder(flat, src_key_padding_mask=~safe_mask)
        flat = self.norm(flat).masked_fill(~flat_mask.unsqueeze(-1), 0.0)
        flat_time = offset_hours.unsqueeze(-1).expand(-1, -1, fields_count).reshape(
            batch, hours * fields_count
        )
        global_text = self.pool(flat, flat_mask)
        return flat, flat_mask, flat_time, global_text


class RealtimeTokenSlotEncoder(nn.Module):
    """Compresses hourly BGE word tokens into field-aware semantic slots.

    Sentence embeddings are too coarse for realtime weather descriptions: the
    useful signal may be a short phrase about ramp, cloud cover, mutation, or
    volatility.  This module extracts a small set of learned semantic slots from
    every hourly text field, then exposes those slots to asynchronous alignment.
    """

    FIELD_COUNT = 3

    def __init__(
        self,
        text_dim: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        layers: int,
        dropout: float,
        slots_per_field: int = 4,
    ):
        super().__init__()
        self.slots_per_field = int(max(slots_per_field, 1))
        self.token_projection = nn.ModuleList(
            [DeepProjection(text_dim, d_model, d_ff, dropout) for _ in range(self.FIELD_COUNT)]
        )
        self.sentence_projection = nn.ModuleList(
            [DeepProjection(text_dim, d_model, d_ff, dropout) for _ in range(self.FIELD_COUNT)]
        )
        self.time_projection = nn.Sequential(
            nn.Linear(5, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.field_embedding = nn.Parameter(torch.empty(1, 1, self.FIELD_COUNT, 1, d_model))
        self.slot_queries = nn.Parameter(torch.empty(self.FIELD_COUNT, self.slots_per_field, d_model))
        nn.init.trunc_normal_(self.field_embedding, std=0.02)
        nn.init.trunc_normal_(self.slot_queries, std=0.02)
        self.slot_score = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.pool = MaskedAttentivePool(d_model)

    def _field_slots(
        self,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
        sentence: torch.Tensor,
        field_index: int,
    ) -> torch.Tensor:
        batch, hours, token_count, _ = tokens.shape
        projected = self.token_projection[field_index](tokens.float())
        sentence_state = self.sentence_projection[field_index](sentence.float())
        slot_query = self.slot_queries[field_index].view(1, 1, self.slots_per_field, -1)
        query = self.slot_score(slot_query + sentence_state.unsqueeze(2))
        logits = torch.einsum("bhsd,bhtd->bhst", query, projected) / np.sqrt(projected.shape[-1])
        mask = token_mask.bool().unsqueeze(2)
        safe_mask = mask.clone()
        empty = ~safe_mask.any(dim=-1, keepdim=True)
        if empty.any():
            safe_mask = torch.where(empty, torch.ones_like(safe_mask), safe_mask)
        logits = logits.masked_fill(~safe_mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        weights = torch.where(mask, weights, torch.zeros_like(weights))
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        slots = torch.einsum("bhst,bhtd->bhsd", weights, projected)
        slots = slots + sentence_state.unsqueeze(2)
        return slots

    def forward(
        self,
        state_tokens: torch.Tensor,
        trend_tokens: torch.Tensor,
        var_tokens: torch.Tensor,
        state_token_mask: torch.Tensor,
        trend_token_mask: torch.Tensor,
        var_token_mask: torch.Tensor,
        state_sentence: torch.Tensor,
        trend_sentence: torch.Tensor,
        var_sentence: torch.Tensor,
        mask: torch.Tensor,
        offset_hours: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        token_sets = [state_tokens, trend_tokens, var_tokens]
        mask_sets = [state_token_mask, trend_token_mask, var_token_mask]
        sentence_sets = [state_sentence, trend_sentence, var_sentence]
        fields = [
            self._field_slots(tokens, token_mask, sentence, index)
            for index, (tokens, token_mask, sentence) in enumerate(
                zip(token_sets, mask_sets, sentence_sets)
            )
        ]
        slots = torch.stack(fields, dim=2)
        time = self.time_projection(_time_features(offset_hours)).unsqueeze(2).unsqueeze(3)
        slots = slots + time + self.field_embedding
        batch, hours, fields_count, slot_count, dim = slots.shape
        flat = slots.reshape(batch, hours * fields_count * slot_count, dim)
        flat_mask = (
            mask.bool()
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, -1, fields_count, slot_count)
            .reshape(batch, hours * fields_count * slot_count)
        )
        safe_mask = flat_mask.clone()
        empty = ~safe_mask.any(dim=1)
        if empty.any():
            safe_mask[empty, 0] = True
        flat = self.encoder(flat, src_key_padding_mask=~safe_mask)
        flat = self.norm(flat).masked_fill(~flat_mask.unsqueeze(-1), 0.0)
        flat_time = (
            offset_hours.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, -1, fields_count, slot_count)
            .reshape(batch, hours * fields_count * slot_count)
        )
        global_text = self.pool(flat, flat_mask)
        return flat, flat_mask, flat_time, global_text


class AsynchronousRealtimeCrossAttention(nn.Module):
    """Regime-conditioned deformable asynchronous attention for realtime text1.

    Query tokens are 15-minute numerical horizon states.  Key/value tokens are
    hourly realtime text fields from the historical window.  The alignment is
    not plain QK attention: every horizon step predicts meteorology-conditioned
    lag centers, lag widths, multi-scale temporal kernels, and field preferences
    over state/trend/variability text.  This lets the model look back to
    different historical textual events under different NWP regimes while
    preserving a strict causal mask.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        scales: Sequence[float] = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 24.0),
        field_count: int = 3,
        slots_per_field: int = 1,
        max_lag_hours: float = 48.0,
        transport_iterations: int = 4,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = d_model // n_heads
        self.scales = tuple(float(scale) for scale in scales)
        self.field_count = int(field_count)
        self.slots_per_field = int(max(slots_per_field, 1))
        self.max_lag_hours = float(max_lag_hours)
        self.transport_iterations = int(max(transport_iterations, 1))
        self.time_basis_dim = len(self.scales) * 2 + 4

        self.query = nn.Linear(d_model * 3, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.time_bias = nn.Sequential(
            nn.Linear(self.time_basis_dim, d_ff // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff // 2, n_heads),
        )
        self.condition_bias = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                d_ff,
                n_heads
                + d_model * 2
                + n_heads * 2
                + n_heads * len(self.scales)
                + n_heads * self.field_count,
            ),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.LayerNorm(d_model),
        )

    def _time_basis(self, dt: torch.Tensor) -> torch.Tensor:
        # dt = horizon_time - realtime_text_time.  For historical realtime text,
        # dt should be non-negative; negative pairs are masked.
        positive_dt = torch.relu(dt)
        features = []
        for scale in self.scales:
            scale = max(scale, 1e-3)
            features.append(torch.exp(-positive_dt / scale))
        for scale in self.scales:
            scale = max(scale, 1e-3)
            features.append(torch.exp(-0.5 * (positive_dt / scale).square()))
        features.extend(
            [
                torch.log1p(positive_dt),
                torch.sin(2.0 * torch.pi * dt / 24.0),
                torch.cos(2.0 * torch.pi * dt / 24.0),
                (dt >= 0.0).to(dt.dtype),
            ]
        )
        return torch.stack(features, dim=-1)

    def forward(
        self,
        horizon_state: torch.Tensor,
        nwp_regime: torch.Tensor,
        response_state: torch.Tensor,
        text_tokens: torch.Tensor,
        horizon_time: torch.Tensor,
        text_time: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        batch, horizon, _ = horizon_state.shape
        text_len = text_tokens.shape[1]
        base_condition = torch.cat([horizon_state, nwp_regime, response_state], dim=-1)
        q = self.query(base_condition)
        k = self.key(text_tokens)
        v = self.value(text_tokens)
        q = q.reshape(batch, horizon, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch, text_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch, text_len, self.n_heads, self.head_dim).transpose(1, 2)

        semantic = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        dt = horizon_time.unsqueeze(-1) - text_time.unsqueeze(-2)
        time_bias = self.time_bias(self._time_basis(dt)).permute(0, 3, 1, 2)
        controller = self.condition_bias(base_condition)
        split_sizes = [
            self.n_heads,
            self.d_model,
            self.d_model,
            self.n_heads,
            self.n_heads,
            self.n_heads * len(self.scales),
            self.n_heads * self.field_count,
        ]
        head_bias, gamma, beta, lag_center_raw, lag_width_raw, scale_raw, field_raw = torch.split(
            controller,
            split_sizes,
            dim=-1,
        )

        lag_center = self.max_lag_hours * torch.sigmoid(lag_center_raw)
        lag_width = 0.25 + 12.0 * torch.sigmoid(lag_width_raw)
        deformable_bias = -0.5 * (
            (dt.unsqueeze(1) - lag_center.permute(0, 2, 1).unsqueeze(-1))
            / lag_width.permute(0, 2, 1).unsqueeze(-1).clamp_min(1e-3)
        ).square()

        positive_dt = torch.relu(dt)
        scale_bank = torch.stack(
            [torch.exp(-positive_dt / max(scale, 1e-3)) for scale in self.scales],
            dim=-1,
        )
        scale_mix = torch.softmax(
            scale_raw.reshape(batch, horizon, self.n_heads, len(self.scales)),
            dim=-1,
        )
        adaptive_kernel = torch.sum(
            scale_mix.permute(0, 2, 1, 3).unsqueeze(3) * scale_bank.unsqueeze(1),
            dim=-1,
        )
        adaptive_kernel_bias = torch.log(adaptive_kernel.clamp_min(1e-6))

        field_logits = field_raw.reshape(batch, horizon, self.n_heads, self.field_count)
        field_id = (
            torch.arange(text_len, device=text_tokens.device) // self.slots_per_field
        ) % self.field_count
        field_bias = field_logits.permute(0, 2, 1, 3).gather(
            dim=-1,
            index=field_id.view(1, 1, 1, text_len).expand(batch, self.n_heads, horizon, text_len),
        )

        logits = (
            semantic
            + time_bias
            + deformable_bias
            + adaptive_kernel_bias
            + field_bias
            + head_bias.permute(0, 2, 1).unsqueeze(-1)
        )

        causal = dt >= -1e-6
        valid = text_mask.bool().unsqueeze(1).unsqueeze(2) & causal.unsqueeze(1)
        safe_valid = valid.clone()
        empty = ~safe_valid.any(dim=-1, keepdim=True)
        if empty.any():
            safe_valid = torch.where(empty, torch.ones_like(safe_valid), safe_valid)
        logits = logits.masked_fill(~safe_valid, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        weights = torch.where(valid, weights, torch.zeros_like(weights))
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        attended = torch.matmul(weights, v).transpose(1, 2).reshape(batch, horizon, self.d_model)
        attended = attended * (1.0 + 0.25 * torch.tanh(gamma)) + 0.25 * torch.tanh(beta)
        context = self.output(torch.cat([horizon_state, attended], dim=-1))
        mean_weights = weights.mean(dim=1)
        any_text = text_mask.any(dim=1, keepdim=True).unsqueeze(-1)
        context = torch.where(any_text, context, torch.zeros_like(context))
        entropy = -(weights.clamp_min(1e-8).log() * weights).sum(dim=-1).mean()
        expected_lag = (mean_weights * torch.relu(dt)).sum(dim=-1).mean()
        field_prob = torch.softmax(field_logits, dim=-1)
        field_entropy = -(field_prob.clamp_min(1e-8).log() * field_prob).sum(dim=-1).mean()
        diagnostics = {
            "weights": mean_weights,
            "entropy": entropy,
            "expected_lag": expected_lag,
            "lag_center_mean": lag_center.mean(),
            "lag_width_mean": lag_width.mean(),
            "field_entropy": field_entropy,
        }
        return context, mean_weights, diagnostics


class RegimeConditionedOptimalTransportResampler(nn.Module):
    """Align hourly text to 15-minute horizons with conditional optimal transport.

    The module constructs a cost matrix between every forecast horizon and every
    hourly text token.  Costs combine temporal distance, semantic mismatch, and
    NWP/response-conditioned demand and supply marginals.  Sinkhorn iterations
    produce a transport plan whose rows are then normalized to aggregate text
    values for each 15-minute forecast point.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float,
        sinkhorn_iterations: int = 8,
        epsilon: float = 0.55,
        scales: Sequence[float] = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 24.0),
    ):
        super().__init__()
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        self.epsilon = float(epsilon)
        self.scales = tuple(float(scale) for scale in scales)
        self.kernel_count = len(self.scales) * 2 + 2
        self.condition = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, self.kernel_count + d_model * 2 + 4),
        )
        self.query_key = nn.Linear(d_model, d_model, bias=False)
        self.text_key = nn.Linear(d_model, d_model, bias=False)
        self.text_value = nn.Linear(d_model, d_model)
        self.text_supply = nn.Sequential(
            nn.LayerNorm(d_model + 5),
            nn.Linear(d_model + 5, d_ff // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff // 2, 1),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.LayerNorm(d_model),
        )

    def _basis(self, dt: torch.Tensor) -> torch.Tensor:
        features = []
        for scale in self.scales:
            scale = max(scale, 1e-3)
            features.append(torch.exp(-0.5 * (dt / scale).square()))
        for scale in self.scales:
            scale = max(scale, 1e-3)
            features.append(torch.exp(-dt.abs() / scale))
        features.append(torch.sin(2.0 * torch.pi * dt / 24.0))
        features.append(torch.cos(2.0 * torch.pi * dt / 24.0))
        return torch.stack(features, dim=-1)

    @staticmethod
    def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
        safe = mask.any(dim=dim, keepdim=True)
        safe_mask = mask.clone()
        if dim == 1:
            empty = ~safe_mask.any(dim=1)
            if empty.any():
                safe_mask[empty, 0] = True
        masked = logits.masked_fill(~safe_mask, torch.finfo(logits.dtype).min)
        masked = torch.where(safe, masked, torch.zeros_like(masked))
        weights = torch.softmax(masked, dim=dim)
        weights = torch.where(safe_mask, weights, torch.zeros_like(weights))
        return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1e-8)

    def _sinkhorn(
        self,
        log_kernel: torch.Tensor,
        target_mass: torch.Tensor,
        source_mass: torch.Tensor,
    ) -> torch.Tensor:
        log_a = torch.log(target_mass.clamp_min(1e-8))
        log_b = torch.log(source_mass.clamp_min(1e-8))
        u = torch.zeros_like(log_a)
        v = torch.zeros_like(log_b)
        for _ in range(max(self.sinkhorn_iterations, 1)):
            u = log_a - torch.logsumexp(log_kernel + v.unsqueeze(1), dim=2)
            v = log_b - torch.logsumexp(log_kernel + u.unsqueeze(2), dim=1)
        return torch.exp(log_kernel + u.unsqueeze(2) + v.unsqueeze(1))

    def forward(
        self,
        horizon_state: torch.Tensor,
        nwp_regime: torch.Tensor,
        response_state: torch.Tensor,
        text_tokens: torch.Tensor,
        horizon_time: torch.Tensor,
        text_time: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        condition = self.condition(
            torch.cat([horizon_state, nwp_regime, response_state], dim=-1)
        )
        kernel_coeff, gamma, beta, controller = torch.split(
            condition,
            [
                self.kernel_count,
                horizon_state.shape[-1],
                horizon_state.shape[-1],
                4,
            ],
            dim=-1,
        )
        gamma = 0.5 * torch.tanh(gamma)
        beta = 0.5 * torch.tanh(beta)
        temporal_weight = F.softplus(controller[..., 0]) + 0.05
        semantic_weight = F.softplus(controller[..., 1]) + 0.05
        target_logits = controller[..., 2]
        temperature = F.softplus(controller[..., 3]).mean(dim=1, keepdim=True) + self.epsilon

        dt = horizon_time.unsqueeze(-1) - text_time.unsqueeze(-2)
        basis = self._basis(dt)
        kernel_prior = torch.sum(kernel_coeff.unsqueeze(2) * basis, dim=-1)

        semantic_logits = torch.matmul(
            self.query_key(horizon_state + 0.5 * response_state),
            self.text_key(text_tokens).transpose(-2, -1),
        ) / np.sqrt(horizon_state.shape[-1])

        temporal_cost = dt.abs() / 6.0
        semantic_cost = 1.0 - torch.tanh(semantic_logits)
        learned_cost = -0.25 * torch.tanh(kernel_prior)
        cost = (
            temporal_weight.unsqueeze(-1) * temporal_cost
            + semantic_weight.unsqueeze(-1) * semantic_cost
            + learned_cost
        )

        text_features = torch.cat([text_tokens, _time_features(text_time)], dim=-1)
        source_logits = self.text_supply(text_features).squeeze(-1)
        safe_text_mask = text_mask.bool().clone()
        empty_text = ~safe_text_mask.any(dim=1)
        if empty_text.any():
            safe_text_mask[empty_text, 0] = True
        target_mass = torch.softmax(target_logits, dim=1)
        source_mass = self._masked_softmax(source_logits, safe_text_mask, dim=1)

        log_kernel = -cost / temperature.unsqueeze(-1).clamp_min(1e-4)
        log_kernel = log_kernel.masked_fill(
            ~safe_text_mask[:, None, :],
            torch.finfo(log_kernel.dtype).min,
        )
        plan = self._sinkhorn(log_kernel, target_mass, source_mass)
        any_text = text_mask.any(dim=1, keepdim=True).unsqueeze(-1)
        plan = torch.where(any_text, plan, torch.zeros_like(plan))
        row_weights = plan / plan.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        value = self.text_value(text_tokens)
        value = value[:, None, :, :] * (1.0 + gamma.unsqueeze(2)) + beta.unsqueeze(2)
        context = torch.sum(row_weights.unsqueeze(-1) * value, dim=2)
        context = self.output(context)
        context = torch.where(any_text, context, torch.zeros_like(context))
        entropy = -(row_weights.clamp_min(1e-8).log() * row_weights).sum(dim=-1).mean()
        expected_cost = (plan * cost).sum(dim=(1, 2)).mean()
        diagnostics = {
            "plan": plan,
            "target_mass": target_mass,
            "source_mass": source_mass,
            "entropy": entropy,
            "expected_cost": expected_cost,
        }
        return context, row_weights, diagnostics


class NWPOrthogonalInnovation(nn.Module):
    """Softly down-weights text components explainable by NWP/response."""

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.text_projection = nn.Linear(d_model, d_model)
        self.nwp_projection = nn.Linear(d_model, d_model)
        self.response_projection = nn.Linear(d_model, d_model)
        self.modulation = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model * 2),
        )
        self.remove_gate = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, max(d_ff // 2, 1)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(d_ff // 2, 1), 1),
        )
        self.refinement = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.LayerNorm(d_model),
        )
        nn.init.zeros_(self.remove_gate[-1].weight)
        nn.init.constant_(self.remove_gate[-1].bias, 2.5)

    def forward(
        self,
        text_context: torch.Tensor,
        nwp_regime: torch.Tensor,
        response_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        text = self.text_projection(text_context)
        nwp = F.normalize(self.nwp_projection(nwp_regime), dim=-1)
        response = F.normalize(self.response_projection(response_state), dim=-1)
        explainable_nwp = (text * nwp).sum(dim=-1, keepdim=True) * nwp
        residual = text - explainable_nwp
        response_basis = response - (response * nwp).sum(dim=-1, keepdim=True) * nwp
        response_basis = F.normalize(response_basis, dim=-1)
        explainable_response = (
            residual * response_basis
        ).sum(dim=-1, keepdim=True) * response_basis
        explainable = explainable_nwp + explainable_response
        remove_alpha = 0.70 * torch.sigmoid(
            self.remove_gate(torch.cat([text, nwp_regime, response_state], dim=-1))
        )
        innovation = text - remove_alpha * explainable
        gamma, beta = self.modulation(
            torch.cat([nwp_regime, response_state], dim=-1)
        ).chunk(2, dim=-1)
        innovation = innovation * (1.0 + 0.25 * torch.tanh(gamma)) + 0.25 * torch.tanh(beta)
        innovation = self.refinement(innovation)
        return innovation, explainable


class ConditionedDeconfoundedInnovation(nn.Module):
    """Softly deconfounds condition-explainable text2 components."""

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.text_projection = nn.Linear(d_model, d_model)
        self.nwp_projection = nn.Linear(d_model, d_model)
        self.response_projection = nn.Linear(d_model, d_model)
        self.condition_projection = nn.Sequential(
            nn.LayerNorm(d_model * 2 + 5),
            nn.Linear(d_model * 2 + 5, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.LayerNorm(d_model),
        )
        self.condition_gate = nn.Sequential(
            nn.LayerNorm(d_model * 2 + 5),
            nn.Linear(d_model * 2 + 5, d_ff // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff // 2, 1),
        )
        self.remove_gate = nn.Sequential(
            nn.LayerNorm(d_model * 3 + 5),
            nn.Linear(d_model * 3 + 5, max(d_ff // 2, 1)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(d_ff // 2, 1), 1),
        )
        self.modulation = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model * 2),
        )
        self.refinement = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.LayerNorm(d_model),
        )
        nn.init.zeros_(self.condition_gate[-1].weight)
        nn.init.constant_(self.condition_gate[-1].bias, -0.2)
        nn.init.zeros_(self.remove_gate[-1].weight)
        nn.init.constant_(self.remove_gate[-1].bias, 1.5)

    def forward(
        self,
        text_context: torch.Tensor,
        nwp_regime: torch.Tensor,
        response_state: torch.Tensor,
        horizon_feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        text = self.text_projection(text_context)
        nwp = F.normalize(self.nwp_projection(nwp_regime), dim=-1)
        response = F.normalize(self.response_projection(response_state), dim=-1)
        explainable_nwp = (text * nwp).sum(dim=-1, keepdim=True) * nwp
        residual = text - explainable_nwp
        response_basis = response - (response * nwp).sum(dim=-1, keepdim=True) * nwp
        response_basis = F.normalize(response_basis, dim=-1)
        explainable_response = (
            residual * response_basis
        ).sum(dim=-1, keepdim=True) * response_basis
        explainable = explainable_nwp + explainable_response

        condition = torch.cat([nwp_regime, response_state, horizon_feature], dim=-1)
        condition_basis = F.normalize(self.condition_projection(condition), dim=-1)
        condition_gate = torch.sigmoid(self.condition_gate(condition).squeeze(-1))
        explainable_condition = (
            text * condition_basis
        ).sum(dim=-1, keepdim=True) * condition_basis
        explainable = explainable + condition_gate.unsqueeze(-1) * explainable_condition

        remove_alpha = 0.70 * torch.sigmoid(
            self.remove_gate(torch.cat([text, nwp_regime, response_state, horizon_feature], dim=-1))
        )
        innovation = text - remove_alpha * explainable
        gamma, beta = self.modulation(
            torch.cat([nwp_regime, response_state], dim=-1)
        ).chunk(2, dim=-1)
        innovation = innovation * (1.0 + 0.25 * torch.tanh(gamma)) + 0.25 * torch.tanh(beta)
        innovation = self.refinement(innovation)
        condition_overlap = (
            F.normalize(innovation, dim=-1) * condition_basis
        ).sum(dim=-1).abs()
        diagnostics = {
            "condition_gate": condition_gate,
            "remove_alpha": remove_alpha.squeeze(-1),
            "condition_overlap": condition_overlap,
            "condition_explainable": explainable_condition,
        }
        return innovation, explainable, diagnostics


class FutureExtremeWeatherShockModule(nn.Module):
    """Decodes future high-risk text into event-like residual shocks.

    The module is intentionally different from another cross-attention block.
    Hourly high-risk Text2 first writes evidence into a learnable semantic
    prototype memory.  Every 15-minute horizon state then selects risk
    prototypes under NWP and recent-response conditions, predicts event
    onset/intensity/duration, and integrates signed shock derivatives into a
    short-lived residual correction.  The high-risk text is first residualized
    against the low-frequency future trend so this branch specializes in
    semantic anomalies instead of repeating the long-horizon trend branch.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float,
        prototype_count: int = 8,
        max_delta: float = 0.12,
        max_duration_hours: float = 6.0,
    ):
        super().__init__()
        self.prototype_count = int(max(prototype_count, 2))
        self.max_delta = float(max_delta)
        self.max_duration_hours = float(max_duration_hours)
        self.prototype_prior_scale = nn.Parameter(torch.tensor(1.0))
        self.risk_prototypes = nn.Parameter(torch.empty(self.prototype_count, d_model))
        nn.init.trunc_normal_(self.risk_prototypes, std=0.02)

        self.text_key = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model, bias=False),
        )
        self.anomaly_projection = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.LayerNorm(d_model),
        )
        self.horizon_query = nn.Sequential(
            nn.LayerNorm(d_model * 4 + 5),
            nn.Linear(d_model * 4 + 5, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=False),
        )
        self.event_decoder = nn.Sequential(
            nn.LayerNorm(d_model * 6 + 5 + self.prototype_count),
            nn.Linear(d_model * 6 + 5 + self.prototype_count, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, max(d_ff // 2, 1)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(d_ff // 2, 1), 4),
        )
        nn.init.normal_(self.event_decoder[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.event_decoder[-1].bias)
        with torch.no_grad():
            self.event_decoder[-1].bias[1] = -2.0
            self.event_decoder[-1].bias[2] = -1.0
            self.event_decoder[-1].bias[3] = -0.5

    @staticmethod
    def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = mask.unsqueeze(-1).to(value.dtype)
        return (value * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)

    def forward(
        self,
        horizon_state: torch.Tensor,
        high_context: torch.Tensor,
        high_innovation: torch.Tensor,
        nwp_regime: torch.Tensor,
        response_state: torch.Tensor,
        low_hour: torch.Tensor,
        high_hour: torch.Tensor,
        alignment_weights: torch.Tensor,
        horizon_time: torch.Tensor,
        text_time: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch, horizon, dim = horizon_state.shape
        text_mask = text_mask.bool()
        any_text = text_mask.any(dim=1, keepdim=True)
        future_mask = any_text.to(horizon_state.dtype)

        high_anomaly = self.anomaly_projection(
            torch.cat([high_hour, low_hour, high_hour - low_hour], dim=-1)
        )
        prototypes = F.normalize(self.risk_prototypes, dim=-1)
        text_key = F.normalize(self.text_key(high_anomaly), dim=-1)
        token_logits = torch.matmul(text_key, prototypes.t()) / np.sqrt(dim)
        token_proto = torch.softmax(token_logits, dim=-1)
        token_proto = token_proto * text_mask.unsqueeze(-1).to(token_proto.dtype)

        if alignment_weights.shape[2] != token_proto.shape[1]:
            raise RuntimeError(
                "alignment_weights and high_hour must share the same text length"
            )
        horizon_proto_prior = torch.matmul(alignment_weights, token_proto)
        horizon_proto_prior = horizon_proto_prior / horizon_proto_prior.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)

        horizon_feature = _time_features(horizon_time)
        query_input = torch.cat(
            [horizon_state, high_innovation, nwp_regime, response_state, horizon_feature],
            dim=-1,
        )
        query = F.normalize(self.horizon_query(query_input), dim=-1)
        semantic_logits = torch.matmul(query, prototypes.t()) / np.sqrt(dim)
        proto_logits = semantic_logits + F.softplus(self.prototype_prior_scale) * torch.log(
            horizon_proto_prior.clamp_min(1e-6)
        )
        proto_weights = torch.softmax(proto_logits, dim=-1)
        proto_weights = proto_weights * future_mask.unsqueeze(-1)
        proto_context = torch.matmul(proto_weights, self.risk_prototypes)

        decoder_input = torch.cat(
            [
                horizon_state,
                high_context,
                high_innovation,
                nwp_regime,
                response_state,
                proto_context,
                horizon_feature,
                proto_weights,
            ],
            dim=-1,
        )
        raw_derivative, onset_logit, intensity_raw, duration_raw = self.event_decoder(
            decoder_input
        ).unbind(dim=-1)
        short_prior = torch.exp(
            -torch.relu(horizon_time) / max(self.max_duration_hours, 1e-3)
        )
        onset = torch.sigmoid(onset_logit) * short_prior * future_mask
        intensity = torch.sigmoid(intensity_raw)
        duration = 0.25 + self.max_duration_hours * torch.sigmoid(duration_raw)
        signed_impulse = self.max_delta * torch.tanh(
            raw_derivative / max(self.max_delta, 1e-6)
        )
        shock_derivative = signed_impulse * onset * intensity

        dt = horizon_time.unsqueeze(2) - horizon_time.unsqueeze(1)
        causal = (dt >= -1e-6).to(horizon_state.dtype)
        decay = torch.exp(-torch.relu(dt) / duration.unsqueeze(1).clamp_min(1e-3))
        response_kernel = causal * decay
        raw_shock = torch.sum(response_kernel * shock_derivative.unsqueeze(1), dim=2)
        shock_delta = self.max_delta * torch.tanh(raw_shock / max(self.max_delta, 1e-6))
        shock_delta = shock_delta * future_mask
        shock_derivative = shock_derivative * future_mask
        event_risk = onset * intensity

        valid_horizon = future_mask.expand(-1, horizon).bool()
        proto_entropy_each = -(
            proto_weights.clamp_min(1e-8).log() * proto_weights
        ).sum(dim=-1)
        if bool(valid_horizon.any().item()):
            prototype_entropy = proto_entropy_each[valid_horizon].mean()
            mean_proto = proto_weights[valid_horizon].mean(dim=0)
            prototype_balance = np.log(float(self.prototype_count)) + (
                mean_proto.clamp_min(1e-8).log() * mean_proto
            ).sum()
        else:
            prototype_entropy = proto_entropy_each.new_tensor(0.0)
            prototype_balance = proto_entropy_each.new_tensor(0.0)
        risk_tv = (
            event_risk[:, 1:] - event_risk[:, :-1]
        ).abs().mean() if horizon > 1 else event_risk.new_tensor(0.0)

        diagnostics = {
            "shock_delta": shock_delta,
            "shock_derivative": shock_derivative,
            "extreme_event_risk": event_risk,
            "risk_onset": onset,
            "risk_intensity": intensity * future_mask,
            "risk_duration": duration * future_mask,
            "risk_prototype_weights": proto_weights,
            "risk_prototype_entropy": prototype_entropy,
            "risk_prototype_balance": prototype_balance,
            "risk_tv": risk_tv,
        }
        return shock_delta, diagnostics


class FuzzyExtremeSemanticExtractor(nn.Module):
    """Extracts fuzzy extreme-weather semantics from Text2 anomalies.

    Text2 high-risk prompts are hourly and often vague.  This module treats
    them as fuzzy semantic evidence instead of deterministic labels.  It learns
    a small set of extreme-weather prototypes and returns horizon-wise fuzzy
    memberships plus ambiguity.  These values condition residual correction but
    never directly overwrite the numerical forecast.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float,
        fuzzy_sets: int = 6,
    ):
        super().__init__()
        self.fuzzy_sets = int(max(fuzzy_sets, 2))
        self.prototypes = nn.Parameter(torch.empty(self.fuzzy_sets, d_model))
        self.width = nn.Parameter(torch.zeros(self.fuzzy_sets))
        nn.init.trunc_normal_(self.prototypes, std=0.02)
        self.anomaly = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.LayerNorm(d_model),
        )
        self.context = nn.Sequential(
            nn.LayerNorm(d_model * 4 + 5 + self.fuzzy_sets),
            nn.Linear(d_model * 4 + 5 + self.fuzzy_sets, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.LayerNorm(d_model),
        )
        self.intensity = nn.Sequential(
            nn.LayerNorm(d_model * 2 + self.fuzzy_sets + 5),
            nn.Linear(d_model * 2 + self.fuzzy_sets + 5, d_ff // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff // 2, 1),
        )

    def forward(
        self,
        horizon_state: torch.Tensor,
        nwp_regime: torch.Tensor,
        response_state: torch.Tensor,
        low_hour: torch.Tensor,
        high_hour: torch.Tensor,
        alignment_weights: torch.Tensor,
        horizon_time: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        text_mask = text_mask.bool()
        any_text = text_mask.any(dim=1, keepdim=True)
        future_mask = any_text.to(horizon_state.dtype)
        anomaly = self.anomaly(torch.cat([high_hour, low_hour, high_hour - low_hour], dim=-1))
        prototypes = F.normalize(self.prototypes, dim=-1)
        anomaly_unit = F.normalize(anomaly, dim=-1)
        distance = 1.0 - torch.matmul(anomaly_unit, prototypes.t())
        width = F.softplus(self.width).view(1, 1, self.fuzzy_sets) + 0.05
        token_membership = torch.exp(-distance.square() / width)
        token_membership = token_membership * text_mask.unsqueeze(-1).to(token_membership.dtype)

        membership = torch.matmul(alignment_weights, token_membership)
        membership = membership / membership.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        membership = membership * future_mask.unsqueeze(-1)
        entropy = -(membership.clamp_min(1e-8).log() * membership).sum(dim=-1)
        ambiguity = entropy / np.log(float(self.fuzzy_sets))
        fuzzy_context = torch.matmul(membership, self.prototypes)
        horizon_feature = _time_features(horizon_time)
        fuzzy_context = self.context(
            torch.cat(
                [
                    horizon_state,
                    nwp_regime,
                    response_state,
                    fuzzy_context,
                    horizon_feature,
                    membership,
                ],
                dim=-1,
            )
        )
        horizon_anomaly = torch.matmul(alignment_weights, anomaly)
        fuzzy_intensity = torch.sigmoid(
            self.intensity(
                torch.cat(
                    [
                        fuzzy_context,
                        horizon_anomaly,
                        membership,
                        horizon_feature,
                    ],
                    dim=-1,
                )
            ).squeeze(-1)
        )
        fuzzy_intensity = fuzzy_intensity * future_mask
        valid = future_mask.expand_as(ambiguity).bool()
        if bool(valid.any().item()):
            fuzzy_entropy = entropy[valid].mean()
            fuzzy_ambiguity = ambiguity[valid].mean()
        else:
            fuzzy_entropy = entropy.new_tensor(0.0)
            fuzzy_ambiguity = ambiguity.new_tensor(0.0)
        return {
            "fuzzy_context": fuzzy_context,
            "fuzzy_membership": membership,
            "fuzzy_intensity": fuzzy_intensity,
            "fuzzy_ambiguity": ambiguity * future_mask,
            "fuzzy_entropy": fuzzy_entropy,
            "fuzzy_ambiguity_mean": fuzzy_ambiguity,
        }


class RealtimeAsynchronousResidualInjector(nn.Module):
    """Stage-1 realtime text injection over the numerical baseline.

    The asynchronous aligner has already handled the hourly-to-15-minute
    mismatch.  This module turns the aligned realtime text innovation into a
    conservative residual update, controlled by NWP and recent numerical
    response.  It is zero-initialized so the first-stage numerical baseline is
    preserved before training.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float,
        max_delta: float,
        scalar_dim: int = 15,
    ):
        super().__init__()
        self.max_delta = float(max_delta)
        self.scalar_dim = int(scalar_dim)
        input_dim = d_model * 4 + self.scalar_dim
        self.local = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, max(d_ff // 2, 1)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(d_ff // 2, 1), 1),
        )
        self.confidence = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, 1),
        )
        self.opportunity = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, max(d_ff // 2, 1)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(d_ff // 2, 1), 1),
        )
        self.direction = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, d_ff // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff // 2, 1),
        )
        nn.init.zeros_(self.local[-1].weight)
        nn.init.zeros_(self.local[-1].bias)
        nn.init.zeros_(self.confidence[-1].weight)
        nn.init.constant_(self.confidence[-1].bias, -1.5)
        nn.init.zeros_(self.opportunity[-1].weight)
        nn.init.constant_(self.opportunity[-1].bias, -1.0)
        nn.init.zeros_(self.direction[-1].weight)
        nn.init.zeros_(self.direction[-1].bias)

    def forward(
        self,
        horizon_state: torch.Tensor,
        nwp_regime: torch.Tensor,
        response_state: torch.Tensor,
        rt_innovation: torch.Tensor,
        scalar_feature: torch.Tensor,
        realtime_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if scalar_feature.shape[-1] != self.scalar_dim:
            raise RuntimeError(
                f"realtime scalar dimension mismatch: got {scalar_feature.shape[-1]}, "
                f"expected {self.scalar_dim}"
            )
        x = torch.cat(
            [horizon_state, nwp_regime, response_state, rt_innovation, scalar_feature],
            dim=-1,
        )
        local = self.local(x).squeeze(-1)
        confidence = torch.sigmoid(self.confidence(x).squeeze(-1)) * realtime_mask
        learned_opportunity = torch.sigmoid(self.opportunity(x).squeeze(-1))
        numeric_opportunity = scalar_feature[..., 5].clamp(0.0, 1.0)
        daylight_prior = scalar_feature[..., 7].clamp(0.0, 1.0)
        valid_prior = scalar_feature[..., 8].clamp(0.0, 1.0)
        opportunity = learned_opportunity * (0.10 + 0.90 * numeric_opportunity)
        opportunity = opportunity * (0.25 + 0.75 * daylight_prior) * valid_prior * realtime_mask
        directional_prior = self.max_delta * torch.tanh(self.direction(x).squeeze(-1))
        directional_prior = directional_prior * opportunity
        raw = (local * confidence + 0.25 * directional_prior) * opportunity
        if self.max_delta > 0:
            adaptive_cap = self.max_delta * (0.20 + 0.80 * opportunity)
            delta = adaptive_cap * torch.tanh(raw / adaptive_cap.clamp_min(1e-6))
        else:
            delta = raw
        return delta, {
            "realtime_confidence": confidence,
            "realtime_local": local,
            "realtime_opportunity": opportunity,
            "realtime_numeric_opportunity": numeric_opportunity * realtime_mask,
            "realtime_direction_prior": directional_prior,
        }


class Text2ResidualCorrectionDecoder(nn.Module):
    """Corrects the numerical baseline with forecast-time Text2 residual evidence.

    Text2 is treated as a residual signal.  Low-frequency forecast-time text
    supplies slow residual drift, high-frequency forecast-time text and fuzzy
    extreme semantics supply short-term residual evidence, while realtime text,
    NWP, and recent numerical response only condition confidence and regime
    adaptation.
    """

    def __init__(
        self,
        horizon: int,
        d_model: int,
        d_ff: int,
        condition_dim: int,
        experts: int,
        basis_rank: int,
        dropout: float,
        max_delta: float,
        text_scalar_dim: int = TEXT2_SCALAR_DIM,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.experts = int(experts)
        self.basis_rank = int(basis_rank)
        self.max_delta = float(max_delta)
        self.scalar_dim = 18
        self.text_scalar_dim = int(max(text_scalar_dim, 0))

        n = torch.arange(horizon, dtype=torch.float32)
        k = torch.arange(basis_rank, dtype=torch.float32).unsqueeze(1)
        basis = torch.cos(torch.pi / horizon * (n + 0.5) * k)
        basis[0] *= 1.0 / np.sqrt(horizon)
        if basis_rank > 1:
            basis[1:] *= np.sqrt(2.0 / horizon)
        self.register_buffer("basis", basis, persistent=True)

        self.text2_evidence = nn.Sequential(
            nn.LayerNorm(d_model * 5 + self.scalar_dim + self.text_scalar_dim),
            nn.Linear(d_model * 5 + self.scalar_dim + self.text_scalar_dim, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.LayerNorm(d_model),
        )
        self.local_residual = nn.Sequential(
            nn.LayerNorm(d_model * 4 + self.scalar_dim + self.text_scalar_dim),
            nn.Linear(d_model * 4 + self.scalar_dim + self.text_scalar_dim, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, max(d_ff // 2, 1)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(d_ff // 2, 1), 1),
        )
        self.route = nn.Sequential(
            nn.LayerNorm(condition_dim),
            nn.Linear(condition_dim, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, experts),
        )
        self.coefficient = nn.Sequential(
            nn.LayerNorm(condition_dim),
            nn.Linear(condition_dim, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, experts * basis_rank),
        )
        self.confidence = nn.Sequential(
            nn.LayerNorm(d_model * 6 + self.scalar_dim + self.text_scalar_dim),
            nn.Linear(d_model * 6 + self.scalar_dim + self.text_scalar_dim, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, 1),
        )
        self.opportunity_trigger = nn.Sequential(
            nn.LayerNorm(d_model * 4 + self.scalar_dim + self.text_scalar_dim),
            nn.Linear(d_model * 4 + self.scalar_dim + self.text_scalar_dim, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, max(d_ff // 2, 1)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(d_ff // 2, 1), 1),
        )
        self.semantic_prior = nn.Sequential(
            nn.LayerNorm(d_model * 4 + self.scalar_dim + self.text_scalar_dim),
            nn.Linear(d_model * 4 + self.scalar_dim + self.text_scalar_dim, d_ff // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff // 2, 1),
        )
        self.semantic_amplitude = nn.Sequential(
            nn.LayerNorm(d_model * 4 + self.scalar_dim + self.text_scalar_dim),
            nn.Linear(d_model * 4 + self.scalar_dim + self.text_scalar_dim, d_ff // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff // 2, 1),
        )
        nn.init.zeros_(self.local_residual[-1].weight)
        nn.init.zeros_(self.local_residual[-1].bias)
        nn.init.zeros_(self.coefficient[-1].weight)
        nn.init.zeros_(self.coefficient[-1].bias)
        nn.init.zeros_(self.confidence[-1].weight)
        nn.init.constant_(self.confidence[-1].bias, -1.0)
        nn.init.zeros_(self.opportunity_trigger[-1].weight)
        nn.init.constant_(self.opportunity_trigger[-1].bias, -0.8)
        nn.init.zeros_(self.semantic_prior[-1].weight)
        nn.init.zeros_(self.semantic_prior[-1].bias)
        nn.init.zeros_(self.semantic_amplitude[-1].weight)
        nn.init.constant_(self.semantic_amplitude[-1].bias, -1.0)

    def forward(
        self,
        horizon_state: torch.Tensor,
        nwp_regime: torch.Tensor,
        response_state: torch.Tensor,
        rt_innovation: torch.Tensor,
        low_innovation: torch.Tensor,
        high_innovation: torch.Tensor,
        fuzzy_context: torch.Tensor,
        shock_delta: torch.Tensor,
        shock_derivative: torch.Tensor,
        risk_weight: torch.Tensor,
        fuzzy_intensity: torch.Tensor,
        fuzzy_ambiguity: torch.Tensor,
        residual_opportunity: torch.Tensor,
        horizon_feature: torch.Tensor,
        text_scalar: torch.Tensor,
        future_mask: torch.Tensor,
        condition: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        scalar = torch.cat(
            [
                shock_delta.unsqueeze(-1),
                shock_derivative.unsqueeze(-1),
                risk_weight.unsqueeze(-1),
                fuzzy_intensity.unsqueeze(-1),
                fuzzy_ambiguity.unsqueeze(-1),
                residual_opportunity,
                horizon_feature,
                text_scalar,
            ],
            dim=-1,
        )
        if scalar.shape[-1] != self.scalar_dim + self.text_scalar_dim:
            raise RuntimeError(
                f"text2 residual scalar dimension mismatch: got {scalar.shape[-1]}, "
                f"expected {self.scalar_dim + self.text_scalar_dim}"
            )
        evidence = self.text2_evidence(
            torch.cat(
                [
                    low_innovation,
                    high_innovation,
                    fuzzy_context,
                    nwp_regime,
                    response_state,
                    scalar,
                ],
                dim=-1,
            )
        )
        local_input = torch.cat(
            [evidence, low_innovation, high_innovation, fuzzy_context, scalar],
            dim=-1,
        )
        local = self.local_residual(local_input).squeeze(-1)
        route = torch.softmax(self.route(condition), dim=-1)
        coeff = self.coefficient(condition).reshape(
            condition.shape[0], self.experts, self.basis_rank
        )
        spectral_each = torch.matmul(coeff, self.basis)
        spectral = torch.sum(route.unsqueeze(-1) * spectral_each, dim=1)
        confidence_input = torch.cat(
            [
                evidence,
                horizon_state,
                nwp_regime,
                response_state,
                rt_innovation,
                fuzzy_context,
                scalar,
            ],
            dim=-1,
        )
        confidence = torch.sigmoid(self.confidence(confidence_input).squeeze(-1))
        half = residual_opportunity.shape[-1] // 2
        signed_disagreement = residual_opportunity[..., :half]
        absolute_disagreement = residual_opportunity[..., half:]
        numeric_prior = torch.tanh(4.0 * absolute_disagreement.mean(dim=-1))
        text_down = text_scalar[..., 0].clamp(0.0, 1.0)
        text_up = text_scalar[..., 1].clamp(0.0, 1.0)
        text_extreme = text_scalar[..., 2].clamp(0.0, 1.0)
        text_fuzzy = text_scalar[..., 3].clamp(0.0, 1.0)
        text_stable = text_scalar[..., 4].clamp(0.0, 1.0)
        text_risk_prior = (0.58 * text_extreme + 0.42 * text_fuzzy) * (1.0 - 0.30 * text_stable)
        semantic_certainty = (
            0.26 * risk_weight
            + 0.30 * fuzzy_intensity
            + 0.18 * torch.tanh(shock_derivative.abs() / max(self.max_delta, 1e-6))
            + 0.30 * text_risk_prior
        )
        semantic_certainty = semantic_certainty.clamp(0.0, 1.0)
        semantic_certainty = semantic_certainty * (1.0 - 0.45 * fuzzy_ambiguity).clamp(0.10, 1.0)
        trigger_input = torch.cat(
            [evidence, nwp_regime, response_state, fuzzy_context, scalar],
            dim=-1,
        )
        learned_trigger = torch.sigmoid(
            self.opportunity_trigger(trigger_input).squeeze(-1)
        )
        prior_trigger = torch.maximum(numeric_prior, semantic_certainty)
        opportunity_gate = (0.20 + 0.80 * learned_trigger) * (0.20 + 0.80 * prior_trigger)
        opportunity_gate = (
            opportunity_gate
            + 0.12 * semantic_certainty * (0.35 + 0.65 * text_risk_prior)
        ).clamp(0.0, 1.0)
        opportunity_gate = opportunity_gate * future_mask
        signed_prior = torch.tanh(
            signed_disagreement.mean(dim=-1) / max(self.max_delta * 0.25, 1e-6)
        )
        semantic_direction = torch.tanh(self.semantic_prior(local_input).squeeze(-1))
        semantic_amplitude = torch.sigmoid(self.semantic_amplitude(local_input).squeeze(-1))
        text_direction = (text_up - text_down).clamp(-1.0, 1.0)
        semantic_drive = (
            0.45 * semantic_certainty
            + 0.35 * numeric_prior
            + 0.20 * risk_weight.clamp(0.0, 1.0)
        ).clamp(0.0, 1.0)
        direction_prior = 0.28 * self.max_delta * (
            0.50 * signed_prior + 0.30 * semantic_direction + 0.20 * text_direction
        ) * semantic_drive * (0.40 + 0.60 * semantic_amplitude)
        trusted_confidence = confidence * opportunity_gate
        prior_correction = direction_prior * opportunity_gate
        raw = ((local + spectral) * trusted_confidence + prior_correction) * future_mask
        adaptive_cap = self.max_delta * (
            0.15 + 0.75 * opportunity_gate + 0.10 * semantic_certainty * future_mask
        )
        if self.max_delta > 0:
            correction = adaptive_cap * torch.tanh(raw / adaptive_cap.clamp_min(1e-6))
        else:
            correction = raw
        diagnostics = {
            "text2_residual_evidence": evidence,
            "residual_confidence": trusted_confidence * future_mask,
            "residual_base_confidence": confidence * future_mask,
            "residual_opportunity_gate": opportunity_gate,
            "residual_numeric_prior": numeric_prior * future_mask,
            "residual_semantic_certainty": semantic_certainty * future_mask,
            "residual_direction_prior": direction_prior * future_mask,
            "residual_prior_correction": prior_correction * future_mask,
            "residual_local": local,
            "residual_spectral": spectral,
        }
        return correction, diagnostics


class HorizonWiseCandidateCalibrator(nn.Module):
    """Disagreement-aware horizon calibration for numerical hypotheses.

    The module calibrates numerical hypotheses under daylight and candidate
    disagreement regimes.  It starts as identity for backward compatibility, then
    learns horizon-specific scale/bias plus bounded adjustments driven by
    candidate spread, signed disagreement, and daylight context.
    """

    def __init__(self, horizon: int, num_candidates: int):
        super().__init__()
        self.horizon = int(horizon)
        self.num_candidates = int(num_candidates)
        self.log_scale = nn.Parameter(torch.zeros(self.horizon, self.num_candidates))
        self.bias = nn.Parameter(torch.zeros(self.horizon, self.num_candidates))
        self.day_log_scale = nn.Parameter(torch.zeros(1, self.num_candidates))
        self.day_bias = nn.Parameter(torch.zeros(1, self.num_candidates))
        self.disagreement_scale = nn.Parameter(torch.zeros(1, self.num_candidates))
        self.disagreement_bias = nn.Parameter(torch.zeros(1, self.num_candidates))
        self.direction_scale = nn.Parameter(torch.zeros(1, self.num_candidates))
        self.direction_bias = nn.Parameter(torch.zeros(1, self.num_candidates))

    def forward(
        self,
        candidates: torch.Tensor,
        daylight: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if candidates.shape[1] != self.horizon or candidates.shape[-1] != self.num_candidates:
            raise RuntimeError(
                f"candidate calibration shape mismatch: got {tuple(candidates.shape)}, "
                f"expected horizon={self.horizon}, candidates={self.num_candidates}"
            )
        day = daylight.to(candidates.dtype).unsqueeze(-1).clamp(0.0, 1.0)
        anchor = candidates[..., :1]
        delta = candidates - anchor
        spread = torch.tanh(delta.abs().mean(dim=-1, keepdim=True) / 0.08)
        direction = torch.tanh(delta.mean(dim=-1, keepdim=True) / 0.08)
        log_scale = (
            self.log_scale.unsqueeze(0)
            + day * self.day_log_scale.unsqueeze(0)
            + spread * self.disagreement_scale.unsqueeze(0)
            + direction * self.direction_scale.unsqueeze(0)
        )
        scale = torch.exp(log_scale.clamp(-0.25, 0.25))
        bias = (
            self.bias.unsqueeze(0)
            + day * self.day_bias.unsqueeze(0)
            + spread * self.disagreement_bias.unsqueeze(0)
            + direction * self.direction_bias.unsqueeze(0)
        )
        calibrated = scale * candidates + bias
        calibrated[..., 0] = candidates[..., 0]
        return calibrated, {
            "candidate_cal_scale": scale,
            "candidate_cal_bias": bias,
            "candidate_cal_disagreement": spread,
            "candidate_cal_direction": direction,
        }


class TextConditionedNumericalHypothesisRouter(nn.Module):
    """Routes among calibrated numerical hypotheses as a diagnostic branch.

    This is not the main paper contribution.  It exposes observable numerical
    disagreement and text-conditioned hypothesis preference so the residual
    release layer can measure opportunity.  Final prediction remains
    baseline-plus-released-residual instead of direct hypothesis replacement.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float,
        num_candidates: int,
        text_scalar_dim: int,
        opportunity_dim: int = 8,
        time_dim: int = 5,
        max_route_delta: float = 0.08,
    ):
        super().__init__()
        self.num_candidates = int(num_candidates)
        self.text_scalar_dim = int(text_scalar_dim)
        self.opportunity_dim = int(opportunity_dim)
        self.time_dim = int(time_dim)
        self.max_route_delta = float(max_route_delta)
        numeric_scalar_dim = self.opportunity_dim + self.time_dim + self.num_candidates + 4
        text_scalar_total = self.text_scalar_dim + 4
        numeric_input_dim = d_model * 3 + numeric_scalar_dim
        text_input_dim = d_model * 4 + text_scalar_total
        self.numeric_feature = nn.Sequential(
            nn.LayerNorm(numeric_input_dim),
            nn.Linear(numeric_input_dim, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_ff),
        )
        self.text_feature = nn.Sequential(
            nn.LayerNorm(text_input_dim),
            nn.Linear(text_input_dim, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_ff),
        )
        self.base_improvement = nn.Linear(d_ff, self.num_candidates)
        self.text_adjustment = nn.Linear(d_ff, self.num_candidates)
        self.release = nn.Linear(d_ff * 2 + 3, 1)
        nn.init.zeros_(self.base_improvement.weight)
        nn.init.zeros_(self.base_improvement.bias)
        nn.init.zeros_(self.text_adjustment.weight)
        nn.init.zeros_(self.text_adjustment.bias)
        nn.init.zeros_(self.release.weight)
        nn.init.constant_(self.release.bias, -3.5)

    def forward(
        self,
        horizon_state: torch.Tensor,
        nwp_regime: torch.Tensor,
        response_state: torch.Tensor,
        rt_innovation: torch.Tensor,
        low_innovation: torch.Tensor,
        high_innovation: torch.Tensor,
        fuzzy_context: torch.Tensor,
        text_scalar: torch.Tensor,
        candidates: torch.Tensor,
        residual_opportunity: torch.Tensor,
        horizon_feature: torch.Tensor,
        future_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        anchor = candidates[..., 0]
        candidate_delta = candidates - anchor.unsqueeze(-1)
        clipped_delta = self.max_route_delta * torch.tanh(
            candidate_delta / max(self.max_route_delta, 1e-6)
        )
        candidate_std = candidate_delta.std(dim=-1, unbiased=False)
        candidate_range = candidate_delta.max(dim=-1).values - candidate_delta.min(dim=-1).values
        candidate_mean_abs = candidate_delta.abs().mean(dim=-1)
        candidate_signed_mean = candidate_delta.mean(dim=-1)
        numeric_scalar = torch.cat(
            [
                residual_opportunity.detach(),
                horizon_feature,
                candidate_delta.detach(),
                candidate_std.unsqueeze(-1).detach(),
                candidate_range.unsqueeze(-1).detach(),
                candidate_mean_abs.unsqueeze(-1).detach(),
                candidate_signed_mean.unsqueeze(-1).detach(),
            ],
            dim=-1,
        )
        text_scalar_input = torch.cat(
            [
                text_scalar,
                candidate_std.unsqueeze(-1).detach(),
                candidate_range.unsqueeze(-1).detach(),
                candidate_mean_abs.unsqueeze(-1).detach(),
                candidate_signed_mean.unsqueeze(-1).detach(),
            ],
            dim=-1,
        )
        numeric_hidden = self.numeric_feature(
            torch.cat([horizon_state, nwp_regime, response_state, numeric_scalar], dim=-1)
        )
        text_hidden = self.text_feature(
            torch.cat([rt_innovation, low_innovation, high_innovation, fuzzy_context, text_scalar_input], dim=-1)
        )
        base_score = self.base_improvement(numeric_hidden)
        text_delta = 0.30 * torch.tanh(self.text_adjustment(text_hidden))
        improvement_score = base_score + text_delta
        selection_score = improvement_score.clone()
        selection_score[..., 0] = 0.0
        prob = torch.softmax(selection_score / 0.45, dim=-1)
        routed_delta = torch.sum(prob * clipped_delta, dim=-1)
        max_improve = improvement_score[..., 1:].max(dim=-1).values
        text_down = text_scalar[..., 0].clamp(0.0, 1.0)
        text_up = text_scalar[..., 1].clamp(0.0, 1.0)
        text_extreme = text_scalar[..., 2].clamp(0.0, 1.0)
        text_fuzzy = text_scalar[..., 3].clamp(0.0, 1.0)
        text_stable = text_scalar[..., 4].clamp(0.0, 1.0)
        text_conf_high = text_scalar[..., 8].clamp(0.0, 1.0) if self.text_scalar_dim > 8 else 0.0
        text_conf_low = text_scalar[..., 10].clamp(0.0, 1.0) if self.text_scalar_dim > 10 else 0.0
        semantic_risk = (
            0.34 * text_extreme
            + 0.30 * text_fuzzy
            + 0.18 * (text_down + text_up).clamp(0.0, 1.0)
            + 0.18 * (1.0 - text_stable)
        ).clamp(0.0, 1.0)
        semantic_reliability = (
            0.55 + 0.35 * semantic_risk + 0.15 * text_conf_high - 0.25 * text_conf_low
        ).clamp(0.20, 1.0)
        disagreement_release = torch.tanh(
            candidate_range.detach() / max(self.max_route_delta * 0.75, 1e-6)
        ).clamp(0.0, 1.0)
        positive_release = torch.sigmoid((max_improve - 0.04) / 0.12)
        release_input = torch.cat(
            [
                numeric_hidden,
                text_hidden,
                max_improve.unsqueeze(-1),
                candidate_range.unsqueeze(-1).detach(),
                candidate_mean_abs.unsqueeze(-1).detach(),
            ],
            dim=-1,
        )
        learned_release = torch.sigmoid(self.release(release_input).squeeze(-1))
        release_gate = (
            learned_release
            * positive_release
            * (0.20 + 0.80 * disagreement_release)
            * semantic_reliability
            * future_mask
        )
        route_delta = release_gate * routed_delta
        hard_index = prob.argmax(dim=-1, keepdim=True)
        hard_prob = torch.zeros_like(prob).scatter_(-1, hard_index, 1.0)
        entropy = -torch.sum(prob * torch.log(prob.clamp_min(1e-8)), dim=-1)
        return {
            "router_logits": improvement_score,
            "router_prob": prob,
            "router_hard_prob": hard_prob,
            "router_gate": release_gate,
            "router_entropy_h": entropy,
            "router_top_prob": prob.max(dim=-1).values,
            "route_prediction": anchor + routed_delta,
            "route_delta_unscaled": route_delta,
            "candidate_std": candidate_std,
            "candidate_range": candidate_range,
            "candidate_mean_abs": candidate_mean_abs,
            "base_improvement_score": base_score,
            "text_improvement_delta": text_delta,
            "improvement_score": improvement_score,
            "semantic_release": semantic_reliability,
            "positive_release": positive_release,
            "disagreement_release": disagreement_release,
        }


class CounterfactualEvidentialReleaseGate(nn.Module):
    """Evidence-calibrated Text2 residual release.

    The module treats Text2 correction as a residual proposal that must pass a
    small evidential test before release.  The prior is the conservative v20
    gate, while learnable evidence estimates agreement, counterfactual text gain,
    numerical uncertainty, and extreme-event opportunity.  Zero initialization
    keeps the initial behavior close to the prior.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float,
        scalar_dim: int,
    ):
        super().__init__()
        self.scalar_dim = int(scalar_dim)
        input_dim = d_model * 6 + self.scalar_dim
        hidden = max(d_ff // 2, 1)
        self.feature = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden),
        )
        self.evidence = nn.Linear(hidden, 2)
        self.logit_offset = nn.Linear(hidden, 1)
        self.cap = nn.Linear(hidden, 1)
        nn.init.zeros_(self.evidence.weight)
        nn.init.constant_(self.evidence.bias, 1.2)
        nn.init.zeros_(self.logit_offset.weight)
        nn.init.zeros_(self.logit_offset.bias)
        nn.init.zeros_(self.cap.weight)
        nn.init.constant_(self.cap.bias, 1.5)

    def forward(
        self,
        horizon_state: torch.Tensor,
        nwp_regime: torch.Tensor,
        response_state: torch.Tensor,
        rt_innovation: torch.Tensor,
        residual_evidence: torch.Tensor,
        fuzzy_context: torch.Tensor,
        residual_proposal: torch.Tensor,
        neutral_proposal: torch.Tensor,
        prior_gate: torch.Tensor,
        decoder_gate: torch.Tensor,
        semantic_gate: torch.Tensor,
        extreme_gate: torch.Tensor,
        router_delta: torch.Tensor,
        candidate_range: torch.Tensor,
        residual_numeric_prior: torch.Tensor,
        semantic_certainty: torch.Tensor,
        fuzzy_ambiguity: torch.Tensor,
        aligned_event_score: torch.Tensor,
        text_ramp_gate: torch.Tensor,
        horizon_feature: torch.Tensor,
        text_scalar: torch.Tensor,
        future_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        proposal_abs = residual_proposal.detach().abs()
        neutral_abs = neutral_proposal.detach().abs()
        proposal_advantage = (proposal_abs - neutral_abs).clamp(-1.0, 1.0)
        proposal_disagreement = (residual_proposal.detach() - neutral_proposal.detach()).abs()
        route_agreement = torch.tanh(
            residual_proposal.detach() * router_delta.detach() / 0.01
        )
        text_direction = (
            text_scalar[..., 1].detach().clamp(0.0, 1.0)
            - text_scalar[..., 0].detach().clamp(0.0, 1.0)
        ).clamp(-1.0, 1.0)
        proposal_direction = torch.tanh(residual_proposal.detach() / 0.04)
        semantic_agreement = (0.5 + 0.5 * text_direction * proposal_direction).clamp(0.0, 1.0)
        numeric_uncertainty = torch.tanh(candidate_range.detach() / 0.08).clamp(0.0, 1.0)
        prior_gate = prior_gate.detach().clamp(1e-5, 1.0 - 1e-5)
        scalar = torch.cat(
            [
                prior_gate.unsqueeze(-1),
                decoder_gate.detach().unsqueeze(-1),
                semantic_gate.detach().unsqueeze(-1),
                extreme_gate.detach().unsqueeze(-1),
                proposal_abs.unsqueeze(-1),
                proposal_disagreement.unsqueeze(-1),
                proposal_advantage.unsqueeze(-1),
                route_agreement.unsqueeze(-1),
                semantic_agreement.unsqueeze(-1),
                numeric_uncertainty.unsqueeze(-1),
                residual_numeric_prior.detach().unsqueeze(-1),
                semantic_certainty.detach().unsqueeze(-1),
                fuzzy_ambiguity.detach().unsqueeze(-1),
                aligned_event_score.detach().unsqueeze(-1),
                text_ramp_gate.detach().unsqueeze(-1),
                horizon_feature,
                text_scalar.detach(),
            ],
            dim=-1,
        )
        if scalar.shape[-1] != self.scalar_dim:
            raise RuntimeError(
                f"release scalar dimension mismatch: got {scalar.shape[-1]}, "
                f"expected {self.scalar_dim}"
            )
        hidden = self.feature(
            torch.cat(
                [
                    horizon_state,
                    nwp_regime,
                    response_state,
                    rt_innovation,
                    residual_evidence,
                    fuzzy_context,
                    scalar,
                ],
                dim=-1,
            )
        )
        evidence = F.softplus(self.evidence(hidden)) + 1e-4
        alpha = 1.0 + evidence[..., 0]
        beta = 1.0 + evidence[..., 1]
        evidence_mean = alpha / (alpha + beta).clamp_min(1e-6)
        uncertainty = (2.0 / (alpha + beta).clamp_min(2.0)).clamp(0.0, 1.0)
        prior_logit = torch.logit(prior_gate)
        offset = 0.65 * torch.tanh(self.logit_offset(hidden).squeeze(-1))
        learned_logit = prior_logit + offset + 0.35 * torch.logit(evidence_mean.clamp(1e-5, 1.0 - 1e-5))
        learned_gate = torch.sigmoid(learned_logit)
        cap = 0.20 + 0.80 * torch.sigmoid(self.cap(hidden).squeeze(-1))
        gate = torch.minimum(learned_gate, cap)
        gate = gate * (1.0 - 0.18 * uncertainty) * future_mask
        diagnostics = {
            "release_alpha": alpha,
            "release_beta": beta,
            "release_evidence_mean": evidence_mean,
            "release_uncertainty": uncertainty,
            "release_cap": cap,
            "release_logit_offset": offset,
            "release_agreement": semantic_agreement,
            "release_route_agreement": route_agreement,
            "release_counterfactual_gain": proposal_advantage,
            "release_numeric_uncertainty": numeric_uncertainty,
        }
        return gate.clamp(0.0, 1.0), diagnostics


class TextualCausalResidualIntervention(nn.Module):
    """Counterfactual textual treatment effect for forecast-time residuals.

    The module does not directly fuse Text2 into the forecast.  It first estimates
    the residual effect that disappears under a neutral-text intervention, then
    releases the stable low-frequency treatment and event-local high-frequency
    treatment under observable NWP/text regimes.
    """

    def __init__(
        self,
        pred_len: int,
        d_model: int,
        d_ff: int,
        dropout: float,
        max_delta: float,
        scalar_dim: int,
    ):
        super().__init__()
        self.max_delta = float(max_delta)
        self.scalar_dim = int(scalar_dim)
        self.bands = SmoothDCTBands(pred_len)
        input_dim = d_model * 6 + self.scalar_dim
        hidden = max(d_ff // 2, 1)
        self.regime = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden),
        )
        self.low_release = nn.Linear(hidden, 1)
        self.high_release = nn.Linear(hidden, 1)
        self.proposal_mix = nn.Linear(hidden, 1)
        self.effect_scale = nn.Linear(hidden, 1)
        self.event_bias = nn.Linear(hidden, 1)
        nn.init.zeros_(self.low_release.weight)
        nn.init.constant_(self.low_release.bias, 0.20)
        nn.init.zeros_(self.high_release.weight)
        nn.init.constant_(self.high_release.bias, -0.35)
        nn.init.zeros_(self.proposal_mix.weight)
        nn.init.constant_(self.proposal_mix.bias, 0.35)
        nn.init.zeros_(self.effect_scale.weight)
        nn.init.zeros_(self.effect_scale.bias)
        nn.init.zeros_(self.event_bias.weight)
        nn.init.zeros_(self.event_bias.bias)

    def forward(
        self,
        horizon_state: torch.Tensor,
        nwp_regime: torch.Tensor,
        response_state: torch.Tensor,
        rt_innovation: torch.Tensor,
        residual_evidence: torch.Tensor,
        fuzzy_context: torch.Tensor,
        factual_proposal: torch.Tensor,
        neutral_proposal: torch.Tensor,
        evidential_gate: torch.Tensor,
        prior_gate: torch.Tensor,
        event_gate: torch.Tensor,
        text_ramp_gate: torch.Tensor,
        residual_numeric_prior: torch.Tensor,
        fuzzy_ambiguity: torch.Tensor,
        candidate_range: torch.Tensor,
        horizon_feature: torch.Tensor,
        text_scalar: torch.Tensor,
        future_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        treatment_effect = factual_proposal - neutral_proposal.detach()
        counterfactual_gap = treatment_effect
        treatment_low, treatment_high = self.bands.split(treatment_effect)
        text_direction = (
            text_scalar[..., 1].detach().clamp(0.0, 1.0)
            - text_scalar[..., 0].detach().clamp(0.0, 1.0)
        ).clamp(-1.0, 1.0)
        scalar = torch.cat(
            [
                evidential_gate.detach().unsqueeze(-1),
                prior_gate.detach().unsqueeze(-1),
                event_gate.detach().unsqueeze(-1),
                text_ramp_gate.detach().unsqueeze(-1),
                residual_numeric_prior.detach().unsqueeze(-1),
                fuzzy_ambiguity.detach().unsqueeze(-1),
                torch.tanh(candidate_range.detach() / 0.08).unsqueeze(-1),
                factual_proposal.detach().abs().unsqueeze(-1),
                neutral_proposal.detach().abs().unsqueeze(-1),
                treatment_effect.detach().abs().unsqueeze(-1),
                torch.tanh(factual_proposal.detach() * treatment_effect.detach() / 0.01).unsqueeze(-1),
                torch.tanh(treatment_high.detach() / max(self.max_delta * 0.25, 1e-6)).unsqueeze(-1),
                text_direction.unsqueeze(-1),
                horizon_feature,
                text_scalar.detach(),
            ],
            dim=-1,
        )
        if scalar.shape[-1] != self.scalar_dim:
            raise RuntimeError(
                f"textual intervention scalar dimension mismatch: got {scalar.shape[-1]}, "
                f"expected {self.scalar_dim}"
            )
        hidden = self.regime(
            torch.cat(
                [
                    horizon_state,
                    nwp_regime,
                    response_state,
                    rt_innovation,
                    residual_evidence,
                    fuzzy_context,
                    scalar,
                ],
                dim=-1,
            )
        )
        low_observable = (
            0.35 * evidential_gate.detach()
            + 0.35 * prior_gate.detach()
            + 0.30 * residual_numeric_prior.detach().clamp(0.0, 1.0)
        ).clamp(0.0, 1.0)
        high_observable = (
            0.35 * event_gate.detach().clamp(0.0, 1.0)
            + 0.35 * text_ramp_gate.detach().clamp(0.0, 1.0)
            + 0.20 * evidential_gate.detach()
            + 0.10 * torch.tanh(candidate_range.detach() / 0.08).clamp(0.0, 1.0)
        ).clamp(0.0, 1.0)
        low_release = (
            torch.sigmoid(self.low_release(hidden).squeeze(-1)) * low_observable * future_mask
        ).clamp(0.0, 1.0)
        high_release = (
            torch.sigmoid(self.high_release(hidden).squeeze(-1)) * high_observable * future_mask
        ).clamp(0.0, 1.0)
        scale = 0.65 + 0.55 * torch.sigmoid(self.effect_scale(hidden).squeeze(-1))
        event_bias = (
            0.06
            * self.max_delta
            * torch.tanh(self.event_bias(hidden).squeeze(-1))
            * high_observable
            * text_direction
            * future_mask
        )
        event_bias = self.bands.project(event_bias, "high")
        released_low = low_release * treatment_low * scale
        released_high = high_release * treatment_high * scale
        mix = torch.sigmoid(self.proposal_mix(hidden).squeeze(-1))
        treatment_augmented = (
            mix * factual_proposal
            + (1.0 - mix) * treatment_effect
        ) * future_mask
        released = released_low + released_high + event_bias
        if self.max_delta > 0:
            released = self.max_delta * torch.tanh(released / self.max_delta)
        released = released * future_mask
        diagnostics = {
            "text2_treatment_effect": treatment_effect,
            "counterfactual_gap_delta": counterfactual_gap,
            "treatment_augmented_proposal": treatment_augmented,
            "treatment_low_delta": treatment_low,
            "treatment_high_delta": treatment_high,
            "released_text2_treatment": released,
            "released_treatment_low": released_low,
            "released_treatment_high": released_high,
            "event_treatment_delta": released_high + event_bias,
            "trend_treatment_delta": released_low,
            "treatment_low_release": low_release,
            "treatment_high_release": high_release,
            "treatment_effect_scale": scale,
            "treatment_proposal_mix": mix,
            "treatment_event_bias": event_bias,
            "treatment_event_focus": high_observable,
            "treatment_trend_focus": low_observable,
        }
        return released, diagnostics


class SolarUnifiedTextFusionModel(nn.Module):
    def __init__(
        self,
        baseline_model: NumericalSolarBaseline,
        input_dim: int,
        text1_dim: int,
        text2_dim: int,
        nwp_dim: int,
        pred_len: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        text_layers: int,
        nwp_layers: int,
        experts: int,
        basis_rank: int,
        dropout: float,
        cutoff_index: float,
        max_delta: float,
        risk_horizon_hours: float,
        text2_max_delta: Optional[float] = None,
        realtime_slots_per_field: int = 4,
        freeze_baseline: bool = True,
        use_realtime_condition: bool = True,
        use_fuzzy_extreme: bool = True,
        use_shock_evidence: bool = True,
        use_low_text2: bool = True,
        use_high_text2: bool = True,
        use_text_router: bool = True,
        use_text2_correction: bool = True,
        text2_gradient_warmup: float = 0.05,
    ):
        super().__init__()
        self.baseline = baseline_model
        self.d_model = int(d_model)
        self.freeze_baseline = bool(freeze_baseline)
        self.use_realtime_condition = bool(use_realtime_condition)
        self.use_fuzzy_extreme = bool(use_fuzzy_extreme)
        self.use_shock_evidence = bool(use_shock_evidence)
        self.use_low_text2 = bool(use_low_text2)
        self.use_high_text2 = bool(use_high_text2)
        self.use_text_router = bool(use_text_router)
        self.use_text2_correction = bool(use_text2_correction)
        self.text2_gradient_warmup = float(max(text2_gradient_warmup, 0.0))
        self.max_delta = float(max_delta)
        self.text2_max_delta = float(
            text2_max_delta if text2_max_delta is not None and text2_max_delta > 0 else max_delta
        )
        self.realtime_slots_per_field = int(max(realtime_slots_per_field, 1))
        self.realtime_scalar_dim = 15
        text2_scale_init = 0.12 if self.use_text2_correction else 0.0
        router_scale_init = 0.08 if self.use_text2_correction else 0.0
        ramp_scale_init = 0.10 if self.use_text2_correction else 0.0
        self.text2_stage_scale = nn.Parameter(_atanh_scalar(text2_scale_init))
        self.router_stage_scale = nn.Parameter(_atanh_scalar(router_scale_init))
        self.text_ramp_stage_scale = nn.Parameter(_atanh_scalar(ramp_scale_init))
        self.nwp_encoder = NWPRegimeEncoder(
            nwp_dim, d_model, n_heads, d_ff, nwp_layers, dropout
        )
        self.response_encoder = ObservedNumericalResponseEncoder(
            input_dim, d_model, n_heads, d_ff, max(nwp_layers, 1), dropout
        )
        self.realtime_text_encoder = RealtimeTokenSlotEncoder(
            text1_dim,
            d_model,
            n_heads,
            d_ff,
            text_layers,
            dropout,
            slots_per_field=self.realtime_slots_per_field,
        )
        self.text_encoder = HourlyFutureTextEncoder(
            text2_dim, d_model, n_heads, d_ff, text_layers, dropout
        )
        self.realtime_aligner = AsynchronousRealtimeCrossAttention(
            d_model,
            n_heads,
            d_ff,
            dropout,
            scales=(0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 24.0),
            slots_per_field=self.realtime_slots_per_field,
        )
        self.low_resampler = RegimeConditionedOptimalTransportResampler(
            d_model, d_ff, dropout, scales=(4.0, 8.0, 12.0, 24.0, 36.0, 48.0)
        )
        self.high_resampler = RegimeConditionedOptimalTransportResampler(
            d_model, d_ff, dropout, scales=(0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
        )
        self.realtime_innovation = NWPOrthogonalInnovation(d_model, d_ff, dropout)
        self.low_innovation = ConditionedDeconfoundedInnovation(d_model, d_ff, dropout)
        self.high_innovation = ConditionedDeconfoundedInnovation(d_model, d_ff, dropout)
        self.condition_norm = nn.LayerNorm(d_model * 8)
        condition_dim = d_model * 8
        self.realtime_injector = RealtimeAsynchronousResidualInjector(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            max_delta=max_delta * 0.5,
            scalar_dim=self.realtime_scalar_dim,
        )
        self.residual_decoder = Text2ResidualCorrectionDecoder(
            pred_len,
            d_model,
            d_ff,
            condition_dim,
            experts,
            basis_rank,
            dropout,
            self.text2_max_delta,
            TEXT2_SCALAR_DIM,
        )
        self.candidate_calibrator = HorizonWiseCandidateCalibrator(
            pred_len,
            len(ROUTER_CANDIDATE_NAMES),
        )
        self.text_router = TextConditionedNumericalHypothesisRouter(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            num_candidates=len(ROUTER_CANDIDATE_NAMES),
            text_scalar_dim=TEXT2_SCALAR_DIM,
            opportunity_dim=8,
            time_dim=5,
            max_route_delta=self.text2_max_delta * 0.70,
        )
        release_scalar_dim = 15 + 5 + TEXT2_SCALAR_DIM
        self.release_gate = CounterfactualEvidentialReleaseGate(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            scalar_dim=release_scalar_dim,
        )
        intervention_scalar_dim = 13 + 5 + TEXT2_SCALAR_DIM
        self.textual_intervention = TextualCausalResidualIntervention(
            pred_len=pred_len,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            max_delta=self.text2_max_delta,
            scalar_dim=intervention_scalar_dim,
        )
        self.risk_controller = nn.Sequential(
            nn.LayerNorm(d_model * 4 + 5),
            nn.Linear(d_model * 4 + 5, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, 1),
        )
        self.extreme_shock = FutureExtremeWeatherShockModule(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            prototype_count=max(experts + 2, 4),
            max_delta=self.text2_max_delta * 0.75,
            max_duration_hours=max(risk_horizon_hours * 2.0, 1.0),
        )
        self.fuzzy_extreme = FuzzyExtremeSemanticExtractor(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            fuzzy_sets=max(experts, 4),
        )
        self.bands = SmoothDCTBands(pred_len, cutoff_index=cutoff_index)
        self.risk_horizon_hours = float(risk_horizon_hours)
        if freeze_baseline:
            for parameter in self.baseline.parameters():
                parameter.requires_grad = False
            self.baseline.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_baseline:
            self.baseline.eval()
        return self

    def _baseline_forward(self, batch):
        if self.freeze_baseline:
            with torch.no_grad():
                return self.baseline(batch, return_hidden=True)
        return self.baseline(batch, return_hidden=True)

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = mask.unsqueeze(-1).to(x.dtype)
        return (x * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)

    @staticmethod
    def _horizon_text_coverage(
        future_offset: torch.Tensor,
        text_offset: torch.Tensor,
        text_mask: torch.Tensor,
        max_distance_hours: float,
    ) -> torch.Tensor:
        valid = text_mask.bool()
        distance = (future_offset.unsqueeze(-1) - text_offset.unsqueeze(1)).abs()
        distance = distance.masked_fill(~valid.unsqueeze(1), 1e6)
        min_distance = distance.min(dim=-1).values
        return (min_distance <= float(max_distance_hours)).to(future_offset.dtype)

    def forward(self, batch: Dict[str, torch.Tensor], return_parts: bool = False):
        baseline_prediction, hidden = self._baseline_forward(batch)
        horizon_state = hidden["fused_horizon_state"]
        if horizon_state.shape[-1] != self.d_model:
            raise RuntimeError(
                "fusion d_model must match the loaded numerical baseline hidden size: "
                f"baseline={horizon_state.shape[-1]}, fusion={self.d_model}"
            )
        nwp_regime, nwp_global = self.nwp_encoder(horizon_state, batch)
        response_state, response_global = self.response_encoder(horizon_state, batch)
        horizon_feature = _time_features(batch["future_offset_hours"])
        aux_step_delta = hidden["aux_prediction"] - hidden["ot_prediction"]
        past_step_delta = hidden["past_nwp_prediction"] - hidden["aux_prediction"]
        future_step_delta = hidden["stage1_prediction"] - hidden["past_nwp_prediction"]
        signed_numeric_opportunity = torch.stack(
            [aux_step_delta, past_step_delta, future_step_delta],
            dim=-1,
        )
        absolute_numeric_opportunity = signed_numeric_opportunity.abs()
        numeric_disagreement = absolute_numeric_opportunity.mean(dim=-1)
        numeric_opportunity = torch.tanh(
            numeric_disagreement / max(self.max_delta * 0.35, 1e-6)
        )
        daylight_prior = ((daylight_weight(batch["future_time_features"], 2.0) - 1.0)).clamp(0.0, 1.0)
        future_valid = batch["future_nwp_mask"].to(horizon_state.dtype)
        future_text_available = batch["text2_mask"].any(dim=1, keepdim=True).to(horizon_state.dtype)

        rt_tokens, rt_mask, rt_time, rt_global = self.realtime_text_encoder(
            batch["text_state_tokens"],
            batch["text_trend_tokens"],
            batch["text_var_tokens"],
            batch["text_state_token_mask"].bool(),
            batch["text_trend_token_mask"].bool(),
            batch["text_var_token_mask"].bool(),
            batch["text_state_sentence"],
            batch["text_trend_sentence"],
            batch["text_var_sentence"],
            batch["text1_mask"].bool(),
            batch["text1_offset_hours"],
        )
        rt_context, rt_weights, rt_alignment = self.realtime_aligner(
            horizon_state,
            nwp_regime,
            response_state,
            rt_tokens,
            batch["future_offset_hours"],
            rt_time,
            rt_mask.bool(),
        )

        rt_innovation, rt_explainable = self.realtime_innovation(
            rt_context, nwp_regime, response_state
        )
        zero_context = torch.zeros_like(horizon_state)
        zero_scalar = torch.zeros_like(baseline_prediction)
        zero_global = torch.zeros_like(nwp_global)
        if self.use_text2_correction:
            low_hour, high_hour, text_global = self.text_encoder(
                batch["text2_low"],
                batch["text2_high"],
                batch["text2_mask"].bool(),
                batch["text2_offset_hours"],
            )
            text_scalar = batch.get("text2_scalar")
            if text_scalar is None:
                text_scalar = torch.zeros(
                    batch["text2_low"].shape[0],
                    batch["text2_low"].shape[1],
                    TEXT2_SCALAR_DIM,
                    dtype=horizon_state.dtype,
                    device=horizon_state.device,
                )
            low_context, low_weights, low_transport = self.low_resampler(
                horizon_state,
                nwp_regime,
                response_state,
                low_hour,
                batch["future_offset_hours"],
                batch["text2_offset_hours"],
                batch["text2_mask"].bool(),
            )
            high_context, high_weights, high_transport = self.high_resampler(
                horizon_state,
                nwp_regime,
                response_state,
                high_hour,
                batch["future_offset_hours"],
                batch["text2_offset_hours"],
                batch["text2_mask"].bool(),
            )
            text_horizon_scalar = torch.matmul(high_weights, text_scalar.float())
            low_innovation, low_explainable, low_diag = self.low_innovation(
                low_context, nwp_regime, response_state, horizon_feature
            )
            high_innovation, high_explainable, high_diag = self.high_innovation(
                high_context, nwp_regime, response_state, horizon_feature
            )
            if not self.use_low_text2:
                low_innovation = torch.zeros_like(low_innovation)
                low_explainable = torch.zeros_like(low_explainable)
                low_diag = {
                    "condition_gate": torch.zeros_like(horizon_state[..., 0]),
                    "condition_overlap": torch.zeros_like(horizon_state[..., 0]),
                }
            if not self.use_high_text2:
                high_innovation = torch.zeros_like(high_innovation)
                high_explainable = torch.zeros_like(high_explainable)
                high_diag = {
                    "condition_gate": torch.zeros_like(horizon_state[..., 0]),
                    "condition_overlap": torch.zeros_like(horizon_state[..., 0]),
                }
        else:
            low_hour = torch.zeros(
                batch["text2_low"].shape[0],
                batch["text2_low"].shape[1],
                self.d_model,
                dtype=horizon_state.dtype,
                device=horizon_state.device,
            )
            high_hour = torch.zeros_like(low_hour)
            text_global = zero_global
            low_context = zero_context
            high_context = zero_context
            low_weights = torch.zeros(
                horizon_state.shape[0],
                horizon_state.shape[1],
                batch["text2_low"].shape[1],
                dtype=horizon_state.dtype,
                device=horizon_state.device,
            )
            high_weights = torch.zeros_like(low_weights)
            low_transport = {
                "expected_cost": horizon_state.new_tensor(0.0),
                "entropy": horizon_state.new_tensor(0.0),
            }
            high_transport = {
                "expected_cost": horizon_state.new_tensor(0.0),
                "entropy": horizon_state.new_tensor(0.0),
            }
            low_innovation = zero_context
            high_innovation = zero_context
            low_explainable = zero_context
            high_explainable = zero_context
            low_diag = {
                "condition_gate": zero_scalar,
                "condition_overlap": zero_scalar,
            }
            high_diag = {
                "condition_gate": zero_scalar,
                "condition_overlap": zero_scalar,
            }
            text_scalar = torch.zeros(
                batch["text2_low"].shape[0],
                batch["text2_low"].shape[1],
                TEXT2_SCALAR_DIM,
                dtype=horizon_state.dtype,
                device=horizon_state.device,
            )
            text_horizon_scalar = torch.zeros(
                horizon_state.shape[0],
                horizon_state.shape[1],
                TEXT2_SCALAR_DIM,
                dtype=horizon_state.dtype,
                device=horizon_state.device,
            )
        if not self.use_realtime_condition:
            rt_innovation = torch.zeros_like(rt_innovation)
            rt_global = torch.zeros_like(rt_global)

        has_future_text = batch["text2_mask"].any(dim=1, keepdim=True)
        has_realtime_text = batch["text1_mask"].any(dim=1, keepdim=True)
        has_text = has_future_text | has_realtime_text
        horizon_mask = has_text.to(horizon_state.dtype)
        future_mask = has_future_text.to(horizon_state.dtype)
        realtime_mask = has_realtime_text.to(horizon_state.dtype)
        realtime_scalar = torch.cat(
            [
                horizon_feature,
                numeric_opportunity.unsqueeze(-1),
                numeric_disagreement.unsqueeze(-1),
                daylight_prior.unsqueeze(-1),
                future_valid.unsqueeze(-1),
                signed_numeric_opportunity,
                absolute_numeric_opportunity,
            ],
            dim=-1,
        )
        realtime_delta, realtime_parts = self.realtime_injector(
            horizon_state,
            nwp_regime,
            response_state,
            rt_innovation,
            realtime_scalar,
            realtime_mask.expand(-1, horizon_state.shape[1]),
        )
        if not self.use_realtime_condition:
            realtime_delta = torch.zeros_like(realtime_delta)
            realtime_parts = {
                "realtime_confidence": torch.zeros_like(realtime_parts["realtime_confidence"]),
                "realtime_local": torch.zeros_like(realtime_parts["realtime_local"]),
                "realtime_opportunity": torch.zeros_like(realtime_parts["realtime_opportunity"]),
                "realtime_numeric_opportunity": torch.zeros_like(realtime_parts["realtime_numeric_opportunity"]),
                "realtime_direction_prior": torch.zeros_like(realtime_parts["realtime_direction_prior"]),
            }
        stage1_text_prediction = baseline_prediction + realtime_delta
        rt_global_innovation = rt_innovation.mean(dim=1)
        low_global = low_innovation.mean(dim=1)
        high_global = high_innovation.mean(dim=1)
        horizon_global = horizon_state.mean(dim=1)
        condition = self.condition_norm(
            torch.cat(
                [
                    horizon_global,
                    nwp_global,
                    response_global,
                    rt_global,
                    text_global,
                    rt_global_innovation,
                    low_global,
                    high_global,
                ],
                dim=-1,
            )
        )
        neutral_condition = self.condition_norm(
            torch.cat(
                [
                    horizon_global,
                    nwp_global,
                    response_global,
                    rt_global,
                    zero_global,
                    rt_global_innovation,
                    torch.zeros_like(low_global),
                    torch.zeros_like(high_global),
                ],
                dim=-1,
            )
        )

        signed_opportunity = torch.stack(
            [
                aux_step_delta,
                past_step_delta,
                future_step_delta,
                realtime_delta,
            ],
            dim=-1,
        )
        residual_opportunity = torch.cat(
            [signed_opportunity, signed_opportunity.abs()],
            dim=-1,
        ).detach()
        if self.use_text2_correction:
            risk_logits = self.risk_controller(
                torch.cat(
                    [
                        horizon_state,
                        high_innovation,
                        nwp_regime,
                        response_state,
                        horizon_feature,
                    ],
                    dim=-1,
                )
            ).squeeze(-1)
            text_event_hour = (
                0.58 * text_scalar[..., 2].float().clamp(0.0, 1.0)
                + 0.42 * text_scalar[..., 3].float().clamp(0.0, 1.0)
            ).clamp(0.0, 1.0)
            aligned_event_score = torch.sum(
                high_weights.detach() * text_event_hour.unsqueeze(1),
                dim=-1,
            ).clamp(0.0, 1.0)
            risk_weight = torch.sigmoid(risk_logits) * (0.20 + 0.80 * aligned_event_score)
            fuzzy_parts = self.fuzzy_extreme(
                horizon_state,
                nwp_regime,
                response_state,
                low_hour,
                high_hour,
                high_weights,
                batch["future_offset_hours"],
                batch["text2_mask"].bool(),
            )
            if not self.use_fuzzy_extreme:
                fuzzy_parts = {
                    "fuzzy_context": torch.zeros_like(fuzzy_parts["fuzzy_context"]),
                    "fuzzy_membership": torch.zeros_like(fuzzy_parts["fuzzy_membership"]),
                    "fuzzy_intensity": torch.zeros_like(fuzzy_parts["fuzzy_intensity"]),
                    "fuzzy_ambiguity": torch.zeros_like(fuzzy_parts["fuzzy_ambiguity"]),
                    "fuzzy_entropy": fuzzy_parts["fuzzy_entropy"].new_tensor(0.0),
                    "fuzzy_ambiguity_mean": fuzzy_parts["fuzzy_ambiguity_mean"].new_tensor(0.0),
                }
            shock_delta, shock_parts = self.extreme_shock(
                horizon_state,
                high_context,
                high_innovation,
                nwp_regime,
                response_state,
                low_hour,
                high_hour,
                high_weights,
                batch["future_offset_hours"],
                batch["text2_offset_hours"],
                batch["text2_mask"].bool(),
            )
            if not self.use_shock_evidence:
                shock_parts = {
                    **shock_parts,
                    "shock_derivative": torch.zeros_like(shock_parts["shock_derivative"]),
                    "extreme_event_risk": torch.zeros_like(shock_parts["extreme_event_risk"]),
                    "risk_onset": torch.zeros_like(shock_parts["risk_onset"]),
                    "risk_intensity": torch.zeros_like(shock_parts["risk_intensity"]),
                    "risk_duration": torch.zeros_like(shock_parts["risk_duration"]),
                    "risk_tv": shock_parts["risk_tv"].new_tensor(0.0),
                }
                shock_delta = torch.zeros_like(shock_delta)
            text_extreme_prior = text_horizon_scalar[..., 2].detach().clamp(0.0, 1.0)
            text_fuzzy_prior = text_horizon_scalar[..., 3].detach().clamp(0.0, 1.0)
            text_unstable_prior = (1.0 - text_horizon_scalar[..., 4].detach().clamp(0.0, 1.0))
            text_risk_focus = (
                0.45 * text_extreme_prior
                + 0.35 * text_fuzzy_prior
                + 0.20 * text_unstable_prior
            ).clamp(0.0, 1.0)
            high_risk_focus = (
                0.34 * risk_weight.detach()
                + 0.26 * fuzzy_parts["fuzzy_intensity"].detach()
                + 0.24 * text_risk_focus
                + 0.16 * aligned_event_score.detach()
            ).clamp(0.0, 1.0)
            high_risk_focus = (
                high_risk_focus
                * future_valid
                * future_text_available.expand_as(future_valid)
            )
        else:
            risk_weight = zero_scalar
            fuzzy_parts = {
                "fuzzy_context": zero_context,
                "fuzzy_membership": torch.zeros(
                    *zero_scalar.shape,
                    max(self.residual_decoder.experts, 4),
                    dtype=horizon_state.dtype,
                    device=horizon_state.device,
                ),
                "fuzzy_intensity": zero_scalar,
                "fuzzy_ambiguity": zero_scalar,
                "fuzzy_entropy": horizon_state.new_tensor(0.0),
                "fuzzy_ambiguity_mean": horizon_state.new_tensor(0.0),
            }
            shock_delta = zero_scalar
            shock_parts = {
                "shock_derivative": zero_scalar,
                "extreme_event_risk": zero_scalar,
                "risk_onset": torch.zeros(horizon_state.shape[0], dtype=horizon_state.dtype, device=horizon_state.device),
                "risk_intensity": torch.zeros(horizon_state.shape[0], dtype=horizon_state.dtype, device=horizon_state.device),
                "risk_duration": torch.zeros(horizon_state.shape[0], dtype=horizon_state.dtype, device=horizon_state.device),
                "risk_prototype_weights": torch.zeros(
                    horizon_state.shape[0],
                    self.extreme_shock.prototype_count,
                    dtype=horizon_state.dtype,
                    device=horizon_state.device,
                ),
                "risk_prototype_entropy": horizon_state.new_tensor(0.0),
                "risk_prototype_balance": horizon_state.new_tensor(0.0),
                "risk_tv": horizon_state.new_tensor(0.0),
            }
            high_risk_focus = zero_scalar
            aligned_event_score = zero_scalar
        low_text_coverage = self._horizon_text_coverage(
            batch["future_offset_hours"],
            batch["text2_offset_hours"],
            batch["text2_mask"],
            max_distance_hours=3.0,
        )
        high_text_coverage = self._horizon_text_coverage(
            batch["future_offset_hours"],
            batch["text2_offset_hours"],
            batch["text2_mask"],
            max_distance_hours=0.75,
        )
        text2_horizon_mask = torch.maximum(low_text_coverage, high_text_coverage)
        text2_horizon_mask = text2_horizon_mask * future_valid
        if self.use_text2_correction:
            correction, residual_parts = self.residual_decoder(
                horizon_state,
                nwp_regime,
                response_state,
                rt_innovation,
                low_innovation,
                high_innovation,
                fuzzy_parts["fuzzy_context"],
                shock_delta,
                shock_parts["shock_derivative"],
                risk_weight,
                fuzzy_parts["fuzzy_intensity"],
                fuzzy_parts["fuzzy_ambiguity"],
                residual_opportunity,
                horizon_feature,
                text_horizon_scalar,
                text2_horizon_mask,
                condition,
            )
        else:
            correction = torch.zeros_like(stage1_text_prediction)
            residual_parts = {
                "text2_residual_evidence": zero_context,
                "residual_confidence": zero_scalar,
                "residual_base_confidence": zero_scalar,
                "residual_opportunity_gate": zero_scalar,
                "residual_numeric_prior": zero_scalar,
                "residual_semantic_certainty": zero_scalar,
                "residual_direction_prior": zero_scalar,
                "residual_prior_correction": zero_scalar,
                "residual_local": zero_scalar,
                "residual_spectral": zero_scalar,
            }
        router_candidates = torch.stack(
            [
                stage1_text_prediction,
                hidden["chain_prediction"],
                hidden["nwp_prior_prediction"],
                hidden["periodic_prediction"],
            ],
            dim=-1,
        )
        calibrated_candidates, calibration_parts = self.candidate_calibrator(
            router_candidates,
            daylight_prior,
        )
        if self.use_text2_correction and self.use_text_router:
            router_parts = self.text_router(
                horizon_state,
                nwp_regime,
                response_state,
                rt_innovation,
                low_innovation,
                high_innovation,
                fuzzy_parts["fuzzy_context"],
                text_horizon_scalar,
                calibrated_candidates,
                residual_opportunity,
                horizon_feature,
                text2_horizon_mask,
            )
        else:
            uniform_prob = torch.zeros_like(router_candidates)
            uniform_prob[..., 0] = 1.0
            router_parts = {
                "router_logits": torch.zeros_like(router_candidates),
                "router_prob": uniform_prob,
                "router_hard_prob": uniform_prob,
                "router_gate": zero_scalar,
                "router_entropy_h": zero_scalar,
                "router_top_prob": torch.ones_like(zero_scalar),
                "route_prediction": stage1_text_prediction,
                "route_delta_unscaled": zero_scalar,
                "candidate_std": zero_scalar,
                "candidate_range": zero_scalar,
                "candidate_mean_abs": zero_scalar,
                "base_improvement_score": torch.zeros_like(router_candidates),
                "text_improvement_delta": torch.zeros_like(router_candidates),
                "improvement_score": torch.zeros_like(router_candidates),
                "semantic_release": zero_scalar,
                "positive_release": zero_scalar,
                "disagreement_release": zero_scalar,
            }
        router_stage_scale = torch.tanh(self.router_stage_scale)
        raw_route_delta = router_parts["route_delta_unscaled"]
        if self.training and self.use_text2_correction and self.text2_gradient_warmup > 0.0:
            route_delta = (
                router_stage_scale * raw_route_delta
                + self.text2_gradient_warmup
                * (raw_route_delta - raw_route_delta.detach())
            )
        else:
            route_delta = router_stage_scale * raw_route_delta
        text2_stage_scale = torch.tanh(self.text2_stage_scale)
        raw_text2_correction = correction
        if self.training and self.use_text2_correction and self.text2_gradient_warmup > 0.0:
            scaled_correction = (
                text2_stage_scale * raw_text2_correction
                + self.text2_gradient_warmup
                * (raw_text2_correction - raw_text2_correction.detach())
            )
        else:
            scaled_correction = text2_stage_scale * raw_text2_correction
        raw_low_delta, raw_high_delta = self.bands.split(scaled_correction)
        risk_gate = (
            (0.10 + 0.90 * high_risk_focus.clamp(0.0, 1.0))
            * residual_parts["residual_opportunity_gate"].detach().clamp(0.0, 1.0)
            * high_text_coverage
        ).clamp(0.0, 1.0)
        text_direction = (
            text_horizon_scalar[..., 1].detach().clamp(0.0, 1.0)
            - text_horizon_scalar[..., 0].detach().clamp(0.0, 1.0)
        ).clamp(-1.0, 1.0)
        numeric_direction = torch.tanh(
            residual_parts["residual_direction_prior"].detach()
            / max(self.text2_max_delta * 0.15, 1e-6)
        )
        ramp_direction = torch.tanh(0.65 * numeric_direction + 0.35 * text_direction)
        text_ramp_stage_scale = torch.tanh(self.text_ramp_stage_scale)
        text_ramp_gate = (
            high_text_coverage
            * future_valid
            * (0.20 + 0.80 * high_risk_focus.clamp(0.0, 1.0))
            * (0.25 + 0.75 * aligned_event_score.detach().clamp(0.0, 1.0))
            * residual_parts["residual_opportunity_gate"].detach().clamp(0.0, 1.0)
        ).clamp(0.0, 1.0)
        text_ramp_prior = (
            text_ramp_stage_scale
            * self.text2_max_delta
            * 0.18
            * text_ramp_gate
            * ramp_direction
        )
        text_ramp_prior = self.bands.project(text_ramp_prior, "high")
        text_ramp_prior = text_ramp_prior * high_text_coverage * future_valid
        text2_residual_proposal = (
            raw_low_delta * low_text_coverage
            + risk_gate * raw_high_delta
            + text_ramp_prior
        )
        decoder_release_gate = (
            residual_parts["residual_opportunity_gate"].clamp(0.0, 1.0)
            * text2_horizon_mask
        )
        semantic_router_gate = router_parts["router_gate"].clamp(0.0, 1.0)
        extreme_release_gate = high_risk_focus.clamp(0.0, 1.0) * text2_horizon_mask
        if self.use_text2_correction:
            if self.use_text_router:
                prior_release_gate = (
                    0.50 * semantic_router_gate
                    + 0.35 * decoder_release_gate
                    + 0.15 * extreme_release_gate
                ).clamp(0.0, 1.0)
            else:
                prior_release_gate = decoder_release_gate
            prior_release_gate = prior_release_gate * text2_horizon_mask
        else:
            prior_release_gate = torch.zeros_like(text2_residual_proposal)

        zero_innovation = torch.zeros_like(low_innovation)
        if self.use_text2_correction:
            neutral_shock, _ = self.extreme_shock(
                horizon_state,
                torch.zeros_like(high_context),
                zero_innovation,
                nwp_regime,
                response_state,
                torch.zeros_like(low_hour),
                torch.zeros_like(high_hour),
                torch.zeros_like(high_weights),
                batch["future_offset_hours"],
                batch["text2_offset_hours"],
                batch["text2_mask"].bool(),
            )
            neutral_correction, neutral_residual_parts = self.residual_decoder(
                horizon_state,
                nwp_regime,
                response_state,
                zero_innovation,
                zero_innovation,
                zero_innovation,
                torch.zeros_like(fuzzy_parts["fuzzy_context"]),
                neutral_shock,
                torch.zeros_like(shock_parts["shock_derivative"]),
                torch.zeros_like(risk_weight),
                torch.zeros_like(fuzzy_parts["fuzzy_intensity"]),
                torch.zeros_like(fuzzy_parts["fuzzy_ambiguity"]),
                torch.zeros_like(residual_opportunity),
                horizon_feature,
                torch.zeros_like(text_horizon_scalar),
                text2_horizon_mask,
                neutral_condition,
            )
        else:
            neutral_correction = torch.zeros_like(text2_residual_proposal)
            neutral_residual_parts = {"residual_opportunity_gate": torch.zeros_like(text2_residual_proposal)}
        if self.use_text2_correction:
            neutral_scaled = text2_stage_scale * neutral_correction
            neutral_low, neutral_high = self.bands.split(neutral_scaled)
            neutral_risk_gate = (
                0.10
                * neutral_residual_parts["residual_opportunity_gate"].detach().clamp(0.0, 1.0)
                * high_text_coverage
            ).clamp(0.0, 1.0)
            neutral_proposal = neutral_low * low_text_coverage + neutral_risk_gate * neutral_high
            neutral_release_gate = (
                neutral_residual_parts["residual_opportunity_gate"].clamp(0.0, 1.0)
                * text2_horizon_mask
            )
            neutral_correction = neutral_release_gate * neutral_proposal
        else:
            neutral_proposal = torch.zeros_like(text2_residual_proposal)
            neutral_release_gate = torch.zeros_like(text2_residual_proposal)
        if self.use_text2_correction and self.use_text_router:
            neutral_router_parts = self.text_router(
                horizon_state,
                nwp_regime,
                response_state,
                rt_innovation,
                zero_innovation,
                zero_innovation,
                torch.zeros_like(fuzzy_parts["fuzzy_context"]),
                torch.zeros_like(text_horizon_scalar),
                calibrated_candidates,
                residual_opportunity,
                horizon_feature,
                text2_horizon_mask,
            )
            neutral_route_delta = router_stage_scale * neutral_router_parts["route_delta_unscaled"]
        else:
            neutral_route_delta = torch.zeros_like(text2_residual_proposal)

        if self.use_text2_correction:
            safe_release_gate, release_parts = self.release_gate(
                horizon_state,
                nwp_regime,
                response_state,
                rt_innovation,
                residual_parts["text2_residual_evidence"],
                fuzzy_parts["fuzzy_context"],
                text2_residual_proposal,
                neutral_proposal,
                prior_release_gate,
                decoder_release_gate,
                semantic_router_gate,
                extreme_release_gate,
                route_delta,
                router_parts["candidate_range"],
                residual_parts["residual_numeric_prior"],
                residual_parts["residual_semantic_certainty"],
                fuzzy_parts["fuzzy_ambiguity"],
                aligned_event_score,
                text_ramp_gate,
                horizon_feature,
                text_horizon_scalar,
                text2_horizon_mask,
            )
            scalar_extreme_budget = (
                text_horizon_scalar[..., 2].detach().clamp(0.0, 1.0)
                if text_horizon_scalar.shape[-1] > 2
                else torch.zeros_like(safe_release_gate)
            )
            scalar_fuzzy_budget = (
                text_horizon_scalar[..., 3].detach().clamp(0.0, 1.0)
                if text_horizon_scalar.shape[-1] > 3
                else torch.zeros_like(safe_release_gate)
            )
            scalar_stable_budget = (
                text_horizon_scalar[..., 4].detach().clamp(0.0, 1.0)
                if text_horizon_scalar.shape[-1] > 4
                else torch.zeros_like(safe_release_gate)
            )
            event_nwp_disagreement_budget = torch.tanh(
                numeric_disagreement.detach() / max(self.max_delta * 0.25, 1e-6)
            ).clamp(0.0, 1.0)
            event_pressure_budget = (
                0.30 * high_risk_focus.detach().clamp(0.0, 1.0)
                + 0.20 * aligned_event_score.detach().clamp(0.0, 1.0)
                + 0.18 * fuzzy_parts["fuzzy_intensity"].detach().clamp(0.0, 1.0)
                + 0.14 * text_ramp_gate.detach().clamp(0.0, 1.0)
                + 0.10 * scalar_extreme_budget
                + 0.05 * scalar_fuzzy_budget
                + 0.03 * event_nwp_disagreement_budget
            ).clamp(0.0, 1.0)
            stable_pressure_budget = (
                0.48 * scalar_stable_budget
                + 0.20 * (1.0 - high_risk_focus.detach().clamp(0.0, 1.0))
                + 0.14 * (1.0 - aligned_event_score.detach().clamp(0.0, 1.0))
                + 0.10 * (1.0 - fuzzy_parts["fuzzy_intensity"].detach().clamp(0.0, 1.0))
                + 0.08 * (1.0 - event_nwp_disagreement_budget)
            ).clamp(0.0, 1.0)
            event_margin_budget = torch.sigmoid(5.5 * (event_pressure_budget - stable_pressure_budget + 0.05))
            sparse_floor_budget = (0.015 + 0.055 * event_pressure_budget).clamp(0.0, 0.08)
            event_release_budget = (
                sparse_floor_budget
                + (1.0 - sparse_floor_budget)
                * event_margin_budget
                * (0.35 + 0.65 * event_pressure_budget)
            ).clamp(0.0, 1.0)
            event_release_budget = (
                event_release_budget * text2_horizon_mask * future_valid
            ).clamp(0.0, 1.0)
            safe_release_gate = (safe_release_gate * event_release_budget).clamp(0.0, 1.0)
        else:
            safe_release_gate = torch.zeros_like(text2_residual_proposal)
            event_release_budget = torch.zeros_like(text2_residual_proposal)
            release_parts = {
                "release_alpha": torch.ones_like(text2_residual_proposal),
                "release_beta": torch.ones_like(text2_residual_proposal),
                "release_evidence_mean": torch.zeros_like(text2_residual_proposal),
                "release_uncertainty": torch.ones_like(text2_residual_proposal),
                "release_cap": torch.zeros_like(text2_residual_proposal),
                "release_logit_offset": torch.zeros_like(text2_residual_proposal),
                "release_agreement": torch.zeros_like(text2_residual_proposal),
                "release_route_agreement": torch.zeros_like(text2_residual_proposal),
                "release_counterfactual_gain": torch.zeros_like(text2_residual_proposal),
                "release_numeric_uncertainty": torch.zeros_like(text2_residual_proposal),
            }
        if self.use_text2_correction:
            released_treatment, intervention_parts = self.textual_intervention(
                horizon_state,
                nwp_regime,
                response_state,
                rt_innovation,
                residual_parts["text2_residual_evidence"],
                fuzzy_parts["fuzzy_context"],
                text2_residual_proposal,
                neutral_proposal,
                safe_release_gate,
                prior_release_gate,
                extreme_release_gate,
                text_ramp_gate,
                residual_parts["residual_numeric_prior"],
                fuzzy_parts["fuzzy_ambiguity"],
                router_parts["candidate_range"],
                horizon_feature,
                text_horizon_scalar,
                text2_horizon_mask,
            )
            proposal_release = safe_release_gate * text2_residual_proposal
            scalar_down = text_horizon_scalar[..., 0].detach().clamp(0.0, 1.0)
            scalar_up = text_horizon_scalar[..., 1].detach().clamp(0.0, 1.0)
            scalar_extreme = text_horizon_scalar[..., 2].detach().clamp(0.0, 1.0)
            scalar_fuzzy = text_horizon_scalar[..., 3].detach().clamp(0.0, 1.0)
            scalar_stable = text_horizon_scalar[..., 4].detach().clamp(0.0, 1.0)
            scalar_conf_high = (
                text_horizon_scalar[..., 8].detach().clamp(0.0, 1.0)
                if text_horizon_scalar.shape[-1] > 8
                else torch.zeros_like(scalar_extreme)
            )
            scalar_conf_med = (
                text_horizon_scalar[..., 9].detach().clamp(0.0, 1.0)
                if text_horizon_scalar.shape[-1] > 9
                else torch.zeros_like(scalar_extreme)
            )
            text_scalar_confidence = (scalar_conf_high + 0.5 * scalar_conf_med).clamp(0.0, 1.0)
            text_scalar_direction = (scalar_up - scalar_down).clamp(-1.0, 1.0)
            text_scalar_risk = (
                0.42 * scalar_extreme
                + 0.30 * scalar_fuzzy
                + 0.18 * (1.0 - scalar_stable)
                + 0.10 * text_scalar_confidence
            ).clamp(0.0, 1.0)
            fuzzy_intensity_evidence = fuzzy_parts["fuzzy_intensity"].detach().clamp(0.0, 1.0)
            fuzzy_ambiguity_evidence = fuzzy_parts["fuzzy_ambiguity"].detach().clamp(0.0, 1.0)
            shock_risk_evidence = shock_parts["extreme_event_risk"].detach().clamp(0.0, 1.0)
            shock_direction = torch.tanh(
                shock_parts["shock_derivative"].detach() / max(self.text2_max_delta * 0.12, 1e-6)
            )
            realtime_evidence = (
                realtime_parts["realtime_confidence"].detach().clamp(0.0, 1.0)
                * realtime_parts["realtime_opportunity"].detach().clamp(0.0, 1.0)
                * realtime_mask
            ).clamp(0.0, 1.0)
            nwp_disagreement_evidence = torch.tanh(
                numeric_disagreement.detach() / max(self.max_delta * 0.25, 1e-6)
            ).clamp(0.0, 1.0)
            dataset_evidence_release_gate = (
                text2_horizon_mask
                * future_valid
                * (0.16 + 0.28 * safe_release_gate.detach())
                * (
                    0.18
                    + 0.24 * text_scalar_risk
                    + 0.18 * aligned_event_score.detach().clamp(0.0, 1.0)
                    + 0.16 * fuzzy_parts["fuzzy_intensity"].detach().clamp(0.0, 1.0)
                    + 0.14 * nwp_disagreement_evidence
                    + 0.10 * realtime_evidence
                )
            ).clamp(0.0, 1.0)
            dataset_direction = torch.tanh(
                0.42 * text_scalar_direction
                + 0.28 * ramp_direction.detach()
                + 0.18 * torch.tanh(realtime_delta.detach() / max(self.max_delta * 0.25, 1e-6))
                + 0.12 * torch.tanh(
                    residual_parts["residual_direction_prior"].detach()
                    / max(self.text2_max_delta * 0.15, 1e-6)
                )
            )
            semantic_risk_release_gate = (
                text2_horizon_mask
                * future_valid
                * (
                    0.22 * text_scalar_risk
                    + 0.20 * fuzzy_intensity_evidence
                    + 0.18 * shock_risk_evidence
                    + 0.16 * aligned_event_score.detach().clamp(0.0, 1.0)
                    + 0.14 * nwp_disagreement_evidence
                    + 0.10 * realtime_evidence
                )
                * (0.35 + 0.65 * safe_release_gate.detach().clamp(0.0, 1.0))
                * (1.0 - 0.35 * fuzzy_ambiguity_evidence)
            ).clamp(0.0, 1.0)
            extreme_direction = torch.tanh(
                0.35 * dataset_direction
                + 0.25 * ramp_direction.detach()
                + 0.25 * shock_direction
                + 0.15 * text_scalar_direction
            )
            semantic_signed_risk_gate = (
                semantic_risk_release_gate * extreme_direction
            ).clamp(-1.0, 1.0)
            ramp_signed_risk_gate = (
                text2_horizon_mask
                * future_valid
                * (0.35 + 0.65 * text_scalar_risk)
                * (0.45 + 0.55 * fuzzy_intensity_evidence)
                * ramp_direction.detach()
            ).clamp(-1.0, 1.0)
            shock_signed_risk_gate = (
                text2_horizon_mask
                * future_valid
                * (0.30 + 0.70 * shock_risk_evidence)
                * shock_direction
            ).clamp(-1.0, 1.0)
            text_scalar_signed_risk_gate = (
                text2_horizon_mask
                * future_valid
                * text_scalar_risk
                * text_scalar_confidence.clamp(0.10, 1.0)
                * text_scalar_direction
            ).clamp(-1.0, 1.0)
            observable_extreme_gate = (
                text2_horizon_mask
                * future_valid
                * (
                    0.24 * text_scalar_risk
                    + 0.20 * fuzzy_intensity_evidence
                    + 0.18 * aligned_event_score.detach().clamp(0.0, 1.0)
                    + 0.16 * nwp_disagreement_evidence
                    + 0.12 * realtime_evidence
                    + 0.10 * shock_risk_evidence
                )
            ).clamp(0.0, 1.0)
            observable_stable_gate = (
                text2_horizon_mask
                * future_valid
                * (1.0 - observable_extreme_gate)
                * (0.45 + 0.55 * scalar_stable)
            ).clamp(0.0, 1.0)
            event_pressure = (
                0.32 * observable_extreme_gate
                + 0.18 * semantic_risk_release_gate
                + 0.16 * ramp_signed_risk_gate.abs()
                + 0.14 * shock_signed_risk_gate.abs()
                + 0.10 * nwp_disagreement_evidence
                + 0.10 * realtime_evidence
            ).clamp(0.0, 1.0)
            stable_pressure = (
                0.44 * observable_stable_gate
                + 0.22 * scalar_stable
                + 0.14 * (1.0 - text_scalar_risk)
                + 0.10 * (1.0 - fuzzy_intensity_evidence)
                + 0.10 * (1.0 - nwp_disagreement_evidence)
            ).clamp(0.0, 1.0) * text2_horizon_mask * future_valid
            event_stability_margin_gate = torch.sigmoid(
                5.0 * (event_pressure - stable_pressure + 0.04)
            ) * text2_horizon_mask * future_valid
            competitive_event_gate = (
                event_stability_margin_gate
                * (0.35 + 0.65 * event_pressure)
                * (0.30 + 0.70 * safe_release_gate.detach().clamp(0.0, 1.0))
            ).clamp(0.0, 1.0)
            competitive_stable_gate = (
                (1.0 - event_stability_margin_gate)
                * observable_stable_gate
                * (0.20 + 0.80 * safe_release_gate.detach().clamp(0.0, 1.0))
            ).clamp(0.0, 0.55)
            dataset_evidence_delta = (
                self.text2_max_delta
                * 0.08
                * dataset_evidence_release_gate
                * dataset_direction
            )
            dataset_evidence_high_delta = self.bands.project(dataset_evidence_delta, "high")
            dataset_evidence_low_delta = self.bands.project(dataset_evidence_delta, "low")
            extreme_weather_evidence_delta = self.bands.project(
                self.text2_max_delta
                * 0.13
                * semantic_risk_release_gate
                * extreme_direction,
                "high",
            )
            fuzzy_ramp_evidence_delta = self.bands.project(
                self.text2_max_delta
                * 0.10
                * text2_horizon_mask
                * future_valid
                * fuzzy_intensity_evidence
                * text_scalar_risk
                * ramp_direction.detach(),
                "high",
            )
            shock_direction_delta = self.bands.project(
                self.text2_max_delta
                * 0.08
                * text2_horizon_mask
                * future_valid
                * shock_risk_evidence
                * shock_direction,
                "high",
            )
            text_scalar_direction_delta = (
                self.text2_max_delta
                * 0.05
                * text2_horizon_mask
                * future_valid
                * text_scalar_confidence
                * text_scalar_direction
            )
            text_scalar_risk_delta = (
                self.text2_max_delta
                * 0.05
                * text2_horizon_mask
                * future_valid
                * text_scalar_risk
                * ramp_direction.detach()
            )
            realtime_evidence_delta = (
                self.text2_max_delta
                * 0.04
                * realtime_evidence
                * torch.tanh(realtime_delta.detach() / max(self.max_delta * 0.25, 1e-6))
            )
            nwp_disagreement_delta = (
                self.text2_max_delta
                * 0.04
                * text2_horizon_mask
                * future_valid
                * nwp_disagreement_evidence
                * torch.tanh(
                    residual_parts["residual_direction_prior"].detach()
                    / max(self.text2_max_delta * 0.15, 1e-6)
                )
            )
            extreme_counterfactual_delta = observable_extreme_gate * (
                text2_residual_proposal - neutral_proposal.detach()
            )
            extreme_proposal_delta = observable_extreme_gate * text2_residual_proposal
            extreme_dataset_delta = observable_extreme_gate * dataset_evidence_delta
            extreme_realtime_delta = observable_extreme_gate * realtime_evidence_delta
            extreme_nwp_disagreement_delta = observable_extreme_gate * nwp_disagreement_delta
            extreme_ramp_prior_delta = observable_extreme_gate * text_ramp_prior
            stable_proposal_delta = observable_stable_gate * text2_residual_proposal
            stable_dataset_delta = observable_stable_gate * dataset_evidence_delta
            event_budget_basis = (
                0.18
                + 0.34 * observable_extreme_gate.detach()
                + 0.20 * semantic_risk_release_gate.detach()
                + 0.16 * ramp_signed_risk_gate.detach().abs()
                + 0.12 * shock_signed_risk_gate.detach().abs()
            ).clamp(0.0, 1.0) * text2_horizon_mask * future_valid
            event_direction_basis = torch.tanh(
                0.34 * extreme_direction.detach()
                + 0.26 * ramp_direction.detach()
                + 0.20 * shock_direction.detach()
                + 0.20 * text_scalar_direction.detach()
            )
            event_counterfactual_basis_delta = self.bands.project(
                event_budget_basis
                * (text2_residual_proposal - neutral_proposal.detach()),
                "high",
            )
            event_proposal_basis_delta = self.bands.project(
                event_budget_basis * text2_residual_proposal,
                "high",
            )
            event_dataset_basis_delta = self.bands.project(
                event_budget_basis * dataset_evidence_delta,
                "high",
            )
            event_ramp_basis_delta = self.bands.project(
                self.text2_max_delta
                * 0.14
                * event_budget_basis
                * event_direction_basis,
                "high",
            )
            event_realtime_basis_delta = self.bands.project(
                event_budget_basis * realtime_evidence_delta,
                "high",
            )
            event_nwp_disagreement_basis_delta = self.bands.project(
                event_budget_basis * nwp_disagreement_delta,
                "high",
            )
            normal_trend_basis_delta = self.bands.project(
                (1.0 - event_budget_basis)
                * observable_stable_gate
                * intervention_parts["trend_treatment_delta"],
                "low",
            )
            competitive_event_treatment_delta = self.bands.project(
                competitive_event_gate * intervention_parts["event_treatment_delta"],
                "high",
            )
            competitive_trend_treatment_delta = self.bands.project(
                competitive_stable_gate * intervention_parts["trend_treatment_delta"],
                "low",
            )
            correction = (
                proposal_release
                + 0.16 * competitive_event_treatment_delta
                + 0.06 * competitive_trend_treatment_delta
            ) * text2_horizon_mask * future_valid
            if self.text2_max_delta > 0:
                correction = self.text2_max_delta * torch.tanh(
                    correction / max(self.text2_max_delta, 1e-6)
                )
            intervention_parts["released_text2_treatment"] = released_treatment
            intervention_parts["text2_intervention"] = correction
            intervention_parts["proposal_release_delta"] = proposal_release
            intervention_parts["dataset_evidence_delta"] = dataset_evidence_delta
            intervention_parts["dataset_evidence_high_delta"] = dataset_evidence_high_delta
            intervention_parts["dataset_evidence_low_delta"] = dataset_evidence_low_delta
            intervention_parts["dataset_evidence_release_gate"] = dataset_evidence_release_gate
            intervention_parts["text_scalar_direction_delta"] = text_scalar_direction_delta
            intervention_parts["text_scalar_risk_delta"] = text_scalar_risk_delta
            intervention_parts["realtime_evidence_delta"] = realtime_evidence_delta
            intervention_parts["nwp_disagreement_delta"] = nwp_disagreement_delta
            intervention_parts["extreme_weather_evidence_delta"] = extreme_weather_evidence_delta
            intervention_parts["fuzzy_ramp_evidence_delta"] = fuzzy_ramp_evidence_delta
            intervention_parts["shock_direction_delta"] = shock_direction_delta
            intervention_parts["semantic_risk_release_gate"] = semantic_risk_release_gate
            intervention_parts["semantic_signed_risk_gate"] = semantic_signed_risk_gate
            intervention_parts["ramp_signed_risk_gate"] = ramp_signed_risk_gate
            intervention_parts["shock_signed_risk_gate"] = shock_signed_risk_gate
            intervention_parts["text_scalar_signed_risk_gate"] = text_scalar_signed_risk_gate
            intervention_parts["extreme_counterfactual_delta"] = extreme_counterfactual_delta
            intervention_parts["extreme_proposal_delta"] = extreme_proposal_delta
            intervention_parts["extreme_dataset_delta"] = extreme_dataset_delta
            intervention_parts["extreme_realtime_delta"] = extreme_realtime_delta
            intervention_parts["extreme_nwp_disagreement_delta"] = extreme_nwp_disagreement_delta
            intervention_parts["extreme_ramp_prior_delta"] = extreme_ramp_prior_delta
            intervention_parts["event_counterfactual_basis_delta"] = event_counterfactual_basis_delta
            intervention_parts["event_proposal_basis_delta"] = event_proposal_basis_delta
            intervention_parts["event_dataset_basis_delta"] = event_dataset_basis_delta
            intervention_parts["event_ramp_basis_delta"] = event_ramp_basis_delta
            intervention_parts["event_realtime_basis_delta"] = event_realtime_basis_delta
            intervention_parts["event_nwp_disagreement_basis_delta"] = event_nwp_disagreement_basis_delta
            intervention_parts["normal_trend_basis_delta"] = normal_trend_basis_delta
            intervention_parts["stable_proposal_delta"] = stable_proposal_delta
            intervention_parts["stable_dataset_delta"] = stable_dataset_delta
            intervention_parts["observable_extreme_gate"] = observable_extreme_gate
            intervention_parts["observable_stable_gate"] = observable_stable_gate
            intervention_parts["competitive_event_gate"] = competitive_event_gate
            intervention_parts["competitive_stable_gate"] = competitive_stable_gate
            intervention_parts["competitive_event_treatment_delta"] = competitive_event_treatment_delta
            intervention_parts["competitive_trend_treatment_delta"] = competitive_trend_treatment_delta
            intervention_parts["event_stability_margin_gate"] = event_stability_margin_gate
            intervention_parts["future_daylight_gate"] = daylight_prior
        else:
            correction = torch.zeros_like(text2_residual_proposal)
            intervention_parts = {
                "text2_treatment_effect": torch.zeros_like(text2_residual_proposal),
                "counterfactual_gap_delta": torch.zeros_like(text2_residual_proposal),
                "treatment_low_delta": torch.zeros_like(text2_residual_proposal),
                "treatment_high_delta": torch.zeros_like(text2_residual_proposal),
                "released_text2_treatment": torch.zeros_like(text2_residual_proposal),
                "released_treatment_low": torch.zeros_like(text2_residual_proposal),
                "released_treatment_high": torch.zeros_like(text2_residual_proposal),
                "event_treatment_delta": torch.zeros_like(text2_residual_proposal),
                "trend_treatment_delta": torch.zeros_like(text2_residual_proposal),
                "treatment_low_release": torch.zeros_like(text2_residual_proposal),
                "treatment_high_release": torch.zeros_like(text2_residual_proposal),
                "treatment_effect_scale": torch.ones_like(text2_residual_proposal),
                "treatment_event_bias": torch.zeros_like(text2_residual_proposal),
                "treatment_augmented_proposal": torch.zeros_like(text2_residual_proposal),
                "treatment_proposal_mix": torch.zeros_like(text2_residual_proposal),
                "proposal_release_delta": torch.zeros_like(text2_residual_proposal),
                "dataset_evidence_delta": torch.zeros_like(text2_residual_proposal),
                "dataset_evidence_high_delta": torch.zeros_like(text2_residual_proposal),
                "dataset_evidence_low_delta": torch.zeros_like(text2_residual_proposal),
                "dataset_evidence_release_gate": torch.zeros_like(text2_residual_proposal),
                "text_scalar_direction_delta": torch.zeros_like(text2_residual_proposal),
                "text_scalar_risk_delta": torch.zeros_like(text2_residual_proposal),
                "realtime_evidence_delta": torch.zeros_like(text2_residual_proposal),
                "nwp_disagreement_delta": torch.zeros_like(text2_residual_proposal),
                "extreme_weather_evidence_delta": torch.zeros_like(text2_residual_proposal),
                "fuzzy_ramp_evidence_delta": torch.zeros_like(text2_residual_proposal),
                "shock_direction_delta": torch.zeros_like(text2_residual_proposal),
                "semantic_risk_release_gate": torch.zeros_like(text2_residual_proposal),
                "semantic_signed_risk_gate": torch.zeros_like(text2_residual_proposal),
                "ramp_signed_risk_gate": torch.zeros_like(text2_residual_proposal),
                "shock_signed_risk_gate": torch.zeros_like(text2_residual_proposal),
                "text_scalar_signed_risk_gate": torch.zeros_like(text2_residual_proposal),
                "extreme_counterfactual_delta": torch.zeros_like(text2_residual_proposal),
                "extreme_proposal_delta": torch.zeros_like(text2_residual_proposal),
                "extreme_dataset_delta": torch.zeros_like(text2_residual_proposal),
                "extreme_realtime_delta": torch.zeros_like(text2_residual_proposal),
                "extreme_nwp_disagreement_delta": torch.zeros_like(text2_residual_proposal),
                "extreme_ramp_prior_delta": torch.zeros_like(text2_residual_proposal),
                "event_counterfactual_basis_delta": torch.zeros_like(text2_residual_proposal),
                "event_proposal_basis_delta": torch.zeros_like(text2_residual_proposal),
                "event_dataset_basis_delta": torch.zeros_like(text2_residual_proposal),
                "event_ramp_basis_delta": torch.zeros_like(text2_residual_proposal),
                "event_realtime_basis_delta": torch.zeros_like(text2_residual_proposal),
                "event_nwp_disagreement_basis_delta": torch.zeros_like(text2_residual_proposal),
                "normal_trend_basis_delta": torch.zeros_like(text2_residual_proposal),
                "stable_proposal_delta": torch.zeros_like(text2_residual_proposal),
                "stable_dataset_delta": torch.zeros_like(text2_residual_proposal),
                "observable_extreme_gate": torch.zeros_like(text2_residual_proposal),
                "observable_stable_gate": torch.zeros_like(text2_residual_proposal),
                "competitive_event_gate": torch.zeros_like(text2_residual_proposal),
                "competitive_stable_gate": torch.zeros_like(text2_residual_proposal),
                "competitive_event_treatment_delta": torch.zeros_like(text2_residual_proposal),
                "competitive_trend_treatment_delta": torch.zeros_like(text2_residual_proposal),
                "event_stability_margin_gate": torch.zeros_like(text2_residual_proposal),
                "future_daylight_gate": torch.zeros_like(text2_residual_proposal),
                "treatment_event_focus": torch.zeros_like(text2_residual_proposal),
                "treatment_trend_focus": torch.zeros_like(text2_residual_proposal),
            }
        prediction = stage1_text_prediction + correction
        route_prediction_delta = router_parts["route_prediction"] - stage1_text_prediction
        low_delta, high_delta = self.bands.split(correction)
        rt_delta = high_delta * realtime_mask
        evidence_high = high_delta
        high_delta = evidence_high
        spectral_high_delta = evidence_high - shock_delta.detach()

        parts = {
            "stage1_prediction": baseline_prediction,
            "stage1_text_prediction": stage1_text_prediction,
            "baseline_prediction": baseline_prediction,
            "ot_only_prediction": hidden["ot_only_prediction"],
            "ot_prediction": hidden["ot_prediction"],
            "aux_prediction": hidden["aux_prediction"],
            "past_nwp_prediction": hidden["past_nwp_prediction"],
            "chain_prediction": hidden["chain_prediction"],
            "pre_periodic_prediction": hidden["pre_periodic_prediction"],
            "nwp_prior_prediction": hidden["nwp_prior_prediction"],
            "periodic_prediction": hidden["periodic_prediction"],
            "text2_stage_scale": text2_stage_scale,
            "router_stage_scale": router_stage_scale,
            "text_ramp_stage_scale": text_ramp_stage_scale,
            "raw_text2_correction": raw_text2_correction,
            "scaled_text2_correction": scaled_correction,
            "text2_residual_proposal": text2_residual_proposal,
            "prior_release_gate": prior_release_gate,
            "safe_release_gate": safe_release_gate,
            "event_release_budget": event_release_budget,
            "decoder_release_gate": decoder_release_gate,
            "semantic_router_gate": semantic_router_gate,
            "extreme_release_gate": extreme_release_gate,
            "neutral_release_gate": neutral_release_gate,
            "neutral_text2_proposal": neutral_proposal,
            **intervention_parts,
            **release_parts,
            "text_ramp_prior": text_ramp_prior,
            "text_ramp_gate": text_ramp_gate,
            "text_ramp_direction": ramp_direction,
            "risk_gate": risk_gate,
            "aligned_event_score": aligned_event_score,
            "low_text_coverage": low_text_coverage,
            "high_text_coverage": high_text_coverage,
            "router_candidates": router_candidates,
            "calibrated_candidates": calibrated_candidates,
            "candidate_cal_scale": calibration_parts["candidate_cal_scale"],
            "candidate_cal_bias": calibration_parts["candidate_cal_bias"],
            "router_candidate_names": ROUTER_CANDIDATE_NAMES,
            "router_logits": router_parts["router_logits"],
            "router_prob": router_parts["router_prob"],
            "router_hard_prob": router_parts["router_hard_prob"],
            "router_gate": router_parts["router_gate"],
            "router_entropy_h": router_parts["router_entropy_h"],
            "router_top_prob": router_parts["router_top_prob"],
            "route_prediction": router_parts["route_prediction"],
            "route_prediction_delta": route_prediction_delta,
            "route_delta_unscaled": raw_route_delta,
            "route_delta": route_delta,
            "text2_intervention": correction,
            "neutral_route_delta": neutral_route_delta,
            "neutral_text2_intervention": neutral_correction,
            "candidate_std": router_parts["candidate_std"],
            "candidate_range": router_parts["candidate_range"],
            "candidate_mean_abs": router_parts["candidate_mean_abs"],
            "base_improvement_score": router_parts["base_improvement_score"],
            "text_improvement_delta": router_parts["text_improvement_delta"],
            "improvement_score": router_parts["improvement_score"],
            "semantic_release": router_parts["semantic_release"],
            "positive_release": router_parts["positive_release"],
            "disagreement_release": router_parts["disagreement_release"],
            "realtime_delta": realtime_delta,
            "realtime_confidence": realtime_parts["realtime_confidence"],
            "realtime_local": realtime_parts["realtime_local"],
            "realtime_opportunity": realtime_parts["realtime_opportunity"],
            "realtime_numeric_opportunity": realtime_parts["realtime_numeric_opportunity"],
            "realtime_direction_prior": realtime_parts["realtime_direction_prior"],
            "realtime_scalar": realtime_scalar,
            "correction": correction,
            "rt_delta": rt_delta,
            "low_delta": low_delta,
            "high_delta": high_delta,
            "spectral_high_delta": spectral_high_delta,
            "shock_delta": shock_delta,
            "shock_derivative": shock_parts["shock_derivative"],
            "extreme_event_risk": shock_parts["extreme_event_risk"],
            "risk_onset": shock_parts["risk_onset"],
            "risk_intensity": shock_parts["risk_intensity"],
            "risk_duration": shock_parts["risk_duration"],
            "risk_prototype_weights": shock_parts["risk_prototype_weights"],
            "risk_prototype_entropy": shock_parts["risk_prototype_entropy"],
            "risk_prototype_balance": shock_parts["risk_prototype_balance"],
            "risk_tv": shock_parts["risk_tv"],
            "high_risk_focus": high_risk_focus,
            "text2_residual_evidence": residual_parts["text2_residual_evidence"],
            "text2_scalar": text_scalar,
            "text2_horizon_scalar": text_horizon_scalar,
            "residual_confidence": residual_parts["residual_confidence"],
            "residual_base_confidence": residual_parts["residual_base_confidence"],
            "residual_opportunity_gate": residual_parts["residual_opportunity_gate"],
            "residual_numeric_prior": residual_parts["residual_numeric_prior"],
            "residual_semantic_certainty": residual_parts["residual_semantic_certainty"],
            "residual_direction_prior": residual_parts["residual_direction_prior"],
            "residual_prior_correction": residual_parts["residual_prior_correction"],
            "residual_local": residual_parts["residual_local"],
            "residual_spectral": residual_parts["residual_spectral"],
            "residual_opportunity": residual_opportunity,
            "fuzzy_context": fuzzy_parts["fuzzy_context"],
            "fuzzy_membership": fuzzy_parts["fuzzy_membership"],
            "fuzzy_intensity": fuzzy_parts["fuzzy_intensity"],
            "fuzzy_ambiguity": fuzzy_parts["fuzzy_ambiguity"],
            "fuzzy_entropy": fuzzy_parts["fuzzy_entropy"],
            "fuzzy_ambiguity_mean": fuzzy_parts["fuzzy_ambiguity_mean"],
            "rt_innovation": rt_innovation,
            "low_innovation": low_innovation,
            "high_innovation": high_innovation,
            "rt_explainable": rt_explainable,
            "low_explainable": low_explainable,
            "high_explainable": high_explainable,
            "low_condition_gate": low_diag["condition_gate"],
            "high_condition_gate": high_diag["condition_gate"],
            "low_condition_overlap": low_diag["condition_overlap"],
            "high_condition_overlap": high_diag["condition_overlap"],
            "nwp_regime": nwp_regime,
            "nwp_global": nwp_global,
            "response_state": response_state,
            "response_global": response_global,
            "rt_global": rt_global,
            "text_global": text_global,
            "neutral_correction": neutral_correction,
            "rt_weights": rt_weights,
            "low_weights": low_weights,
            "high_weights": high_weights,
            "rt_alignment": rt_alignment,
            "low_transport": low_transport,
            "high_transport": high_transport,
            "risk_weight": risk_weight,
            "has_text": has_text,
            "has_future_text": has_future_text,
            "has_realtime_text": has_realtime_text,
        }
        if return_parts:
            return prediction, parts
        return prediction

    def objective(
        self,
        batch: Dict[str, torch.Tensor],
        prediction: torch.Tensor,
        parts: Dict[str, torch.Tensor],
        weights,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        training_prediction = prediction if self.use_text2_correction else parts["stage1_text_prediction"]
        main = event_weighted_forecasting_loss(
            training_prediction,
            batch["y"],
            batch["future_time_features"],
            weights.daytime_weight,
            0.55 * weights.numeric_event,
        )
        realtime_target = batch["y"] - parts["stage1_prediction"].detach()
        realtime_abs = realtime_target.detach().abs()
        realtime_scale = realtime_abs.mean(dim=1, keepdim=True).detach().clamp_min(1e-3)
        realtime_need = torch.sigmoid((realtime_abs - 0.60 * realtime_scale) / 0.04).detach()
        realtime_need = torch.maximum(
            realtime_need,
            parts["realtime_numeric_opportunity"].detach().clamp(0.0, 1.0),
        )
        hard_residual = (realtime_abs / realtime_scale).clamp(0.0, 3.0).detach()
        realtime_daylight = parts["realtime_scalar"][..., 7].detach().clamp(0.0, 1.0)
        realtime_valid = parts["has_realtime_text"].to(realtime_need.dtype).expand_as(realtime_need)
        realtime_weight = (0.10 + 0.90 * realtime_need) * (0.25 + 0.75 * realtime_daylight)
        realtime_weight = realtime_weight * (0.75 + 0.25 * hard_residual.square())
        realtime_weight = realtime_weight * realtime_valid
        realtime_loss = F.smooth_l1_loss(
            parts["realtime_delta"],
            realtime_target.detach(),
            reduction="none",
            beta=0.25,
        )
        realtime_residual = (realtime_loss * realtime_weight).sum() / realtime_weight.sum().clamp_min(1.0)
        realtime_opportunity = safe_probability_bce(
            parts["realtime_opportunity"],
            realtime_need * realtime_valid,
        )
        realtime_inactive = (1.0 - realtime_need) * realtime_valid
        realtime_selective_energy = (
            realtime_inactive * parts["realtime_delta"].square()
        ).sum() / realtime_valid.sum().clamp_min(1.0)
        realtime_direction = (
            F.relu(-parts["realtime_delta"] * realtime_target.detach()) * realtime_weight
        ).sum() / realtime_weight.sum().clamp_min(1.0)
        target_residual = batch["y"] - parts["stage1_text_prediction"].detach()
        residual_abs = target_residual.detach().abs()
        residual_scale = residual_abs.mean(dim=1, keepdim=True).detach().clamp_min(1e-3)
        residual_need = torch.sigmoid((residual_abs - 0.75 * residual_scale) / 0.05).detach()
        residual_need = torch.maximum(
            residual_need,
            parts["residual_numeric_prior"].detach().clamp(0.0, 1.0),
        )
        text2_stage_scale = parts["text2_stage_scale"].clamp(-1.0, 1.0)
        text_ramp_stage_scale = parts.get(
            "text_ramp_stage_scale",
            torch.zeros_like(text2_stage_scale),
        ).clamp(-1.0, 1.0)
        stage_scale_target = residual_need.mean().detach()
        stage_scale_loss = F.smooth_l1_loss(
            text2_stage_scale,
            stage_scale_target,
            beta=0.1,
        )
        daylight = daylight_weight(batch["future_time_features"], 2.0) - 1.0
        daylight = daylight.clamp(0.0, 1.0)
        text2_valid = parts["has_future_text"].to(target_residual.dtype).expand_as(target_residual)
        future_valid = batch["future_nwp_mask"].to(target_residual.dtype)
        ramp_target = torch.zeros_like(target_residual)
        ramp_target[:, 1:] = target_residual[:, 1:] - target_residual[:, :-1]
        ramp_abs = ramp_target.detach().abs()
        ramp_scale = ramp_abs.mean(dim=1, keepdim=True).detach().clamp_min(1e-3)
        ramp_need = torch.sigmoid((ramp_abs - 0.65 * ramp_scale) / 0.04).detach()
        text2_weight = (
            (0.05 + 0.95 * residual_need)
            * (0.20 + 0.80 * daylight)
            * text2_valid
            * future_valid
        )
        event_focus = parts.get("aligned_event_score", torch.zeros_like(ramp_need)).to(target_residual.dtype)
        event_residual_focus = torch.maximum(
            ramp_need,
            residual_need * (0.30 + 0.70 * event_focus.detach()),
        )
        weight_norm = text2_weight.sum().clamp_min(1.0)
        weighted_residual = F.smooth_l1_loss(
            parts["text2_intervention"], target_residual.detach(), reduction="none", beta=0.25
        )
        residual = (weighted_residual * text2_weight).sum() / weight_norm
        target_low, target_high = self.bands.split(target_residual.detach())
        low_branch = F.smooth_l1_loss(
            parts["low_delta"], target_low, reduction="none", beta=0.25
        )
        high_branch = F.smooth_l1_loss(
            parts["high_delta"], target_high, reduction="none", beta=0.25
        )
        branch = (text2_weight * (low_branch + high_branch)).sum() / weight_norm
        high_ramp_target = torch.zeros_like(target_high)
        high_ramp_target[:, 1:] = target_high[:, 1:] - target_high[:, :-1]
        event_target = ramp_target.abs().detach()
        shock_event_score = parts["extreme_event_risk"] * parts["shock_derivative"].abs()
        event_score = (
            parts["risk_weight"] * parts["high_delta"].abs()
            + shock_event_score
        )
        high_risk_focus = parts["high_risk_focus"].to(target_residual.dtype)
        high_risk_focus = torch.maximum(
            high_risk_focus,
            ramp_need * (0.35 + 0.65 * event_focus.detach()) * text2_valid * future_valid,
        )
        high_risk_focus = high_risk_focus * (0.20 + 0.80 * daylight)
        ramp_scale_target = high_risk_focus.mean().detach()
        ramp_stage_scale_loss = F.smooth_l1_loss(
            text_ramp_stage_scale,
            ramp_scale_target,
            beta=0.1,
        )
        raw_candidate_error = (
            parts["router_candidates"].detach() - batch["y"].unsqueeze(-1)
        ).square()
        candidate_error = (
            parts["calibrated_candidates"] - batch["y"].unsqueeze(-1)
        ).square()
        router_winner = candidate_error.argmin(dim=-1)
        best_candidate_error = candidate_error.min(dim=-1).values
        stage1_candidate_error = candidate_error[..., 0]
        raw_best_candidate_error = raw_candidate_error.min(dim=-1).values
        router_gain_target = (
            best_candidate_error < stage1_candidate_error - weights.improvement_margin
        ).to(target_residual.dtype)
        disagreement_need = torch.tanh(
            parts["candidate_range"].detach() / max(self.max_delta * 0.40, 1e-6)
        )
        router_weight = (
            (0.12 + 0.88 * torch.maximum(residual_need, disagreement_need))
            * (0.20 + 0.80 * daylight)
            * text2_valid
            * future_valid
        )
        improve_target_per_candidate = (
            stage1_candidate_error.unsqueeze(-1) - candidate_error.detach()
        )
        improve_target_scaled = torch.tanh(
            improve_target_per_candidate / (stage1_candidate_error.detach().mean().clamp_min(1e-4))
        )
        router_regression_loss = F.smooth_l1_loss(
            parts["improvement_score"],
            improve_target_scaled.detach(),
            reduction="none",
            beta=0.20,
        ).mean(dim=-1)
        router = (router_regression_loss * router_weight).sum() / router_weight.sum().clamp_min(1.0)
        calibration_loss = F.smooth_l1_loss(
            parts["calibrated_candidates"],
            batch["y"].unsqueeze(-1).expand_as(parts["calibrated_candidates"]),
            reduction="none",
            beta=0.25,
        ).mean(dim=-1)
        calibration = (calibration_loss * router_weight).sum() / router_weight.sum().clamp_min(1.0)
        calibration_gain = (
            (raw_best_candidate_error - best_candidate_error.detach()) * router_weight
        ).sum() / router_weight.sum().clamp_min(1.0)
        route_error = (
            parts["stage1_text_prediction"].detach()
            + parts["route_delta"]
            - batch["y"]
        ).square()
        router_safe = (
            F.relu(route_error - stage1_candidate_error + weights.improvement_margin)
            * router_weight
        ).sum() / router_weight.sum().clamp_min(1.0)
        router_gain = safe_probability_bce(
            parts["router_gate"],
            router_gain_target * text2_valid * future_valid,
        )
        router_entropy = (
            parts["router_entropy_h"] * router_weight
        ).sum() / router_weight.sum().clamp_min(1.0)
        winner_delta = torch.gather(
            parts["calibrated_candidates"].detach() - parts["stage1_text_prediction"].detach().unsqueeze(-1),
            dim=-1,
            index=router_winner.unsqueeze(-1),
        ).squeeze(-1)
        routed_delta_target = parts["router_gate"].detach() * winner_delta
        route_distill_loss = F.smooth_l1_loss(
            parts["route_delta"],
            routed_delta_target.detach(),
            reduction="none",
            beta=0.10,
        )
        route_distill = (
            route_distill_loss * router_weight * (0.50 + 0.50 * router_gain_target)
        ).sum() / router_weight.sum().clamp_min(1.0)
        router_oracle_gain = (
            (stage1_candidate_error - best_candidate_error) * router_weight
        ).sum() / router_weight.sum().clamp_min(1.0)
        risk_norm = high_risk_focus.sum().clamp_min(1.0)
        event_loss = F.smooth_l1_loss(
            event_score, event_target, reduction="none", beta=0.25
        )
        event = (event_loss * (0.20 + 0.80 * high_risk_focus)).sum() / (
            (0.20 + 0.80 * high_risk_focus).sum().clamp_min(1.0)
        )
        shock_loss = F.smooth_l1_loss(
            parts["shock_derivative"],
            high_ramp_target.detach(),
            reduction="none",
            beta=0.25,
        )
        shock = (shock_loss * high_risk_focus).sum() / risk_norm
        high_direction_error = F.relu(
            -parts["high_delta"] * target_high.detach()
        )
        high_direction = (high_direction_error * high_risk_focus).sum() / risk_norm
        treatment_effect = parts.get(
            "text2_treatment_effect",
            torch.zeros_like(parts["correction"]),
        )
        treatment_low = parts.get("treatment_low_delta", torch.zeros_like(parts["correction"]))
        treatment_high = parts.get("treatment_high_delta", torch.zeros_like(parts["correction"]))
        released_treatment = parts.get(
            "released_text2_treatment",
            parts["correction"],
        )
        released_treatment_high = parts.get(
            "released_treatment_high",
            torch.zeros_like(parts["correction"]),
        )
        event_treatment_delta = parts.get(
            "event_treatment_delta",
            torch.zeros_like(parts["correction"]),
        )
        treatment_focus = parts.get(
            "treatment_event_focus",
            torch.zeros_like(parts["correction"]),
        ).detach().clamp(0.0, 1.0)
        treatment_release_weight = text2_weight * (
            0.35 + 0.65 * torch.maximum(high_risk_focus, treatment_focus)
        )
        treatment_regression = (
            F.smooth_l1_loss(
                released_treatment,
                target_residual.detach(),
                reduction="none",
                beta=0.25,
            )
            * treatment_release_weight
        ).sum() / treatment_release_weight.sum().clamp_min(1.0)
        treatment_counterfactual_margin = 0.015 * residual_abs.detach().clamp_min(0.01)
        factual_effect_err = (
            parts["stage1_text_prediction"].detach()
            + released_treatment
            - batch["y"]
        ).square()
        neutral_effect_err = (
            parts["stage1_text_prediction"].detach()
            + parts["neutral_text2_intervention"].detach()
            - batch["y"]
        ).square()
        treatment_advantage = (
            F.relu(factual_effect_err - neutral_effect_err + treatment_counterfactual_margin)
            * treatment_release_weight
        ).sum() / treatment_release_weight.sum().clamp_min(1.0)
        stable_regime = (
            (1.0 - event_residual_focus.detach().clamp(0.0, 1.0))
            * text2_valid
            * future_valid
            * (0.25 + 0.75 * daylight)
        )
        treatment_stability = (
            stable_regime * treatment_effect.square()
        ).sum() / stable_regime.sum().clamp_min(1.0)
        treatment_event_direction = (
            F.relu(-event_treatment_delta * target_high.detach())
            * high_risk_focus
        ).sum() / risk_norm
        treatment_ramp = torch.zeros_like(event_treatment_delta)
        if event_treatment_delta.shape[1] > 1:
            treatment_ramp[:, 1:] = event_treatment_delta[:, 1:] - event_treatment_delta[:, :-1]
        treatment_ramp_direction = (
            F.relu(-treatment_ramp * high_ramp_target.detach())
            * high_risk_focus
        ).sum() / risk_norm
        treatment_low_consistency = (
            F.smooth_l1_loss(treatment_low, target_low, reduction="none", beta=0.25)
            * text2_weight
            * (1.0 - 0.50 * high_risk_focus)
        ).sum() / weight_norm
        treatment_high_consistency = (
            F.smooth_l1_loss(treatment_high, target_high, reduction="none", beta=0.25)
            * high_risk_focus
        ).sum() / risk_norm
        text_ramp_prior = parts.get("text_ramp_prior", torch.zeros_like(parts["high_delta"]))
        text_ramp_gate = parts.get("text_ramp_gate", torch.zeros_like(high_risk_focus))
        text_ramp_direction = parts.get("text_ramp_direction", torch.zeros_like(high_risk_focus))
        ramp_prior_direction = (
            F.relu(-text_ramp_prior * target_high.detach()) * high_risk_focus
        ).sum() / risk_norm
        ramp_direction_target = torch.sign(high_ramp_target.detach()).clamp(-1.0, 1.0)
        ramp_direction_weight = high_risk_focus * ramp_direction_target.abs()
        ramp_direction_loss = (
            F.relu(-text_ramp_direction * ramp_direction_target) * ramp_direction_weight
        ).sum() / ramp_direction_weight.sum().clamp_min(1.0)
        short_residual_loss = F.smooth_l1_loss(
            parts["high_delta"] + parts["shock_delta"],
            target_high.detach(),
            reduction="none",
            beta=0.25,
        )
        short_residual = (short_residual_loss * high_risk_focus).sum() / risk_norm
        risk_sparse = 0.5 * (
            parts["extreme_event_risk"].mean() + parts["fuzzy_intensity"].mean()
        )
        prototype_entropy = parts["risk_prototype_entropy"]
        prototype_balance = parts["risk_prototype_balance"]
        risk_tv = parts["risk_tv"]
        confidence = parts["residual_confidence"].mean()
        fuzzy_entropy = parts["fuzzy_entropy"]
        fuzzy_ambiguity = parts["fuzzy_ambiguity_mean"]
        residual_opportunity = parts["residual_opportunity"].mean()
        fuzzy_intensity_mean = parts["fuzzy_intensity"].mean()
        shock_abs = parts["shock_delta"].abs().mean()
        text2_scalar_abs = parts["text2_horizon_scalar"].detach().abs().mean()
        text2_risk_scalar = parts["text2_horizon_scalar"][..., 2:4].detach().mean()
        opportunity_target = residual_need.detach()
        opportunity = safe_probability_bce(
            parts["residual_opportunity_gate"],
            opportunity_target,
        )
        candidate_detached = (
            parts["stage1_text_prediction"].detach()
            + parts["text2_intervention"].detach()
        )
        improve_target = (
            (candidate_detached - batch["y"]).square()
            < (parts["stage1_text_prediction"].detach() - batch["y"]).square()
        ).to(target_residual.dtype)
        improve_gate_loss = safe_probability_bce(
            parts["residual_opportunity_gate"],
            improve_target * text2_valid,
        )
        opportunity = 0.5 * opportunity + 0.5 * improve_gate_loss
        release_target = (
            (
                parts["stage1_text_prediction"].detach()
                + parts["text2_residual_proposal"].detach()
                - batch["y"]
            ).square()
            < (
                parts["stage1_text_prediction"].detach()
                - batch["y"]
            ).square() - weights.improvement_margin
        ).to(target_residual.dtype)
        release_weight = text2_weight * (
            0.40
            + 0.40 * high_risk_focus.clamp(0.0, 1.0)
            + 0.20 * parts["release_numeric_uncertainty"].detach().clamp(0.0, 1.0)
        )
        safe_release_prob = parts["safe_release_gate"].float().clamp(1e-5, 1.0 - 1e-5)
        release_evidence_prob = parts["release_evidence_mean"].float().clamp(1e-5, 1.0 - 1e-5)
        release_target_f = release_target.float()
        release_calibration = (
            -(
                release_target_f * torch.log(safe_release_prob)
                + (1.0 - release_target_f) * torch.log1p(-safe_release_prob)
            )
            * release_weight
        ).sum() / release_weight.sum().clamp_min(1.0)
        evidence_loss = (
            -(
                release_target_f * torch.log(release_evidence_prob)
                + (1.0 - release_target_f) * torch.log1p(-release_evidence_prob)
            )
            * release_weight
            * (1.0 - 0.35 * parts["release_uncertainty"].detach().clamp(0.0, 1.0))
        ).sum() / release_weight.sum().clamp_min(1.0)
        release_uncertainty_regular = (
            parts["release_uncertainty"]
            * release_weight
            * release_target
        ).sum() / release_weight.sum().clamp_min(1.0)
        inactive = 1.0 - opportunity_target
        selective_energy = (
            inactive * text2_valid * future_valid * parts["correction"].square()
        ).sum() / (text2_valid * future_valid).sum().clamp_min(1.0)
        normal_release_weight = (
            text2_valid
            * future_valid
            * daylight
            * (1.0 - 0.85 * high_risk_focus.detach().clamp(0.0, 1.0))
            * (1.0 - 0.75 * residual_need.detach().clamp(0.0, 1.0))
        ).clamp(0.0, 1.0)
        normal_release_energy = (
            normal_release_weight * parts["correction"].square()
        ).sum() / normal_release_weight.sum().clamp_min(1.0)
        normal_release_coverage_loss = (
            normal_release_weight * parts["safe_release_gate"].clamp(0.0, 1.0)
        ).sum() / normal_release_weight.sum().clamp_min(1.0)
        normal_event_leakage_loss = (
            normal_release_weight
            * (
                parts["competitive_event_gate"].clamp(0.0, 1.0)
                + 0.50 * parts["event_stability_margin_gate"].clamp(0.0, 1.0)
            )
        ).sum() / normal_release_weight.sum().clamp_min(1.0)
        stable_trend_preserve_loss = (
            normal_release_weight
            * F.relu(
                0.10
                * parts["competitive_stable_gate"].detach().clamp(0.0, 1.0)
                - parts["competitive_stable_gate"].clamp(0.0, 1.0)
            )
        ).sum() / normal_release_weight.sum().clamp_min(1.0)
        event_release_weight = (
            text2_valid
            * future_valid
            * (
                0.45 * high_risk_focus.detach().clamp(0.0, 1.0)
                + 0.25 * parts["event_release_budget"].detach().clamp(0.0, 1.0)
                + 0.20 * parts["text_ramp_gate"].detach().clamp(0.0, 1.0)
                + 0.10 * parts["aligned_event_score"].detach().clamp(0.0, 1.0)
            )
        ).clamp(0.0, 1.0)
        event_release_recall_loss = (
            event_release_weight
            * F.relu(
                0.16
                - (
                    0.55 * parts["safe_release_gate"].clamp(0.0, 1.0)
                    + 0.45 * parts["competitive_event_gate"].clamp(0.0, 1.0)
                )
            )
            * release_target.detach()
        ).sum() / event_release_weight.sum().clamp_min(1.0)
        direction_error = F.relu(
            -parts["correction"] * target_residual.detach()
        )
        direction = (direction_error * text2_weight).sum() / weight_norm

        stage1_error = F.mse_loss(parts["stage1_prediction"].detach(), batch["y"])
        stage1_text_error = F.mse_loss(parts["stage1_text_prediction"], batch["y"])
        model_error = F.mse_loss(training_prediction, batch["y"])
        non_degradation = F.relu(stage1_text_error - stage1_error + weights.improvement_margin)
        if self.use_text2_correction:
            non_degradation = non_degradation + F.relu(
                model_error - stage1_text_error.detach() + weights.improvement_margin
            )
        base_err_h = (parts["stage1_text_prediction"].detach() - batch["y"]).square()
        pred_err_h = (training_prediction - batch["y"]).square()
        horizon_non_degradation = F.relu(pred_err_h - base_err_h + weights.improvement_margin)
        safe_weight = text2_weight * (1.0 - 0.70 * high_risk_focus.clamp(0.0, 1.0))
        horizon_non_degradation = (
            horizon_non_degradation * safe_weight
        ).sum() / safe_weight.sum().clamp_min(1.0)
        normal_horizon_non_degradation = (
            F.relu(pred_err_h - base_err_h + weights.improvement_margin)
            * normal_release_weight
        ).sum() / normal_release_weight.sum().clamp_min(1.0)
        residual_alignment = -(
            F.normalize(parts["correction"], dim=1)
            * F.normalize(target_residual.detach(), dim=1)
        ).sum(dim=1).mean()
        real_text_err = (
            parts["stage1_text_prediction"].detach()
            + parts["text2_intervention"]
            - batch["y"]
        ).square()
        neutral_text_err = (
            parts["stage1_text_prediction"].detach()
            + parts["neutral_text2_intervention"].detach()
            - batch["y"]
        ).square()
        text_contrast_weight = text2_weight * (
            0.50 + 0.50 * high_risk_focus.clamp(0.0, 1.0)
        )
        text_contrast_margin = 0.02 * residual_abs.detach().clamp_min(0.01)
        text_contrast = (
            F.relu(real_text_err - neutral_text_err + text_contrast_margin)
            * text_contrast_weight
        ).sum() / text_contrast_weight.sum().clamp_min(1.0)
        energy = parts["correction"].square().mean() + 0.50 * parts["route_delta"].square().mean()
        smooth = (
            parts["correction"][:, 1:] - parts["correction"][:, :-1]
        ).square().mean() + 0.50 * (
            parts["route_delta"][:, 1:] - parts["route_delta"][:, :-1]
        ).square().mean()

        nwp_unit = F.normalize(parts["nwp_regime"].detach(), dim=-1)
        response_unit = F.normalize(parts["response_state"].detach(), dim=-1)
        rt_unit = F.normalize(parts["rt_innovation"], dim=-1)
        low_unit = F.normalize(parts["low_innovation"], dim=-1)
        high_unit = F.normalize(parts["high_innovation"], dim=-1)
        orthogonal = (
            (rt_unit * nwp_unit).sum(dim=-1).abs().mean()
            + (rt_unit * response_unit).sum(dim=-1).abs().mean()
        )
        if self.use_text2_correction:
            orthogonal = orthogonal + (
                (low_unit * nwp_unit).sum(dim=-1).abs().mean()
                + (high_unit * nwp_unit).sum(dim=-1).abs().mean()
                + (low_unit * response_unit).sum(dim=-1).abs().mean()
                + (high_unit * response_unit).sum(dim=-1).abs().mean()
            )
        low_condition_overlap = parts["low_condition_overlap"].mean()
        high_condition_overlap = parts["high_condition_overlap"].mean()
        deconfound = 0.5 * (low_condition_overlap + high_condition_overlap)
        neutral = parts["neutral_correction"].square().mean()
        transport_cost = (
            parts["low_transport"]["expected_cost"]
            + parts["high_transport"]["expected_cost"]
        )
        transport_entropy = 0.5 * (
            parts["low_transport"]["entropy"]
            + parts["high_transport"]["entropy"]
        )
        async_lag = parts["rt_alignment"]["expected_lag"]
        async_entropy = parts["rt_alignment"]["entropy"]
        async_field_entropy = parts["rt_alignment"]["field_entropy"]
        counterfactual = self._counterfactual_loss(
            torch.cat([parts["rt_global"], parts["text_global"]], dim=-1),
            parts["nwp_global"],
            parts["response_global"],
            parts["text2_intervention"],
            target_residual.detach(),
            parts["has_text"].squeeze(1),
        )
        text2_loss = (
            weights.residual * residual
            + weights.branch * branch
            + weights.event * event
            + weights.shock * shock
            + weights.short_residual * short_residual
            + 0.75 * weights.residual * treatment_regression
            + 0.65 * weights.text_contrast * treatment_advantage
            + 0.50 * weights.selective_energy * treatment_stability
            + 0.65 * weights.direction * treatment_event_direction
            + 0.45 * weights.direction * treatment_ramp_direction
            + 0.35 * weights.branch * (treatment_low_consistency + treatment_high_consistency)
            + weights.risk_sparse * risk_sparse
            + weights.risk_tv * risk_tv
            + weights.prototype_balance * prototype_balance
            + weights.confidence_sparse * confidence
            + weights.fuzzy_ambiguity * fuzzy_ambiguity
            + weights.residual_alignment * residual_alignment
            + weights.opportunity * opportunity
            + 0.50 * weights.opportunity * release_calibration
            + 0.25 * weights.opportunity * evidence_loss
            + 0.10 * weights.opportunity * release_uncertainty_regular
            + weights.selective_energy * selective_energy
            + 1.50 * weights.selective_energy * normal_release_energy
            + 1.95 * weights.selective_energy * normal_release_coverage_loss
            + 1.25 * weights.selective_energy * normal_event_leakage_loss
            + 0.20 * weights.selective_energy * stable_trend_preserve_loss
            + 0.70 * weights.opportunity * event_release_recall_loss
            + weights.direction * direction
            + weights.direction * high_direction
            + 0.5 * weights.direction * ramp_prior_direction
            + 0.5 * weights.direction * ramp_direction_loss
            + weights.energy * energy
            + weights.smooth * smooth
            + weights.neutral * neutral
            + weights.transport_cost * transport_cost
            - weights.transport_entropy * transport_entropy
            - weights.prototype_entropy * prototype_entropy
            + weights.counterfactual * counterfactual
            + weights.text_contrast * text_contrast
            + weights.calibration * calibration
            + weights.router * router
            + weights.router_distill * route_distill
            + weights.router_safe * router_safe
            + weights.router_gain * router_gain
            - weights.router_entropy * router_entropy
        )
        if not self.use_text2_correction:
            text2_loss = text2_loss * 0.0
        total = (
            main
            + weights.realtime_residual * realtime_residual
            + weights.opportunity * realtime_opportunity
            + weights.selective_energy * realtime_selective_energy
            + weights.direction * realtime_direction
            + weights.stage_scale * stage_scale_loss
            + 0.5 * weights.stage_scale * ramp_stage_scale_loss
            + weights.non_degradation * non_degradation
            + weights.non_degradation * horizon_non_degradation
            + 1.25 * weights.non_degradation * normal_horizon_non_degradation
            + weights.orthogonal * orthogonal
            + 0.5 * weights.orthogonal * deconfound
            + weights.async_lag * async_lag
            - weights.async_entropy * async_entropy
            + text2_loss
        )
        terms = {
            "main": float(main.detach()),
            "realtime_residual": float(realtime_residual.detach()),
            "realtime_opportunity_loss": float(realtime_opportunity.detach()),
            "realtime_selective_energy": float(realtime_selective_energy.detach()),
            "realtime_direction": float(realtime_direction.detach()),
            "realtime_delta_abs": float(parts["realtime_delta"].detach().abs().mean()),
            "realtime_confidence": float(parts["realtime_confidence"].detach().mean()),
            "realtime_opportunity": float(parts["realtime_opportunity"].detach().mean()),
            "realtime_numeric_opportunity": float(parts["realtime_numeric_opportunity"].detach().mean()),
            "realtime_need": float(realtime_need.detach().mean()),
            "residual": float(residual.detach()),
            "branch": float(branch.detach()),
            "event": float(event.detach()),
            "shock": float(shock.detach()),
            "short_residual": float(short_residual.detach()),
            "calibration": float(calibration.detach()),
            "calibration_gain": float(calibration_gain.detach()),
            "router": float(router.detach()),
            "route_distill": float(route_distill.detach()),
            "router_safe": float(router_safe.detach()),
            "router_gain": float(router_gain.detach()),
            "router_entropy": float(router_entropy.detach()),
            "router_oracle_gain": float(router_oracle_gain.detach()),
            "router_gate": float(parts["router_gate"].detach().mean()),
            "router_top_prob": float(parts["router_top_prob"].detach().mean()),
            "router_stage_scale": float(parts["router_stage_scale"].detach()),
            "route_delta_abs": float(parts["route_delta"].detach().abs().mean()),
            "route_delta_unscaled_abs": float(parts["route_delta_unscaled"].detach().abs().mean()),
            "safe_release_gate": float(parts["safe_release_gate"].detach().mean()),
            "prior_release_gate": float(parts["prior_release_gate"].detach().mean()),
            "decoder_release_gate": float(parts["decoder_release_gate"].detach().mean()),
            "semantic_router_gate": float(parts["semantic_router_gate"].detach().mean()),
            "extreme_release_gate": float(parts["extreme_release_gate"].detach().mean()),
            "release_calibration": float(release_calibration.detach()),
            "release_evidence_loss": float(evidence_loss.detach()),
            "release_uncertainty_regular": float(release_uncertainty_regular.detach()),
            "release_target": float(release_target.detach().mean()),
            "release_uncertainty": float(parts["release_uncertainty"].detach().mean()),
            "release_evidence_mean": float(parts["release_evidence_mean"].detach().mean()),
            "release_counterfactual_gain": float(parts["release_counterfactual_gain"].detach().mean()),
            "release_agreement": float(parts["release_agreement"].detach().mean()),
            "treatment_effect_abs": float(treatment_effect.detach().abs().mean()),
            "released_treatment_abs": float(released_treatment.detach().abs().mean()),
            "treatment_low_release": float(parts["treatment_low_release"].detach().mean()),
            "treatment_high_release": float(parts["treatment_high_release"].detach().mean()),
            "treatment_event_focus": float(treatment_focus.detach().mean()),
            "treatment_regression": float(treatment_regression.detach()),
            "treatment_advantage": float(treatment_advantage.detach()),
            "treatment_stability": float(treatment_stability.detach()),
            "treatment_event_direction": float(treatment_event_direction.detach()),
            "treatment_ramp_direction": float(treatment_ramp_direction.detach()),
            "candidate_range": float(parts["candidate_range"].detach().mean()),
            "semantic_release": float(parts["semantic_release"].detach().mean()),
            "positive_release": float(parts["positive_release"].detach().mean()),
            "disagreement_release": float(parts["disagreement_release"].detach().mean()),
            "router_gain_target": float(router_gain_target.detach().mean()),
            "high_direction": float(high_direction.detach()),
            "high_risk_focus": float(high_risk_focus.detach().mean()),
            "risk_sparse": float(risk_sparse.detach()),
            "risk_tv": float(risk_tv.detach()),
            "confidence": float(confidence.detach()),
            "residual_opportunity": float(residual_opportunity.detach()),
            "stage_scale_loss": float(stage_scale_loss.detach()),
            "stage_scale_target": float(stage_scale_target.detach()),
            "ramp_stage_scale_loss": float(ramp_stage_scale_loss.detach()),
            "ramp_stage_scale_target": float(ramp_scale_target.detach()),
            "fuzzy_intensity": float(fuzzy_intensity_mean.detach()),
            "text2_scalar_abs": float(text2_scalar_abs.detach()),
            "text2_risk_scalar": float(text2_risk_scalar.detach()),
            "shock_abs": float(shock_abs.detach()),
            "opportunity": float(opportunity.detach()),
            "improve_gate_loss": float(improve_gate_loss.detach()),
            "selective_energy": float(selective_energy.detach()),
            "normal_release_energy": float(normal_release_energy.detach()),
            "normal_release_coverage_loss": float(normal_release_coverage_loss.detach()),
            "normal_event_leakage_loss": float(normal_event_leakage_loss.detach()),
            "stable_trend_preserve_loss": float(stable_trend_preserve_loss.detach()),
            "event_release_recall_loss": float(event_release_recall_loss.detach()),
            "direction": float(direction.detach()),
            "opportunity_gate": float(parts["residual_opportunity_gate"].mean().detach()),
            "text2_stage_scale": float(parts["text2_stage_scale"].detach()),
            "text_ramp_stage_scale": float(text_ramp_stage_scale.detach()),
            "raw_text2_correction_abs": float(parts["raw_text2_correction"].detach().abs().mean()),
            "scaled_text2_correction_abs": float(parts["scaled_text2_correction"].detach().abs().mean()),
            "text_ramp_prior_abs": float(text_ramp_prior.detach().abs().mean()),
            "text_ramp_gate": float(text_ramp_gate.detach().mean()),
            "risk_gate": float(parts["risk_gate"].detach().mean()),
            "aligned_event_score": float(parts["aligned_event_score"].detach().mean()),
            "low_text_coverage": float(parts["low_text_coverage"].detach().mean()),
            "high_text_coverage": float(parts["high_text_coverage"].detach().mean()),
            "numeric_prior": float(parts["residual_numeric_prior"].mean().detach()),
            "semantic_certainty": float(parts["residual_semantic_certainty"].mean().detach()),
            "prior_correction_abs": float(parts["residual_prior_correction"].detach().abs().mean()),
            "fuzzy_entropy": float(fuzzy_entropy.detach()),
            "fuzzy_ambiguity": float(fuzzy_ambiguity.detach()),
            "prototype_entropy": float(prototype_entropy.detach()),
            "prototype_balance": float(prototype_balance.detach()),
            "non_degradation": float(non_degradation.detach()),
            "horizon_non_degradation": float(horizon_non_degradation.detach()),
            "normal_horizon_non_degradation": float(normal_horizon_non_degradation.detach()),
            "residual_alignment": float(residual_alignment.detach()),
            "text_contrast": float(text_contrast.detach()),
            "energy": float(energy.detach()),
            "smooth": float(smooth.detach()),
            "orthogonal": float(orthogonal.detach()),
            "deconfound": float(deconfound.detach()),
            "low_condition_gate": float(parts["low_condition_gate"].detach().mean()),
            "high_condition_gate": float(parts["high_condition_gate"].detach().mean()),
            "low_condition_overlap": float(low_condition_overlap.detach()),
            "high_condition_overlap": float(high_condition_overlap.detach()),
            "neutral": float(neutral.detach()),
            "transport_cost": float(transport_cost.detach()),
            "transport_entropy": float(transport_entropy.detach()),
            "async_lag": float(async_lag.detach()),
            "async_entropy": float(async_entropy.detach()),
            "async_field_entropy": float(async_field_entropy.detach()),
            "async_lag_center": float(parts["rt_alignment"]["lag_center_mean"].detach()),
            "async_lag_width": float(parts["rt_alignment"]["lag_width_mean"].detach()),
            "counterfactual": float(counterfactual.detach()),
        }
        return total, terms

    @staticmethod
    def _counterfactual_loss(
        text_global: torch.Tensor,
        nwp_global: torch.Tensor,
        response_global: torch.Tensor,
        correction: torch.Tensor,
        target_residual: torch.Tensor,
        valid_text: torch.Tensor,
    ) -> torch.Tensor:
        batch = text_global.shape[0]
        if batch < 2 or bool(valid_text.sum().item() < 2):
            return correction.new_tensor(0.0)
        nwp = F.normalize(nwp_global.float(), dim=-1)
        response = F.normalize(response_global.float(), dim=-1)
        text = F.normalize(text_global.float(), dim=-1)
        state = F.normalize(torch.cat([nwp, response], dim=-1), dim=-1)
        nwp_distance = torch.cdist(state, state, p=2)
        nwp_distance = nwp_distance + torch.eye(batch, device=nwp.device) * 1e6
        valid_pair = valid_text.bool().unsqueeze(0) & valid_text.bool().unsqueeze(1)
        nwp_distance = nwp_distance.masked_fill(~valid_pair, 1e6)
        neighbor = nwp_distance.argmin(dim=1)
        has_neighbor = nwp_distance.gather(1, neighbor[:, None]).squeeze(1) < 1e5
        if not bool(has_neighbor.any().item()):
            return correction.new_tensor(0.0)
        semantic_delta = (text - text[neighbor]).abs().mean(dim=-1).detach()
        residual_delta = target_residual - target_residual[neighbor]
        correction_delta = correction - correction[neighbor]
        loss = F.smooth_l1_loss(
            correction_delta, residual_delta, reduction="none", beta=0.25
        )
        loss = loss.mean(dim=1) * semantic_delta
        return loss[has_neighbor].mean()


class LossWeights:
    def __init__(self, args):
        self.daytime_weight = args.daytime_weight
        self.numeric_event = args.w_numeric_event
        self.realtime_residual = args.w_realtime_residual
        self.residual = args.w_residual
        self.history_prior = args.w_history_prior
        self.history_trust = args.w_history_trust
        self.nwp_prior = args.w_nwp_prior
        self.nwp_trust = args.w_nwp_trust
        self.periodic_prior = args.w_periodic_prior
        self.periodic_trust = args.w_periodic_trust
        self.candidate_trust_sparse = args.w_candidate_trust_sparse
        self.branch = args.w_branch
        self.event = args.w_event
        self.shock = args.w_shock
        self.short_residual = args.w_short_residual
        self.risk_sparse = args.w_risk_sparse
        self.risk_tv = args.w_risk_tv
        self.prototype_entropy = args.w_prototype_entropy
        self.prototype_balance = args.w_prototype_balance
        self.confidence_sparse = args.w_confidence_sparse
        self.fuzzy_ambiguity = args.w_fuzzy_ambiguity
        self.non_degradation = args.w_non_degradation
        self.residual_alignment = args.w_residual_alignment
        self.opportunity = args.w_opportunity
        self.selective_energy = args.w_selective_energy
        self.direction = args.w_direction
        self.stage_scale = args.w_stage_scale
        self.energy = args.w_energy
        self.smooth = args.w_smooth
        self.orthogonal = args.w_orthogonal
        self.neutral = args.w_neutral
        self.transport_cost = args.w_transport_cost
        self.transport_entropy = args.w_transport_entropy
        self.async_lag = args.w_async_lag
        self.async_entropy = args.w_async_entropy
        self.counterfactual = args.w_counterfactual
        self.text_contrast = args.w_text_contrast
        self.calibration = args.w_calibration
        self.router = args.w_router
        self.router_distill = args.w_router_distill
        self.router_safe = args.w_router_safe
        self.router_gain = args.w_router_gain
        self.router_entropy = args.w_router_entropy
        self.improvement_margin = args.improvement_margin


def _autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _forward_with_parts(model: nn.Module, batch: Dict[str, torch.Tensor]):
    if isinstance(model, NumericalSolarBaseline):
        return model(batch, return_hidden=True)
    return model(batch, return_parts=True)


def solar_feature_cols() -> Tuple[str, ...]:
    return (TARGET, *SOLAR_AUX_COLS)


def build_baseline_model(args, device: torch.device) -> NumericalSolarBaseline:
    return NumericalSolarBaseline(
        input_dim=len(solar_feature_cols()),
        nwp_dim=len(SOLAR_NWP_COLS),
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_layers=args.d_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        nwp_layers=args.nwp_layers,
        ma_kernel=args.ma_kernel,
        max_delta=args.baseline_max_delta,
        use_history_backbone=args.use_history_backbone,
        use_nwp_prior=args.use_nwp_prior,
        use_periodic_prior=args.use_periodic_prior,
        use_aux_residual=args.use_aux_residual,
        use_past_nwp_residual=args.use_past_nwp_residual,
        use_future_nwp_residual=args.use_future_nwp_residual,
        aux_gate_bias=args.aux_gate_bias,
        past_nwp_gate_bias=args.past_nwp_gate_bias,
        future_nwp_gate_bias=args.future_nwp_gate_bias,
        aux_gate_max=args.aux_gate_max,
        past_nwp_gate_max=args.past_nwp_gate_max,
        future_nwp_gate_max=args.future_nwp_gate_max,
        dataset_count=len(getattr(args, "dataset_names", []) or [getattr(args, "dataset_name", "")]),
        use_dataset_conditioning=args.use_dataset_conditioning,
    ).to(device)


def load_baseline_model(checkpoint_path: str, device: torch.device) -> Tuple[NumericalSolarBaseline, dict]:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved = AttrDict(**checkpoint.get("args", {}))
    for key, value in {
        "seq_len": 96,
        "pred_len": 96,
        "d_model": 512,
        "n_heads": 8,
        "e_layers": 4,
        "d_layers": 3,
        "d_ff": 2048,
        "dropout": 0.1,
        "nwp_layers": 2,
        "ma_kernel": 25,
        "baseline_max_delta": 0.25,
        "use_history_backbone": False,
        "use_nwp_prior": False,
        "use_periodic_prior": False,
        "use_aux_residual": True,
        "use_past_nwp_residual": True,
        "use_future_nwp_residual": True,
        "aux_gate_bias": -2.2,
        "past_nwp_gate_bias": -3.0,
        "future_nwp_gate_bias": -3.0,
        "aux_gate_max": 0.80,
        "past_nwp_gate_max": 0.70,
        "future_nwp_gate_max": 0.70,
        "use_dataset_conditioning": False,
    }.items():
        if not hasattr(saved, key):
            setattr(saved, key, value)
    if not hasattr(saved, "dataset_names"):
        metadata_names = checkpoint.get("metadata", {}).get("dataset_names", [])
        if metadata_names:
            saved.dataset_names = metadata_names
    if not hasattr(saved, "dataset_name") and not hasattr(saved, "dataset_names"):
        saved.dataset_name = "kongzhaopu"
    model = build_baseline_model(saved, device)
    state_dict = dict(checkpoint["model"])
    model_state = model.state_dict()
    for key in (
        "candidate_pool_weights",
        "candidate_pool_enabled",
        "dataset_candidate_pool_weights",
        "dataset_candidate_pool_enabled",
    ):
        if (
            key in state_dict
            and key in model_state
            and state_dict[key].shape != model_state[key].shape
        ):
            state_dict.pop(key)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    allowed_missing_prefixes = (
        "backbone.backbone.active_variable_mask",
        "history_backbone.",
        "history_trust.",
        "nwp_physical_prior.",
        "periodic_prior.",
        "candidate_pool_weights",
        "candidate_pool_enabled",
        "dataset_candidate_pool_weights",
        "dataset_candidate_pool_enabled",
        "dataset_embedding.",
        "dataset_horizon_bias.",
        "dataset_gate_adapter.",
    )
    missing = [
        name for name in missing
        if not any(name.startswith(prefix) for prefix in allowed_missing_prefixes)
    ]
    if missing or unexpected:
        raise RuntimeError(
            f"could not load numerical baseline: missing={missing}, unexpected={unexpected}"
        )
    candidate_pool = checkpoint.get("numerical_candidate_pool") or checkpoint.get("metadata", {}).get(
        "numerical_candidate_pool"
    )
    if candidate_pool is not None:
        apply_numerical_candidate_pool(model, candidate_pool)
    model.eval()
    return model, checkpoint


def train_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    amp,
    weights,
    grad_clip,
    ema_state: Optional[Dict[str, torch.Tensor]] = None,
    ema_decay: Optional[float] = None,
):
    model.train()
    total = 0.0
    examples = 0
    term_sums: Dict[str, float] = {}
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, amp):
            prediction, parts = _forward_with_parts(model, batch)
            loss, terms = model.objective(batch, prediction, parts, weights)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
            optimizer.step()
        if ema_decay is not None:
            ema_state = _update_ema_state(ema_state, model, ema_decay)
        batch_size = prediction.shape[0]
        total += float(loss.detach()) * batch_size
        examples += batch_size
        for key, value in terms.items():
            term_sums[key] = term_sums.get(key, 0.0) + value * batch_size
    return (
        total / max(examples, 1),
        {key: value / max(examples, 1) for key, value in term_sums.items()},
        ema_state,
    )


@torch.inference_mode()
def collect_numerical_candidate_arrays(model, loader, device, amp):
    if not isinstance(model, NumericalSolarBaseline):
        return None
    was_enabled = bool(model.candidate_pool_enabled.item())
    model.candidate_pool_enabled.fill_(False)
    model.eval()
    candidates, targets, stage1, dataset_indices = [], [], [], []
    for batch in loader:
        batch = move_batch(batch, device)
        with _autocast_context(device, amp):
            prediction, parts = model(batch, return_hidden=True)
        candidate_pool = parts.get("numerical_candidate_pool")
        if candidate_pool is None:
            continue
        candidates.append(candidate_pool.float().cpu().numpy())
        targets.append(batch["y"].float().cpu().numpy())
        stage1.append(prediction.float().cpu().numpy())
        if "dataset_index" in batch:
            dataset_indices.append(batch["dataset_index"].long().cpu().numpy().reshape(-1))
    model.candidate_pool_enabled.fill_(was_enabled)
    if not candidates:
        return None
    payload = {
        "candidates": np.concatenate(candidates, axis=0),
        "target": np.concatenate(targets, axis=0),
        "stage1": np.concatenate(stage1, axis=0),
    }
    if dataset_indices:
        payload["dataset_index"] = np.concatenate(dataset_indices, axis=0).astype(np.int64, copy=False)
    return payload


def _fit_convex_candidate_weights(
    candidates: np.ndarray,
    target: np.ndarray,
    *,
    ridge: float,
    simplex_steps: int,
) -> np.ndarray:
    candidates = np.asarray(candidates, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if candidates.ndim != 3:
        raise ValueError(f"numerical candidates must be [N,H,C], got {candidates.shape}")
    horizon = candidates.shape[1]
    count = candidates.shape[2]
    eye = np.eye(count, dtype=np.float64)
    uniform = np.full(count, 1.0 / max(count, 1), dtype=np.float64)
    weights = np.zeros((horizon, count), dtype=np.float64)
    for index in range(horizon):
        x = candidates[:, index, :]
        y = target[:, index]
        system = x.T @ x + float(ridge) * eye
        rhs = x.T @ y + float(ridge) * uniform
        try:
            w = np.linalg.solve(system, rhs)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(system, rhs, rcond=None)[0]
        w = np.clip(w, 0.0, None)
        if not np.isfinite(w).all() or float(w.sum()) <= 1e-8:
            w = uniform.copy()
        else:
            w = w / w.sum()
        for _ in range(int(max(simplex_steps, 0))):
            residual = x @ w - y
            grad = 2.0 * (x.T @ residual) / max(x.shape[0], 1) + 2.0 * float(ridge) * (w - uniform)
            step = 0.12 / (np.linalg.norm(grad) + 1e-8)
            w = np.clip(w - step * grad, 0.0, None)
            if float(w.sum()) <= 1e-8:
                w = uniform.copy()
            else:
                w = w / w.sum()
        weights[index] = w
    return weights.astype(np.float32, copy=False)


def _candidate_pool_record(
    candidates: np.ndarray,
    target: np.ndarray,
    stage1: np.ndarray,
    weights: np.ndarray,
    *,
    ridge: float,
    variant: str,
    dataset_index: Optional[np.ndarray] = None,
) -> dict:
    if weights.ndim == 2:
        calibrated = np.sum(candidates * weights.reshape(1, weights.shape[0], weights.shape[1]), axis=-1)
    elif weights.ndim == 3 and dataset_index is not None:
        selected = weights[np.asarray(dataset_index, dtype=np.int64)]
        calibrated = np.sum(candidates * selected, axis=-1)
    else:
        raise ValueError(f"unsupported candidate pool weight shape: {weights.shape}")
    baseline_mse = _scaled_mse(stage1, target)
    baseline_mae = _scaled_mae(stage1, target)
    mse = _scaled_mse(calibrated, target)
    mae = _scaled_mae(calibrated, target)
    fold_gains = []
    for fold in _fold_slices(stage1.shape[0], folds=4):
        fold_base = _scaled_mse(stage1[fold], target[fold])
        fold_cal = _scaled_mse(calibrated[fold], target[fold])
        fold_gains.append(fold_base - fold_cal)
    fold_gains_arr = np.asarray(fold_gains, dtype=np.float32)
    entropy = -np.sum(weights * np.log(np.clip(weights, 1e-8, 1.0)), axis=-1)
    record = {
        "name": "validation_convex_numerical_candidate_pool",
        "variant": variant,
        "candidate_names": list(NUMERICAL_CANDIDATE_NAMES),
        "weights": weights.astype(np.float32, copy=False),
        "ridge_lambda": float(ridge),
        "val_scaled_MSE": float(mse),
        "val_scaled_MAE": float(mae),
        "val_scaled_improve_MSE": float(baseline_mse - mse),
        "val_scaled_improve_MAE": float(baseline_mae - mae),
        "val_baseline_scaled_MSE": float(baseline_mse),
        "val_baseline_scaled_MAE": float(baseline_mae),
        "val_fold_gain_min": float(fold_gains_arr.min()) if fold_gains_arr.size else 0.0,
        "val_fold_gain_mean": float(fold_gains_arr.mean()) if fold_gains_arr.size else 0.0,
        "val_fold_positive_rate": float(np.mean(fold_gains_arr >= -1e-8)) if fold_gains_arr.size else 1.0,
        "val_weight_entropy_mean": float(np.mean(entropy)),
        "val_weight_max_mean": float(np.mean(np.max(weights, axis=-1))),
    }
    if dataset_index is not None:
        dataset_metrics = {}
        for dataset_id in sorted(np.unique(dataset_index).astype(int).tolist()):
            mask = np.asarray(dataset_index) == int(dataset_id)
            if not mask.any():
                continue
            base_mse = _scaled_mse(stage1[mask], target[mask])
            cal_mse = _scaled_mse(calibrated[mask], target[mask])
            dataset_metrics[str(dataset_id)] = {
                "scaled_MSE": float(cal_mse),
                "scaled_improve_MSE": float(base_mse - cal_mse),
                "coverage": float(np.mean(mask)),
            }
        record["dataset_candidate_metrics"] = dataset_metrics
        if dataset_metrics:
            record["val_dataset_gain_min"] = float(
                min(item["scaled_improve_MSE"] for item in dataset_metrics.values())
            )
            record["val_dataset_positive_rate"] = float(
                np.mean([item["scaled_improve_MSE"] >= -1e-8 for item in dataset_metrics.values()])
            )
        else:
            record["val_dataset_gain_min"] = 0.0
            record["val_dataset_positive_rate"] = 1.0
    return record


def fit_numerical_candidate_pool(
    arrays: Optional[dict],
    *,
    ridge_grid: Sequence[float] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 3.0, 10.0, 30.0),
    simplex_steps: int = 4,
) -> Optional[dict]:
    if arrays is None:
        return None
    candidates = np.asarray(arrays["candidates"], dtype=np.float64)
    target = np.asarray(arrays["target"], dtype=np.float64)
    stage1 = np.asarray(arrays["stage1"], dtype=np.float64)
    if candidates.ndim != 3:
        raise ValueError(f"numerical candidates must be [N,H,C], got {candidates.shape}")
    baseline_mse = _scaled_mse(stage1, target)
    baseline_mae = _scaled_mae(stage1, target)
    horizon = candidates.shape[1]
    count = candidates.shape[2]
    best = None
    for ridge in ridge_grid:
        weights = _fit_convex_candidate_weights(
            candidates,
            target,
            ridge=float(ridge),
            simplex_steps=simplex_steps,
        )
        record = _candidate_pool_record(
            candidates,
            target,
            stage1,
            weights,
            ridge=float(ridge),
            variant="horizon_wise_convex_pool",
        )
        score = (
            record["val_scaled_MSE"]
            + 0.15 * max(0.0, -record["val_scaled_improve_MSE"])
            + 0.10 * max(0.0, -record["val_fold_gain_min"])
            - 0.08 * max(0.0, record["val_scaled_improve_MSE"])
        )
        record["selection_score"] = float(score)
        if best is None or record["selection_score"] < best["selection_score"]:
            best = record
        dataset_index = arrays.get("dataset_index")
        if dataset_index is not None:
            dataset_index_arr = np.asarray(dataset_index, dtype=np.int64).reshape(-1)
            if dataset_index_arr.shape[0] == candidates.shape[0]:
                dataset_count = int(dataset_index_arr.max()) + 1 if dataset_index_arr.size else 1
                global_weights = weights.astype(np.float32, copy=True)
                dataset_weights = np.repeat(global_weights[None, ...], dataset_count, axis=0)
                min_examples = max(32, horizon // 2)
                for dataset_id in range(dataset_count):
                    mask = dataset_index_arr == dataset_id
                    if int(mask.sum()) < min_examples:
                        continue
                    local_weights = _fit_convex_candidate_weights(
                        candidates[mask],
                        target[mask],
                        ridge=float(ridge),
                        simplex_steps=simplex_steps,
                    )
                    local_record = _candidate_pool_record(
                        candidates[mask],
                        target[mask],
                        stage1[mask],
                        local_weights,
                        ridge=float(ridge),
                        variant="local_probe",
                    )
                    if local_record["val_scaled_improve_MSE"] >= -1e-8:
                        dataset_weights[dataset_id] = local_weights
                dataset_record = _candidate_pool_record(
                    candidates,
                    target,
                    stage1,
                    dataset_weights,
                    ridge=float(ridge),
                    variant="dataset_horizon_wise_convex_pool",
                    dataset_index=dataset_index_arr,
                )
                score = (
                    dataset_record["val_scaled_MSE"]
                    + 0.15 * max(0.0, -dataset_record["val_scaled_improve_MSE"])
                    + 0.12 * max(0.0, -dataset_record["val_fold_gain_min"])
                    + 0.20 * max(0.0, -dataset_record.get("val_dataset_gain_min", 0.0))
                    - 0.08 * max(0.0, dataset_record["val_scaled_improve_MSE"])
                )
                dataset_record["selection_score"] = float(score)
                if best is None or dataset_record["selection_score"] < best["selection_score"]:
                    best = dataset_record
    if best is None:
        return None
    if (
        best["val_scaled_improve_MSE"] < -1e-8
        or best["val_fold_positive_rate"] < 0.50
        or best["val_fold_gain_min"] < -max(1e-4, 0.75 * max(best["val_scaled_improve_MSE"], 0.0))
    ):
        fallback = np.zeros((horizon, count), dtype=np.float32)
        fallback[:, NUMERICAL_CANDIDATE_NAMES.index("numeric_chain_final")] = 1.0
        best = {
            "name": "no_candidate_pool_release",
            "variant": "fallback_original_numeric_chain",
            "candidate_names": list(NUMERICAL_CANDIDATE_NAMES),
            "weights": fallback,
            "ridge_lambda": 0.0,
            "val_scaled_MSE": float(baseline_mse),
            "val_scaled_MAE": float(baseline_mae),
            "val_scaled_improve_MSE": 0.0,
            "val_scaled_improve_MAE": 0.0,
            "val_baseline_scaled_MSE": float(baseline_mse),
            "val_baseline_scaled_MAE": float(baseline_mae),
            "val_fold_gain_min": 0.0,
            "val_fold_gain_mean": 0.0,
            "val_fold_positive_rate": 1.0,
            "val_weight_entropy_mean": 0.0,
            "val_weight_max_mean": 1.0,
            "selection_score": float(baseline_mse),
            "fallback_reason": "validation_candidate_pool_not_stable",
        }
    return best


def apply_numerical_candidate_pool(model, candidate_pool: Optional[dict]) -> None:
    if not isinstance(model, NumericalSolarBaseline) or candidate_pool is None:
        return
    weights = np.asarray(candidate_pool.get("weights"), dtype=np.float32)
    expected = (model.pred_len, len(NUMERICAL_CANDIDATE_NAMES))
    expected_dataset = (model.dataset_count, model.pred_len, len(NUMERICAL_CANDIDATE_NAMES))
    old_names = [name for name in candidate_pool.get("candidate_names", [])]
    if weights.ndim == 2 and weights.shape == (model.pred_len, len(old_names)) and old_names:
        upgraded = np.zeros(expected, dtype=np.float32)
        upgraded[:, NUMERICAL_CANDIDATE_NAMES.index("numeric_chain_final")] = 1.0
        for old_index, name in enumerate(old_names):
            if name in NUMERICAL_CANDIDATE_NAMES:
                upgraded[:, NUMERICAL_CANDIDATE_NAMES.index(name)] = weights[:, old_index]
        if "numeric_chain_final" not in old_names:
            upgraded[:, NUMERICAL_CANDIDATE_NAMES.index("numeric_chain_final")] = 0.0
        weights = upgraded
    elif weights.ndim == 3 and old_names and weights.shape[-1] == len(old_names):
        upgraded = np.zeros((weights.shape[0], model.pred_len, len(NUMERICAL_CANDIDATE_NAMES)), dtype=np.float32)
        upgraded[:, :, NUMERICAL_CANDIDATE_NAMES.index("numeric_chain_final")] = 1.0
        for old_index, name in enumerate(old_names):
            if name in NUMERICAL_CANDIDATE_NAMES:
                upgraded[:, :, NUMERICAL_CANDIDATE_NAMES.index(name)] = weights[:, :, old_index]
        if "numeric_chain_final" not in old_names:
            upgraded[:, :, NUMERICAL_CANDIDATE_NAMES.index("numeric_chain_final")] = 0.0
        weights = upgraded
    with torch.no_grad():
        if weights.shape == expected_dataset:
            model.dataset_candidate_pool_weights.copy_(
                torch.from_numpy(weights).to(model.dataset_candidate_pool_weights.device)
            )
            model.dataset_candidate_pool_enabled.fill_(True)
            model.candidate_pool_enabled.fill_(False)
        elif weights.shape == expected:
            model.candidate_pool_weights.copy_(torch.from_numpy(weights).to(model.candidate_pool_weights.device))
            model.candidate_pool_enabled.fill_(True)
            model.dataset_candidate_pool_enabled.fill_(False)
        else:
            raise ValueError(
                f"candidate pool weight shape mismatch: got {weights.shape}, "
                f"expected {expected} or {expected_dataset}"
            )


def write_numerical_candidate_pool_files(out_dir: str, candidate_pool: dict) -> None:
    weights = np.asarray(candidate_pool["weights"], dtype=np.float32)
    np.savez(
        os.path.join(out_dir, "numerical_candidate_pool.npz"),
        weights=weights,
        candidate_names=np.asarray(candidate_pool["candidate_names"]),
        variant=np.asarray(candidate_pool.get("variant", "")),
        ridge_lambda=np.asarray(candidate_pool.get("ridge_lambda", 0.0), dtype=np.float32),
    )
    serializable = {
        key: value
        for key, value in candidate_pool.items()
        if key != "weights"
    }
    if weights.ndim == 3:
        serializable["weights_mean"] = weights.mean(axis=(0, 1)).tolist()
        serializable["weights_first_dataset_first_horizon"] = weights[0, 0].tolist()
        serializable["weights_by_dataset_mean"] = weights.mean(axis=1).tolist()
    else:
        serializable["weights_mean"] = weights.mean(axis=0).tolist()
        serializable["weights_first_horizon"] = weights[0].tolist()
    with open(os.path.join(out_dir, "numerical_candidate_pool.json"), "w", encoding="utf-8") as handle:
        json.dump(serializable, handle, ensure_ascii=False, indent=2)


def _inverse_target_array(y_scaler, values: np.ndarray, dataset_index: Optional[np.ndarray] = None) -> np.ndarray:
    if dataset_index is not None and hasattr(y_scaler, "inverse_target_with_index"):
        return y_scaler.inverse_target_with_index(values, dataset_index)
    return y_scaler.inverse_target(values)


@torch.inference_mode()
def evaluate(
    model,
    loader,
    device,
    y_scaler,
    amp,
    text2_residual_alpha: Optional[np.ndarray] = None,
    residual_calibration: Optional[dict] = None,
    max_batches: int = 0,
):
    model.eval()
    predictions, targets, baselines, stage1_texts, corrections, neutral_corrections, realtime_deltas = [], [], [], [], [], [], []
    route_predictions, route_deltas, neutral_route_deltas, interventions, treatments, router_candidates, calibrated_candidates_all = [], [], [], [], [], [], []
    router_gates, router_top_probs, router_hard_probs, candidate_ranges = [], [], [], []
    realtime_confidences, realtime_opportunities, realtime_numeric_opportunities = [], [], []
    future_time_features_all = []
    dataset_indices = []
    calibration_feature_values: Dict[str, List[np.ndarray]] = {
        key: [] for key in RESIDUAL_CALIBRATION_FEATURES
    }
    branch_keys = [
        "ot_only_prediction",
        "ot_prediction",
        "aux_prediction",
        "past_nwp_prediction",
        "chain_prediction",
        "pre_periodic_prediction",
    ]
    branch_values: Dict[str, List[np.ndarray]] = {key: [] for key in branch_keys}
    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= int(max_batches):
            break
        batch = move_batch(batch, device)
        with _autocast_context(device, amp):
            prediction, parts = _forward_with_parts(model, batch)
        predictions.append(prediction.float().cpu().numpy())
        targets.append(batch["y"].float().cpu().numpy())
        if "dataset_index" in batch:
            dataset_indices.append(batch["dataset_index"].long().cpu().numpy().reshape(-1))
        baseline = parts.get("stage1_prediction", prediction)
        stage1_text = parts.get("stage1_text_prediction", baseline)
        correction = parts.get("correction", torch.zeros_like(prediction))
        neutral_correction = parts.get("neutral_text2_intervention", torch.zeros_like(prediction))
        realtime_delta = parts.get("realtime_delta", torch.zeros_like(prediction))
        route_prediction = parts.get("route_prediction", stage1_text)
        route_delta = parts.get("route_delta", torch.zeros_like(prediction))
        neutral_route_delta = parts.get("neutral_route_delta", torch.zeros_like(prediction))
        intervention = parts.get("text2_intervention", correction)
        treatment = parts.get("text2_treatment_effect", torch.zeros_like(prediction))
        baselines.append(baseline.float().cpu().numpy())
        stage1_texts.append(stage1_text.float().cpu().numpy())
        corrections.append(correction.float().cpu().numpy())
        neutral_corrections.append(neutral_correction.float().cpu().numpy())
        realtime_deltas.append(realtime_delta.float().cpu().numpy())
        route_predictions.append(route_prediction.float().cpu().numpy())
        route_deltas.append(route_delta.float().cpu().numpy())
        neutral_route_deltas.append(neutral_route_delta.float().cpu().numpy())
        interventions.append(intervention.float().cpu().numpy())
        treatments.append(treatment.float().cpu().numpy())
        if "router_candidates" in parts:
            router_candidates.append(parts["router_candidates"].float().cpu().numpy())
        if "calibrated_candidates" in parts:
            calibrated_candidates_all.append(parts["calibrated_candidates"].float().cpu().numpy())
        if "router_gate" in parts:
            router_gates.append(parts["router_gate"].float().cpu().numpy())
        if "router_top_prob" in parts:
            router_top_probs.append(parts["router_top_prob"].float().cpu().numpy())
        if "router_hard_prob" in parts:
            router_hard_probs.append(parts["router_hard_prob"].float().cpu().numpy())
        if "candidate_range" in parts:
            candidate_ranges.append(parts["candidate_range"].float().cpu().numpy())
        if "realtime_confidence" in parts:
            realtime_confidences.append(parts["realtime_confidence"].float().cpu().numpy())
        if "realtime_opportunity" in parts:
            realtime_opportunities.append(parts["realtime_opportunity"].float().cpu().numpy())
        if "realtime_numeric_opportunity" in parts:
            realtime_numeric_opportunities.append(
                parts["realtime_numeric_opportunity"].float().cpu().numpy()
            )
        for key in RESIDUAL_CALIBRATION_FEATURES:
            value = parts.get(key)
            if value is None or value.shape != prediction.shape:
                value = torch.zeros_like(prediction)
            calibration_feature_values[key].append(value.float().cpu().numpy())
        future_time_features_all.append(batch["future_time_features"].float().cpu().numpy())
        for key in branch_keys:
            value = parts.get(key)
            if value is not None and value.shape == prediction.shape:
                branch_values[key].append(value.float().cpu().numpy())

    pred_scaled = np.concatenate(predictions)
    target_scaled = np.concatenate(targets)
    dataset_index_array = np.concatenate(dataset_indices) if dataset_indices else None
    base_scaled = np.concatenate(baselines)
    stage1_text_scaled = np.concatenate(stage1_texts)
    correction_scaled = np.concatenate(corrections)
    neutral_correction_scaled = np.concatenate(neutral_corrections)
    realtime_delta_scaled = np.concatenate(realtime_deltas)
    route_prediction_scaled = np.concatenate(route_predictions)
    route_delta_scaled = np.concatenate(route_deltas)
    neutral_route_delta_scaled = np.concatenate(neutral_route_deltas)
    intervention_scaled = np.concatenate(interventions)
    treatment_scaled = np.concatenate(treatments)
    eval_pred_scaled = pred_scaled
    raw_pred_scaled = pred_scaled
    neutral_eval_scaled = stage1_text_scaled + neutral_correction_scaled
    calibration_features = {
        key: np.concatenate(values, axis=0)
        for key, values in calibration_feature_values.items()
        if values
    }
    calibration_delta_scaled = None
    calibration_name = None
    calibration_alpha = None
    release_budget = None
    normal_release_budget = None
    event_mix = None
    if residual_calibration is not None:
        calibration_name = str(residual_calibration.get("name", "multi_evidence"))
        feature_names = tuple(residual_calibration.get("features", ()))
        weights = np.asarray(residual_calibration.get("weights"), dtype=pred_scaled.dtype)
        cap = _safe_float(residual_calibration.get("cap", 0.0), 0.0)
        variant = str(residual_calibration.get("variant", "single_event_release"))

        if variant == "no_text2_release":
            calibration_delta_scaled = np.zeros_like(stage1_text_scaled, dtype=pred_scaled.dtype)
            release_budget = np.zeros_like(stage1_text_scaled, dtype=pred_scaled.dtype)
            normal_release_budget = np.ones_like(stage1_text_scaled, dtype=pred_scaled.dtype)
            event_mix = np.zeros_like(stage1_text_scaled, dtype=pred_scaled.dtype)
            eval_pred_scaled = stage1_text_scaled
            neutral_eval_scaled = stage1_text_scaled
            if weights.ndim != 2:
                weights = np.zeros((pred_scaled.shape[1], max(len(feature_names), 1)), dtype=pred_scaled.dtype)
        else:
            if not feature_names:
                raise ValueError("residual_calibration requires non-empty features")
            if weights.ndim != 2 or weights.shape[0] != pred_scaled.shape[1] or weights.shape[1] != len(feature_names):
                raise ValueError(
                    "residual_calibration weight shape mismatch: "
                    f"got {weights.shape}, expected {(pred_scaled.shape[1], len(feature_names))}"
                )
            missing_features = [name for name in feature_names if name not in calibration_features]
            if missing_features:
                raise ValueError(f"residual_calibration features are missing from evaluation arrays: {missing_features}")
            stacked = np.stack([calibration_features[name] for name in feature_names], axis=-1)

            if variant == "semantic_route_event_switch":
                event_weights = np.asarray(
                    residual_calibration.get("event_weights", weights),
                    dtype=pred_scaled.dtype,
                )
                expected_shape = (pred_scaled.shape[1], len(feature_names))
                if event_weights.shape != expected_shape:
                    raise ValueError(
                        "semantic route/event switch weight shape mismatch: "
                        f"event={event_weights.shape}, expected={expected_shape}"
                    )
                route_delta = np.sum(
                    stacked * weights.reshape(1, weights.shape[0], weights.shape[1]),
                    axis=-1,
                )
                event_delta = np.sum(
                    stacked * event_weights.reshape(1, event_weights.shape[0], event_weights.shape[1]),
                    axis=-1,
                )
                release_budget = _observable_release_budget(
                    calibration_features,
                    pred_scaled.shape,
                    profile=str(residual_calibration.get("budget_profile", "extreme")),
                )
                event_mix = _observable_regime_mix(
                    calibration_features,
                    pred_scaled.shape,
                    sharpness=_safe_float(residual_calibration.get("regime_mix_sharpness", 1.25), 1.25),
                    floor=_safe_float(residual_calibration.get("regime_mix_floor", 0.03), 0.03),
                    bias=_safe_float(residual_calibration.get("regime_mix_bias", 0.0), 0.0),
                )
                switch_transform = str(residual_calibration.get("switch_transform", "budget_mix"))
                if switch_transform == "budget_only":
                    event_switch = release_budget
                elif switch_transform == "sharp_budget":
                    event_switch = np.clip((release_budget - 0.06) / 0.34, 0.0, 1.0)
                    event_switch = np.maximum(event_switch, release_budget * event_mix)
                else:
                    event_switch = np.clip(release_budget * (0.35 + 0.65 * event_mix), 0.0, 1.0)
                normal_release_budget = 1.0 - event_switch
                calibration_delta_scaled = (
                    normal_release_budget * route_delta
                    + event_switch * event_delta
                )
                event_mix = event_switch
            elif variant == "pareto_regime":
                event_weights = np.asarray(
                    residual_calibration.get("event_weights", weights),
                    dtype=pred_scaled.dtype,
                )
                normal_weights = np.asarray(
                    residual_calibration.get("normal_weights", weights),
                    dtype=pred_scaled.dtype,
                )
                expected_shape = (pred_scaled.shape[1], len(feature_names))
                if event_weights.shape != expected_shape or normal_weights.shape != expected_shape:
                    raise ValueError(
                        "pareto residual calibration weight shape mismatch: "
                        f"event={event_weights.shape}, normal={normal_weights.shape}, expected={expected_shape}"
                    )
                event_delta = np.sum(
                    stacked * event_weights.reshape(1, event_weights.shape[0], event_weights.shape[1]),
                    axis=-1,
                )
                normal_delta = np.sum(
                    stacked * normal_weights.reshape(1, normal_weights.shape[0], normal_weights.shape[1]),
                    axis=-1,
                )
                release_budget = _observable_release_budget(
                    calibration_features,
                    pred_scaled.shape,
                    profile=str(residual_calibration.get("budget_profile", "balanced")),
                )
                normal_release_budget = _observable_release_budget(
                    calibration_features,
                    pred_scaled.shape,
                    profile=str(residual_calibration.get("normal_budget_profile", "normal_safe")),
                )
                event_mix = _observable_regime_mix(
                    calibration_features,
                    pred_scaled.shape,
                    sharpness=_safe_float(residual_calibration.get("regime_mix_sharpness", 1.25), 1.25),
                    floor=_safe_float(residual_calibration.get("regime_mix_floor", 0.03), 0.03),
                    bias=_safe_float(residual_calibration.get("regime_mix_bias", 0.0), 0.0),
                )
                calibration_delta_scaled = (
                    event_mix * event_delta * release_budget
                    + (1.0 - event_mix) * normal_delta * normal_release_budget
                )
            else:
                calibration_delta_scaled = np.sum(
                    stacked * weights.reshape(1, weights.shape[0], weights.shape[1]),
                    axis=-1,
                )
                if bool(residual_calibration.get("use_observable_release_budget", False)):
                    release_budget = _observable_release_budget(
                        calibration_features,
                        pred_scaled.shape,
                        profile=str(residual_calibration.get("budget_profile", "balanced")),
                    )
                    calibration_delta_scaled = calibration_delta_scaled * release_budget

            if cap > 0:
                calibration_delta_scaled = np.clip(calibration_delta_scaled, -cap, cap)
            release_scale = _safe_float(
                residual_calibration.get(
                    "style_release_scale",
                    residual_calibration.get("release_scale", 1.0),
                ),
                1.0,
            )
            calibration_delta_scaled = calibration_delta_scaled * float(np.clip(release_scale, 0.0, 1.0))
            eval_pred_scaled = stage1_text_scaled + calibration_delta_scaled
            neutral_eval_scaled = stage1_text_scaled + neutral_correction_scaled
    elif text2_residual_alpha is not None:
        calibration_alpha = np.asarray(text2_residual_alpha, dtype=pred_scaled.dtype).reshape(-1)
        if calibration_alpha.size == 1:
            calibration_alpha = np.full(pred_scaled.shape[1], float(calibration_alpha[0]), dtype=pred_scaled.dtype)
        elif calibration_alpha.size != pred_scaled.shape[1]:
            raise ValueError(
                f"text2_residual_alpha length mismatch: got {calibration_alpha.size}, expected {pred_scaled.shape[1]}"
            )
        eval_pred_scaled = stage1_text_scaled + intervention_scaled * calibration_alpha.reshape(1, -1)
        neutral_eval_scaled = stage1_text_scaled + neutral_correction_scaled * calibration_alpha.reshape(1, -1)
    # Main evaluation is intentionally kept in normalized target space.  This
    # keeps multi-station metrics comparable and prevents high-capacity stations
    # from dominating MSE/MAE after inverse scaling.
    pred_real = eval_pred_scaled
    target_real = target_scaled
    base_real = base_scaled
    stage1_text_real = stage1_text_scaled
    route_real = route_prediction_scaled
    stage1_plus_route_real = stage1_text_scaled + route_delta_scaled
    neutral_route_real = stage1_text_scaled + neutral_route_delta_scaled
    raw_pred_real = raw_pred_scaled
    neutral_pred_real = neutral_eval_scaled
    final_released_delta_scaled = eval_pred_scaled - stage1_text_scaled
    metrics = {
        f"scaled_{key}": value
        for key, value in regression_metrics(eval_pred_scaled, target_scaled).items()
    }
    metrics.update({f"real_{key}": value for key, value in regression_metrics(pred_real, target_real).items()})
    metrics["metric_space"] = "normalized_target"
    base_metrics = regression_metrics(base_real, target_real)
    stage1_text_metrics = regression_metrics(stage1_text_real, target_real)
    neutral_metrics = regression_metrics(neutral_pred_real, target_real)
    metrics.update({f"stage1_real_{key}": value for key, value in base_metrics.items()})
    metrics.update({f"stage1_text_real_{key}": value for key, value in stage1_text_metrics.items()})
    metrics.update({f"neutral_text_real_{key}": value for key, value in neutral_metrics.items()})
    route_metrics = regression_metrics(route_real, target_real)
    stage1_plus_route_metrics = regression_metrics(stage1_plus_route_real, target_real)
    neutral_route_metrics = regression_metrics(neutral_route_real, target_real)
    metrics.update({f"route_only_real_{key}": value for key, value in route_metrics.items()})
    metrics.update({f"stage1_plus_route_real_{key}": value for key, value in stage1_plus_route_metrics.items()})
    metrics.update({f"neutral_route_real_{key}": value for key, value in neutral_route_metrics.items()})
    metrics["route_only_improve_MSE"] = stage1_text_metrics["MSE"] - route_metrics["MSE"]
    metrics["stage1_plus_route_improve_MSE"] = stage1_text_metrics["MSE"] - stage1_plus_route_metrics["MSE"]
    metrics["route_text_vs_neutral_improve_MSE"] = neutral_route_metrics["MSE"] - stage1_plus_route_metrics["MSE"]
    if router_candidates:
        candidate_scaled = np.concatenate(router_candidates)
        candidate_real = candidate_scaled
        candidate_err = (candidate_real - target_real[..., None]) ** 2
        oracle_pred_real = np.take_along_axis(
            candidate_real,
            np.argmin(candidate_err, axis=-1)[..., None],
            axis=-1,
        )[..., 0]
        oracle_metrics = regression_metrics(oracle_pred_real, target_real)
        metrics.update({f"router_oracle_real_{key}": value for key, value in oracle_metrics.items()})
        metrics["router_oracle_improve_MSE"] = stage1_text_metrics["MSE"] - oracle_metrics["MSE"]
    if calibrated_candidates_all:
        calibrated_scaled = np.concatenate(calibrated_candidates_all)
        calibrated_real = calibrated_scaled
        calibrated_err = (calibrated_real - target_real[..., None]) ** 2
        calibrated_oracle_pred = np.take_along_axis(
            calibrated_real,
            np.argmin(calibrated_err, axis=-1)[..., None],
            axis=-1,
        )[..., 0]
        calibrated_oracle_metrics = regression_metrics(calibrated_oracle_pred, target_real)
        metrics.update({f"calibrated_oracle_real_{key}": value for key, value in calibrated_oracle_metrics.items()})
        metrics["calibrated_oracle_improve_MSE"] = stage1_text_metrics["MSE"] - calibrated_oracle_metrics["MSE"]
        route_err = (stage1_plus_route_real - target_real) ** 2
        oracle_err = np.min(calibrated_err, axis=-1)
        stage_err = (stage1_text_real - target_real) ** 2
        metrics["router_regret_MSE"] = float(np.mean(route_err - oracle_err))
        metrics["stage1_oracle_gap_MSE"] = float(np.mean(stage_err - oracle_err))
        metrics["router_gap_closed"] = float(
            np.mean(stage_err - route_err) / max(float(np.mean(stage_err - oracle_err)), 1e-8)
        )
        released = np.abs(route_delta_scaled) > 1e-5
        metrics["router_release_coverage"] = float(np.mean(released))
        if np.any(released):
            metrics["router_release_precision"] = float(np.mean(route_err[released] < stage_err[released]))
    if router_gates:
        metrics["router_gate_mean"] = float(np.mean(np.concatenate(router_gates)))
    if router_top_probs:
        metrics["router_top_prob_mean"] = float(np.mean(np.concatenate(router_top_probs)))
    if router_hard_probs:
        hard = np.concatenate(router_hard_probs)
        usage = hard.mean(axis=(0, 1))
        for index, value in enumerate(usage):
            name = ROUTER_CANDIDATE_NAMES[index] if index < len(ROUTER_CANDIDATE_NAMES) else f"candidate_{index}"
            metrics[f"router_usage_{name}"] = float(value)
    if candidate_ranges:
        metrics["candidate_range_scaled_mean"] = float(np.mean(np.concatenate(candidate_ranges)))
    if isinstance(model, NumericalSolarBaseline):
        metrics["numerical_candidate_pool_enabled"] = float(model.candidate_pool_enabled.item())
        if bool(model.candidate_pool_enabled.item()):
            weights = model.candidate_pool_weights.detach().float().cpu().numpy()
            for index, name in enumerate(NUMERICAL_CANDIDATE_NAMES):
                metrics[f"numerical_pool_weight_{name}"] = float(np.mean(weights[:, index]))
    metrics["improve_MSE"] = base_metrics["MSE"] - metrics["real_MSE"]
    metrics["improve_MAE"] = base_metrics["MAE"] - metrics["real_MAE"]
    metrics["numeric_to_final_gain_MSE"] = metrics["improve_MSE"]
    metrics["numeric_to_final_gain_MAE"] = metrics["improve_MAE"]
    metrics["stage1_text_improve_MSE"] = base_metrics["MSE"] - stage1_text_metrics["MSE"]
    metrics["numeric_to_text1_gain_MSE"] = metrics["stage1_text_improve_MSE"]
    metrics["numeric_to_text1_gain_MAE"] = base_metrics["MAE"] - stage1_text_metrics["MAE"]
    metrics["text2_residual_improve_MSE"] = stage1_text_metrics["MSE"] - metrics["real_MSE"]
    metrics["text1_to_final_gain_MSE"] = metrics["text2_residual_improve_MSE"]
    metrics["text1_to_final_gain_MAE"] = stage1_text_metrics["MAE"] - metrics["real_MAE"]
    metrics["text2_vs_neutral_improve_MSE"] = neutral_metrics["MSE"] - metrics["real_MSE"]
    metrics["text2_vs_neutral_improve_MAE"] = neutral_metrics["MAE"] - metrics["real_MAE"]
    if dataset_index_array is not None:
        dataset_names = list(getattr(y_scaler, "dataset_names", []))
        for dataset_id in sorted(np.unique(dataset_index_array).astype(int).tolist()):
            dataset_mask = dataset_index_array == int(dataset_id)
            if not np.any(dataset_mask):
                continue
            dataset_name = (
                str(dataset_names[dataset_id])
                if dataset_id < len(dataset_names)
                else f"dataset_{dataset_id}"
            )
            safe_name = re.sub(r"[^0-9A-Za-z_]+", "_", dataset_name)
            ds_pred = pred_real[dataset_mask]
            ds_target = target_real[dataset_mask]
            ds_base = base_real[dataset_mask]
            ds_text1 = stage1_text_real[dataset_mask]
            ds_metrics = regression_metrics(ds_pred, ds_target)
            ds_base_metrics = regression_metrics(ds_base, ds_target)
            ds_text1_metrics = regression_metrics(ds_text1, ds_target)
            prefix = f"dataset_{safe_name}"
            metrics[f"{prefix}_coverage"] = float(np.mean(dataset_mask))
            metrics[f"{prefix}_real_MSE"] = ds_metrics["MSE"]
            metrics[f"{prefix}_real_MAE"] = ds_metrics["MAE"]
            metrics[f"{prefix}_stage1_real_MSE"] = ds_base_metrics["MSE"]
            metrics[f"{prefix}_stage1_text_real_MSE"] = ds_text1_metrics["MSE"]
            metrics[f"{prefix}_numeric_to_text1_gain_MSE"] = (
                ds_base_metrics["MSE"] - ds_text1_metrics["MSE"]
            )
            metrics[f"{prefix}_text1_to_final_gain_MSE"] = (
                ds_text1_metrics["MSE"] - ds_metrics["MSE"]
            )
            metrics[f"{prefix}_numeric_to_final_gain_MSE"] = (
                ds_base_metrics["MSE"] - ds_metrics["MSE"]
            )
    if calibration_alpha is not None:
        raw_metrics = regression_metrics(raw_pred_real, target_real)
        metrics.update({f"raw_real_{key}": value for key, value in raw_metrics.items()})
        metrics["raw_improve_MSE"] = base_metrics["MSE"] - raw_metrics["MSE"]
        metrics["raw_text2_residual_improve_MSE"] = stage1_text_metrics["MSE"] - raw_metrics["MSE"]
        metrics["calibration_alpha_mean"] = float(np.mean(calibration_alpha))
        metrics["calibration_alpha_min"] = float(np.min(calibration_alpha))
        metrics["calibration_alpha_max"] = float(np.max(calibration_alpha))
    if calibration_delta_scaled is not None and residual_calibration is not None:
        raw_metrics = regression_metrics(raw_pred_real, target_real)
        metrics.update({f"raw_real_{key}": value for key, value in raw_metrics.items()})
        metrics["raw_improve_MSE"] = base_metrics["MSE"] - raw_metrics["MSE"]
        metrics["raw_text2_residual_improve_MSE"] = stage1_text_metrics["MSE"] - raw_metrics["MSE"]
        metrics["residual_calibration_name"] = calibration_name
        metrics["residual_calibration_variant"] = str(
            residual_calibration.get("variant", "single_event_release")
        )
        metrics["residual_calibration_lambda"] = _safe_float(residual_calibration.get("lambda", 0.0), 0.0)
        metrics["residual_calibration_cap"] = _safe_float(residual_calibration.get("cap", 0.0), 0.0)
        metrics["residual_calibration_val_weighted_scaled_MSE"] = _safe_float(
            residual_calibration.get("val_weighted_scaled_MSE", 0.0),
            0.0,
        )
        metrics["residual_calibration_val_scaled_MAE"] = _safe_float(
            residual_calibration.get("val_scaled_MAE", 0.0),
            0.0,
        )
        metrics["residual_calibration_val_scaled_improve_MAE"] = _safe_float(
            residual_calibration.get("val_scaled_improve_MAE", 0.0),
            0.0,
        )
        metrics["residual_calibration_val_weighted_scaled_MAE"] = _safe_float(
            residual_calibration.get("val_weighted_scaled_MAE", 0.0),
            0.0,
        )
        metrics["residual_calibration_val_weighted_improve_MSE"] = _safe_float(
            residual_calibration.get("val_weighted_scaled_improve_MSE", 0.0),
            0.0,
        )
        metrics["residual_calibration_val_weighted_improve_MAE"] = _safe_float(
            residual_calibration.get("val_weighted_scaled_improve_MAE", 0.0),
            0.0,
        )
        metrics["residual_calibration_val_delta_rms_scaled"] = _safe_float(
            residual_calibration.get("val_delta_rms_scaled", 0.0),
            0.0,
        )
        metrics["residual_calibration_val_event_score"] = _safe_float(
            residual_calibration.get("val_event_score", 0.0),
            0.0,
        )
        metrics["residual_calibration_val_high_residual_improve_MSE"] = _safe_float(
            residual_calibration.get("val_high_residual_improve_MSE", 0.0),
            0.0,
        )
        metrics["residual_calibration_val_ramp_event_improve_MSE"] = _safe_float(
            residual_calibration.get("val_ramp_event_improve_MSE", 0.0),
            0.0,
        )
        metrics["residual_calibration_val_extreme_union_improve_MSE"] = _safe_float(
            residual_calibration.get("val_extreme_union_improve_MSE", 0.0),
            0.0,
        )
        metrics["residual_calibration_val_normal_degrade_MSE"] = _safe_float(
            residual_calibration.get("val_normal_degrade_MSE", 0.0),
            0.0,
        )
        metrics["residual_calibration_val_normal_improve_MSE"] = _safe_float(
            residual_calibration.get("val_normal_improve_MSE", 0.0),
            0.0,
        )
        metrics["residual_calibration_style_release_scale"] = _safe_float(
            residual_calibration.get(
                "style_release_scale",
                residual_calibration.get("release_scale", 1.0),
            ),
            1.0,
        )
        metrics["residual_calibration_val_fold_global_gain_min"] = _safe_float(
            residual_calibration.get("val_fold_global_gain_min", 0.0),
            0.0,
        )
        metrics["residual_calibration_val_fold_event_gain_min"] = _safe_float(
            residual_calibration.get("val_fold_event_gain_min", 0.0),
            0.0,
        )
        metrics["residual_calibration_val_fold_global_positive_rate"] = _safe_float(
            residual_calibration.get("val_fold_global_positive_rate", 0.0),
            0.0,
        )
        metrics["residual_calibration_val_fold_event_positive_rate"] = _safe_float(
            residual_calibration.get("val_fold_event_positive_rate", 0.0),
            0.0,
        )
        metrics["residual_calibration_val_style_release_score"] = _safe_float(
            residual_calibration.get("val_style_release_score", 0.0),
            0.0,
        )
        metrics["residual_calibration_weight_mean_abs"] = float(np.mean(np.abs(residual_calibration["weights"])))
        if "event_weights" in residual_calibration:
            metrics["residual_calibration_event_weight_mean_abs"] = float(
                np.mean(np.abs(residual_calibration["event_weights"]))
            )
        if "normal_weights" in residual_calibration:
            metrics["residual_calibration_normal_weight_mean_abs"] = float(
                np.mean(np.abs(residual_calibration["normal_weights"]))
            )
        metrics["residual_calibration_delta_rms_scaled"] = float(np.sqrt(np.mean(calibration_delta_scaled ** 2)))
        if release_budget is not None:
            metrics["observable_release_budget_mean"] = float(np.mean(release_budget))
            metrics["observable_release_budget_coverage"] = float(np.mean(release_budget > 1e-4))
            metrics["observable_release_budget_rms"] = float(np.sqrt(np.mean(release_budget ** 2)))
            metrics["observable_release_budget_profile"] = str(
                residual_calibration.get("budget_profile", "balanced")
            )
        if normal_release_budget is not None:
            metrics["observable_normal_budget_mean"] = float(np.mean(normal_release_budget))
            metrics["observable_normal_budget_coverage"] = float(np.mean(normal_release_budget > 1e-4))
            metrics["observable_normal_budget_rms"] = float(np.sqrt(np.mean(normal_release_budget ** 2)))
            metrics["observable_normal_budget_profile"] = str(
                residual_calibration.get("normal_budget_profile", "normal_safe")
            )
        if event_mix is not None:
            metrics["observable_event_mix_mean"] = float(np.mean(event_mix))
            metrics["observable_event_mix_rms"] = float(np.sqrt(np.mean(event_mix ** 2)))
            metrics["observable_event_mix_profile"] = str(
                residual_calibration.get("regime_mix_profile", "custom")
            )
        for index, feature_name in enumerate(residual_calibration.get("features", ())):
            metrics[f"residual_calibration_wmean_{feature_name}"] = float(
                np.mean(residual_calibration["weights"][:, index])
            )
    metrics["correction_rms_scaled"] = float(np.sqrt(np.mean(correction_scaled ** 2)))
    metrics["text2_treatment_rms_scaled"] = float(np.sqrt(np.mean(treatment_scaled ** 2)))
    metrics["text2_treatment_abs_scaled"] = float(np.mean(np.abs(treatment_scaled)))
    for feature_name in [
        "dataset_evidence_delta",
        "dataset_evidence_high_delta",
        "dataset_evidence_low_delta",
        "dataset_evidence_release_gate",
        "text_scalar_direction_delta",
        "text_scalar_risk_delta",
        "realtime_evidence_delta",
        "nwp_disagreement_delta",
        "extreme_weather_evidence_delta",
        "fuzzy_ramp_evidence_delta",
        "shock_direction_delta",
        "semantic_risk_release_gate",
        "semantic_signed_risk_gate",
        "ramp_signed_risk_gate",
        "shock_signed_risk_gate",
        "text_scalar_signed_risk_gate",
        "extreme_counterfactual_delta",
        "extreme_proposal_delta",
        "extreme_dataset_delta",
        "extreme_realtime_delta",
        "extreme_nwp_disagreement_delta",
        "extreme_ramp_prior_delta",
        "event_counterfactual_basis_delta",
        "event_proposal_basis_delta",
        "event_dataset_basis_delta",
        "event_ramp_basis_delta",
        "event_realtime_basis_delta",
        "event_nwp_disagreement_basis_delta",
        "normal_trend_basis_delta",
        "stable_proposal_delta",
        "stable_dataset_delta",
        "observable_extreme_gate",
        "observable_stable_gate",
        "competitive_event_gate",
        "competitive_stable_gate",
        "competitive_event_treatment_delta",
        "competitive_trend_treatment_delta",
        "event_stability_margin_gate",
    ]:
        value = calibration_features.get(feature_name)
        if value is not None:
            metrics[f"{feature_name}_rms_scaled"] = float(np.sqrt(np.mean(value ** 2)))
            metrics[f"{feature_name}_mean_scaled"] = float(np.mean(value))
            metrics[f"{feature_name}_abs_scaled"] = float(np.mean(np.abs(value)))
    metrics["route_delta_rms_scaled"] = float(np.sqrt(np.mean(route_delta_scaled ** 2)))
    metrics["neutral_route_delta_rms_scaled"] = float(np.sqrt(np.mean(neutral_route_delta_scaled ** 2)))
    metrics["intervention_rms_scaled"] = float(np.sqrt(np.mean(intervention_scaled ** 2)))
    released_residual_mask = np.abs(final_released_delta_scaled) > 1e-5
    metrics["released_residual_coverage"] = float(np.mean(released_residual_mask))
    metrics["released_residual_abs_scaled"] = float(np.mean(np.abs(final_released_delta_scaled)))
    metrics["released_residual_rms_scaled"] = float(np.sqrt(np.mean(final_released_delta_scaled ** 2)))
    metrics["realtime_delta_rms_scaled"] = float(np.sqrt(np.mean(realtime_delta_scaled ** 2)))
    metrics["realtime_delta_abs_scaled"] = float(np.mean(np.abs(realtime_delta_scaled)))
    if realtime_confidences:
        metrics["realtime_confidence_mean"] = float(np.mean(np.concatenate(realtime_confidences)))
    if realtime_opportunities:
        metrics["realtime_opportunity_mean"] = float(np.mean(np.concatenate(realtime_opportunities)))
    if realtime_numeric_opportunities:
        metrics["realtime_numeric_opportunity_mean"] = float(
            np.mean(np.concatenate(realtime_numeric_opportunities))
        )
    future_time_features = np.concatenate(future_time_features_all, axis=0)
    daylight_mask = (
        daylight_weight(torch.as_tensor(future_time_features), 2.0).cpu().numpy() > 1.05
    )
    base_err = (stage1_text_real - target_real) ** 2
    high_residual_mask = base_err >= np.nanpercentile(base_err, 80.0)
    target_ramp = np.zeros_like(target_real)
    if target_real.shape[1] > 1:
        target_ramp[:, 1:] = target_real[:, 1:] - target_real[:, :-1]
    ramp_abs = np.abs(target_ramp)
    ramp_threshold = np.nanpercentile(ramp_abs, 85.0)
    event_mask = (ramp_abs >= max(float(ramp_threshold), 1e-6)) & daylight_mask
    extreme_union_mask = (high_residual_mask | event_mask) & daylight_mask
    normal_daylight_mask = daylight_mask & ~extreme_union_mask

    def add_masked_metrics(prefix: str, mask: np.ndarray) -> None:
        mask = np.asarray(mask, dtype=bool)
        metrics[f"{prefix}_coverage"] = float(mask.mean())
        if not mask.any():
            return
        pred_flat = pred_real[mask]
        target_flat = target_real[mask]
        stage_flat = stage1_text_real[mask]
        neutral_flat = neutral_pred_real[mask]
        pred_m = regression_metrics(pred_flat, target_flat)
        stage_m = regression_metrics(stage_flat, target_flat)
        neutral_m = regression_metrics(neutral_flat, target_flat)
        metrics[f"{prefix}_real_MSE"] = pred_m["MSE"]
        metrics[f"{prefix}_real_MAE"] = pred_m["MAE"]
        metrics[f"{prefix}_stage1_text_real_MSE"] = stage_m["MSE"]
        metrics[f"{prefix}_stage1_text_real_MAE"] = stage_m["MAE"]
        metrics[f"{prefix}_neutral_text_real_MSE"] = neutral_m["MSE"]
        metrics[f"{prefix}_neutral_text_real_MAE"] = neutral_m["MAE"]
        metrics[f"{prefix}_text2_residual_improve_MSE"] = stage_m["MSE"] - pred_m["MSE"]
        metrics[f"{prefix}_text2_vs_neutral_improve_MSE"] = neutral_m["MSE"] - pred_m["MSE"]

    add_masked_metrics("daylight", daylight_mask)
    add_masked_metrics("high_residual", high_residual_mask)
    add_masked_metrics("ramp_event", event_mask)
    add_masked_metrics("extreme_union", extreme_union_mask)
    add_masked_metrics("normal_daylight", normal_daylight_mask)
    if normal_daylight_mask.any():
        metrics["normal_daylight_degrade_MSE"] = max(
            0.0,
            metrics["normal_daylight_real_MSE"] - metrics["normal_daylight_stage1_text_real_MSE"],
        )
    else:
        metrics["normal_daylight_degrade_MSE"] = 0.0
    if extreme_union_mask.any():
        metrics["extreme_release_coverage"] = float(np.mean(released_residual_mask[extreme_union_mask]))
        metrics["extreme_release_abs_scaled"] = float(np.mean(np.abs(final_released_delta_scaled[extreme_union_mask])))
    else:
        metrics["extreme_release_coverage"] = 0.0
        metrics["extreme_release_abs_scaled"] = 0.0
    if normal_daylight_mask.any():
        metrics["normal_release_coverage"] = float(np.mean(released_residual_mask[normal_daylight_mask]))
        metrics["normal_release_abs_scaled"] = float(np.mean(np.abs(final_released_delta_scaled[normal_daylight_mask])))
    else:
        metrics["normal_release_coverage"] = 0.0
        metrics["normal_release_abs_scaled"] = 0.0
    released_count = float(np.sum(released_residual_mask))
    if released_count > 0:
        metrics["release_extreme_share"] = float(np.sum(released_residual_mask & extreme_union_mask) / released_count)
    else:
        metrics["release_extreme_share"] = 0.0
    for key, values in branch_values.items():
        if len(values) != len(predictions):
            continue
        branch_scaled = np.concatenate(values)
        branch_metrics = regression_metrics(branch_scaled, target_real)
        prefix = key.replace("_prediction", "")
        metrics[f"{prefix}_real_MSE"] = branch_metrics["MSE"]
        metrics[f"{prefix}_real_MAE"] = branch_metrics["MAE"]
    for step in [1, 4, 12, 24, 48, 96]:
        if step <= pred_real.shape[1]:
            step_metrics = regression_metrics(pred_real[:, step - 1], target_real[:, step - 1])
            base_step = regression_metrics(base_real[:, step - 1], target_real[:, step - 1])
            metrics.update({f"h{step:03d}_{key}": value for key, value in step_metrics.items()})
            metrics[f"h{step:03d}_stage1_MSE"] = base_step["MSE"]
            metrics[f"h{step:03d}_improve_MSE"] = base_step["MSE"] - step_metrics["MSE"]
    return metrics, pred_real, target_real, base_real, intervention_scaled


def save_model(path: str, model, args, metadata: dict, role: str, extra: Optional[dict] = None):
    model_metadata = metadata
    if extra:
        model_metadata = {**metadata, **extra}
    payload = {
        "model": model.state_dict(),
        "args": _args_metadata(args),
        "metadata": model_metadata,
        "code_version": CODE_VERSION,
        "method_name": METHOD_NAME,
        "method_full_name": METHOD_FULL_NAME,
        "method_core_claim": METHOD_CORE_CLAIM,
        "method_pipeline": METHOD_PIPELINE,
        "method_card": METHOD_CARD,
        "checkpoint_role": role,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def _clone_trainable_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: value.detach().float().cpu().clone()
        for name, value in model.state_dict().items()
        if value.is_floating_point()
    }


def _update_ema_state(
    ema_state: Optional[Dict[str, torch.Tensor]],
    model: nn.Module,
    decay: float,
) -> Dict[str, torch.Tensor]:
    current = _clone_trainable_state(model)
    if ema_state is None:
        return current
    decay = float(np.clip(decay, 0.0, 0.9999))
    for name, value in current.items():
        if name in ema_state and ema_state[name].shape == value.shape:
            ema_state[name].mul_(decay).add_(value, alpha=1.0 - decay)
        else:
            ema_state[name] = value
    return ema_state


def _load_ema_state(model: nn.Module, ema_state: Dict[str, torch.Tensor]) -> None:
    state = model.state_dict()
    for name, value in ema_state.items():
        if name in state and state[name].shape == value.shape and state[name].is_floating_point():
            state[name].copy_(value.to(device=state[name].device, dtype=state[name].dtype))
    model.load_state_dict(state, strict=True)


def _update_swa_state(
    swa_state: Optional[Dict[str, torch.Tensor]],
    model: nn.Module,
    count: int,
) -> Tuple[Dict[str, torch.Tensor], int]:
    current = _clone_trainable_state(model)
    count = int(count)
    if swa_state is None or count <= 0:
        return current, 1
    next_count = count + 1
    for name, value in current.items():
        if name in swa_state and swa_state[name].shape == value.shape:
            swa_state[name].mul_(count / next_count).add_(value, alpha=1.0 / next_count)
        else:
            swa_state[name] = value
    return swa_state, next_count


def _residual_calibration_npz_payload(residual_calibration: dict) -> dict:
    payload = {
        "weights": np.asarray(residual_calibration["weights"], dtype=np.float32),
        "features": np.asarray(residual_calibration["features"]),
        "name": np.asarray(residual_calibration["name"]),
        "variant": np.asarray(residual_calibration.get("variant", "single_event_release")),
        "ridge_lambda": np.asarray(residual_calibration["lambda"], dtype=np.float32),
        "cap": np.asarray(residual_calibration["cap"], dtype=np.float32),
        "budget_profile": np.asarray(residual_calibration.get("budget_profile", "")),
        "normal_budget_profile": np.asarray(residual_calibration.get("normal_budget_profile", "")),
        "regime_mix_profile": np.asarray(residual_calibration.get("regime_mix_profile", "")),
    }
    if "event_weights" in residual_calibration:
        payload["event_weights"] = np.asarray(residual_calibration["event_weights"], dtype=np.float32)
    if "normal_weights" in residual_calibration:
        payload["normal_weights"] = np.asarray(residual_calibration["normal_weights"], dtype=np.float32)
    for key in (
        "regime_mix_sharpness",
        "regime_mix_floor",
        "regime_mix_bias",
        "val_scaled_MSE",
        "val_scaled_MAE",
        "val_scaled_improve_MSE",
        "val_scaled_improve_MAE",
        "val_weighted_scaled_MSE",
        "val_weighted_scaled_MAE",
        "val_weighted_scaled_improve_MSE",
        "val_weighted_scaled_improve_MAE",
        "val_delta_rms_scaled",
        "val_release_coverage",
        "val_normal_release_coverage",
        "val_extreme_release_coverage",
        "val_event_score",
        "val_high_residual_improve_MSE",
        "val_ramp_event_improve_MSE",
        "val_extreme_union_improve_MSE",
        "val_normal_improve_MSE",
        "val_normal_degrade_MSE",
        "val_event_mix_mean",
        "val_event_mix_event_mean",
        "val_event_mix_normal_mean",
        "style_release_scale",
        "release_scale",
        "val_fold_global_gain_min",
        "val_fold_global_gain_mean",
        "val_fold_global_positive_rate",
        "val_fold_event_gain_min",
        "val_fold_event_gain_mean",
        "val_fold_event_positive_rate",
        "val_fold_normal_degrade_max",
        "val_fold_instability",
        "val_style_release_score",
    ):
        payload[key] = np.asarray(
            _safe_float(residual_calibration.get(key, 0.0), 0.0),
            dtype=np.float32,
        )
    return payload


def write_residual_calibration_files(out_dir: str, stem: str, residual_calibration: dict) -> None:
    np.savez(
        os.path.join(out_dir, f"{stem}.npz"),
        **_residual_calibration_npz_payload(residual_calibration),
    )
    serializable = {
        key: value
        for key, value in residual_calibration.items()
        if key not in {"weights", "event_weights", "normal_weights"}
    }
    serializable["features"] = list(residual_calibration["features"])
    serializable["weight_mean"] = residual_calibration["weights"].mean(axis=0).tolist()
    if "event_weights" in residual_calibration:
        serializable["event_weight_mean"] = residual_calibration["event_weights"].mean(axis=0).tolist()
    if "normal_weights" in residual_calibration:
        serializable["normal_weight_mean"] = residual_calibration["normal_weights"].mean(axis=0).tolist()
    with open(os.path.join(out_dir, f"{stem}.json"), "w", encoding="utf-8") as handle:
        json.dump(serializable, handle, ensure_ascii=False, indent=2)


def make_solar_loaders(args, text1_cache=None, text2_cache=None):
    if bool(getattr(args, "multi_dataset", False)):
        return make_multi_loaders(
            dataset_specs=getattr(args, "dataset_specs", None) or _resolve_dataset_specs(args),
            feature_cols=solar_feature_cols(),
            target_col=TARGET,
            nwp_cols=tuple(SOLAR_NWP_COLS),
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            text1_caches=text1_cache if isinstance(text1_cache, dict) else None,
            text2_caches=text2_cache if isinstance(text2_cache, dict) else None,
            use_past_nwp=True,
            use_future_nwp=True,
            use_text1=text1_cache is not None,
            use_text2=text2_cache is not None,
            strict_frequency=True,
            balanced_train_sampling=bool(getattr(args, "balanced_dataset_sampling", True)),
            event_balanced_sampling=bool(getattr(args, "event_balanced_sampling", True)),
            event_sample_quantile=float(getattr(args, "event_sample_quantile", 0.70)),
            event_sample_alpha=float(getattr(args, "event_sample_alpha", 1.50)),
            event_sample_max_weight=float(getattr(args, "event_sample_max_weight", 4.00)),
            prediction_start_hour=args.prediction_start_hour,
            prediction_end_hour=args.prediction_end_hour,
            prediction_include_end=bool(args.prediction_include_end),
        )
    return make_loaders(
        primary_csv=args.solar_csv,
        aux_csv=args.solar_aux_csv,
        nwp_csv=args.solar_nwp_csv,
        context_csv=args.context_csv,
        feature_cols=solar_feature_cols(),
        target_col=TARGET,
        nwp_cols=tuple(SOLAR_NWP_COLS),
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        text1_cache=text1_cache,
        text2_cache=text2_cache,
        use_past_nwp=True,
        use_future_nwp=True,
        use_text1=text1_cache is not None,
        use_text2=text2_cache is not None,
        strict_frequency=True,
        event_balanced_sampling=bool(getattr(args, "event_balanced_sampling", True)),
        event_sample_quantile=float(getattr(args, "event_sample_quantile", 0.70)),
        event_sample_alpha=float(getattr(args, "event_sample_alpha", 1.50)),
        event_sample_max_weight=float(getattr(args, "event_sample_max_weight", 4.00)),
        prediction_start_hour=args.prediction_start_hour,
        prediction_end_hour=args.prediction_end_hour,
        prediction_include_end=bool(args.prediction_include_end),
    )


def train_and_select(
    args,
    model,
    loaders,
    device: torch.device,
    amp: bool,
    out_dir: str,
    experiment_name: str,
    checkpoint_extra: Optional[dict] = None,
    require_zero_initial_correction: bool = False,
    allow_epoch0_fallback: bool = False,
):
    train_loader, val_loader, test_loader, _, y_scaler, _, metadata = loaders
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.min_lr
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp) if amp else None
    weights = LossWeights(args)

    ensure_dir(out_dir)
    best_safe_path = os.path.join(out_dir, "best_safe.pt")
    best_trained_path = os.path.join(out_dir, "best_trained.pt")
    best_ema_path = os.path.join(out_dir, "best_ema.pt")
    best_swa_path = os.path.join(out_dir, "best_swa.pt")
    best_accepted_path = os.path.join(out_dir, ACCEPTED_CHECKPOINT_NAME)
    best_alias_path = os.path.join(out_dir, "best.pt")
    train_log_path = os.path.join(out_dir, "train_log.csv")
    history: List[dict] = []
    patience_count = 0
    trained_best_val = float("inf")
    trained_best_epoch: Optional[int] = None
    ema_state: Optional[Dict[str, torch.Tensor]] = None
    ema_best_val = float("inf")
    ema_best_epoch: Optional[int] = None
    swa_state: Optional[Dict[str, torch.Tensor]] = None
    swa_count = 0
    swa_best_val = float("inf")
    swa_best_epoch: Optional[int] = None

    print(f"\n===== {experiment_name} | train/val/test={metadata['split_sizes']} =====", flush=True)
    init_metrics, _, _, _, init_correction = evaluate(model, val_loader, device, y_scaler, amp=False)
    max_correction = float(np.max(np.abs(init_correction)))
    if require_zero_initial_correction and max_correction > args.init_tolerance:
        raise RuntimeError(f"initial correction is not zero: {max_correction}")
    safe_best_val = init_metrics["real_MSE"]
    safe_best_epoch = 0
    save_model(
        best_safe_path,
        model,
        args,
        metadata,
        "safe_epoch0",
        {"initial_max_correction": max_correction, **(checkpoint_extra or {})},
    )
    history.append(
        {
            "epoch": 0,
            "train_loss": np.nan,
            "lr": 0.0,
            "seconds": 0.0,
            **{f"val_{key}": value for key, value in init_metrics.items()},
        }
    )
    pd.DataFrame(history).to_csv(train_log_path, index=False)
    print(
        f"epoch=000 base val_MSE={safe_best_val:.6f} "
        f"max_correction={max_correction:.3g}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_terms, ema_state = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            amp,
            weights,
            args.grad_clip,
            ema_state=ema_state,
            ema_decay=(
                getattr(args, "ema_decay", 0.98)
                if bool(getattr(args, "use_weight_ema", True))
                else None
            ),
        )
        val_metrics, _, _, _, _ = evaluate(model, val_loader, device, y_scaler, amp)
        ema_metrics = None
        swa_metrics = None
        if ema_state is not None and epoch >= int(getattr(args, "ema_start_epoch", 2)):
            live_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            _load_ema_state(model, ema_state)
            ema_metrics, _, _, _, _ = evaluate(model, val_loader, device, y_scaler, amp=False)
            model.load_state_dict(live_state, strict=True)
        swa_start_epoch = int(getattr(args, "swa_start_epoch", 0) or max(2, args.epochs // 2))
        if bool(getattr(args, "use_weight_swa", True)) and epoch >= swa_start_epoch:
            swa_state, swa_count = _update_swa_state(swa_state, model, swa_count)
            live_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            _load_ema_state(model, swa_state)
            swa_metrics, _, _, _, _ = evaluate(model, val_loader, device, y_scaler, amp=False)
            model.load_state_dict(live_state, strict=True)
        scheduler.step()
        val_mse = val_metrics["real_MSE"]
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - start,
            **{f"train_{key}": value for key, value in train_terms.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        pd.DataFrame(history).to_csv(train_log_path, index=False)
        extra = ""
        if "history_trust" in train_terms:
            extra = (
                f" hist_g={train_terms.get('history_trust', 0.0):.3f}"
                f" nwp_g={train_terms.get('nwp_trust', 0.0):.3f}"
                f" per_g={train_terms.get('periodic_trust', 0.0):.3f}"
                f" aux_g={train_terms.get('aux_gate', 0.0):.3f}"
                f" past_g={train_terms.get('past_nwp_gate', 0.0):.3f}"
                f" fut_g={train_terms.get('future_nwp_gate', 0.0):.3f}"
                f" ot={train_terms.get('ot_only_error', 0.0):.3f}"
                f" aux={train_terms.get('aux_error', 0.0):.3f}"
                f" past={train_terms.get('past_nwp_error', 0.0):.3f}"
                f" chain={train_terms.get('chain_error', 0.0):.3f}"
            )
        elif "opportunity_gate" in train_terms:
            extra = (
                f" rt_d={train_terms.get('realtime_delta_abs', 0.0):.4f}"
                f" rt_c={train_terms.get('realtime_confidence', 0.0):.3f}"
                f" rt_o={train_terms.get('realtime_opportunity', 0.0):.3f}"
                f" rt_no={train_terms.get('realtime_numeric_opportunity', 0.0):.3f}"
                f" gate={train_terms.get('opportunity_gate', 0.0):.3f}"
                f" conf={train_terms.get('confidence', 0.0):.3f}"
                f" shock={train_terms.get('shock_abs', 0.0):.4f}"
                f" rgate={train_terms.get('router_gate', 0.0):.3f}"
                f" safe={train_terms.get('safe_release_gate', 0.0):.3f}"
                f" rdel={train_terms.get('route_delta_abs', 0.0):.4f}"
                f" rtop={train_terms.get('router_top_prob', 0.0):.3f}"
                f" rorc={train_terms.get('router_oracle_gain', 0.0):.4f}"
                f" rdst={train_terms.get('route_distill', 0.0):.4f}"
            )
        print(
            f"epoch={epoch:03d} train={train_loss:.6f} "
            f"val_MSE={val_mse:.6f} "
            f"ema_MSE={(ema_metrics or {}).get('real_MSE', 0.0):.6f} "
            f"swa_MSE={(swa_metrics or {}).get('real_MSE', 0.0):.6f} "
            f"base={val_metrics['stage1_real_MSE']:.6f} "
            f"improve={val_metrics['improve_MSE']:.6f} "
            f"rt={val_metrics.get('realtime_delta_rms_scaled', 0.0):.4f} "
            f"corr={val_metrics['correction_rms_scaled']:.4f} "
            f"route={val_metrics.get('route_delta_rms_scaled', 0.0):.4f}"
            f"{extra}",
            flush=True,
        )
        if trained_best_epoch is None or val_mse < trained_best_val - args.min_delta:
            trained_best_val = val_mse
            trained_best_epoch = epoch
            patience_count = 0
            save_model(
                best_trained_path,
                model,
                args,
                metadata,
                "best_trained",
                {
                    "best_epoch": epoch,
                    "best_val_real_MSE": val_mse,
                    **(checkpoint_extra or {}),
                },
            )
        else:
            patience_count += 1
        if ema_metrics is not None and ema_metrics["real_MSE"] < ema_best_val - args.min_delta:
            ema_best_val = ema_metrics["real_MSE"]
            ema_best_epoch = epoch
            live_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            _load_ema_state(model, ema_state)
            save_model(
                best_ema_path,
                model,
                args,
                metadata,
                "best_ema",
                {
                    "best_epoch": epoch,
                    "best_val_real_MSE": ema_best_val,
                    "ema_decay": getattr(args, "ema_decay", 0.98),
                    **(checkpoint_extra or {}),
                },
            )
            model.load_state_dict(live_state, strict=True)
        if swa_metrics is not None and swa_metrics["real_MSE"] < swa_best_val - args.min_delta:
            swa_best_val = swa_metrics["real_MSE"]
            swa_best_epoch = epoch
            live_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            _load_ema_state(model, swa_state)
            save_model(
                best_swa_path,
                model,
                args,
                metadata,
                "best_swa",
                {
                    "best_epoch": epoch,
                    "best_val_real_MSE": swa_best_val,
                    "swa_count": swa_count,
                    **(checkpoint_extra or {}),
                },
            )
            model.load_state_dict(live_state, strict=True)
        if val_mse < safe_best_val - args.min_delta:
            safe_best_val = val_mse
            safe_best_epoch = epoch
            save_model(
                best_safe_path,
                model,
                args,
                metadata,
                "best_safe",
                {
                    "best_epoch": epoch,
                    "best_val_real_MSE": val_mse,
                    **(checkpoint_extra or {}),
                },
            )
        if patience_count >= args.patience:
            print(f"early stopping at epoch {epoch}", flush=True)
            break

    pd.DataFrame(history).to_csv(train_log_path, index=False)
    if not os.path.exists(best_trained_path):
        raise RuntimeError("no trained checkpoint was saved")

    accepted_path = best_trained_path
    accepted_role = "best_trained"
    accepted_epoch = trained_best_epoch
    accepted_val = trained_best_val
    accepted_new_stage = True
    if os.path.exists(best_ema_path) and ema_best_val < accepted_val - args.min_delta:
        accepted_path = best_ema_path
        accepted_role = "best_ema"
        accepted_epoch = ema_best_epoch
        accepted_val = ema_best_val
        accepted_new_stage = True
    if os.path.exists(best_swa_path) and swa_best_val < accepted_val - args.min_delta:
        accepted_path = best_swa_path
        accepted_role = "best_swa"
        accepted_epoch = swa_best_epoch
        accepted_val = swa_best_val
        accepted_new_stage = True
    if allow_epoch0_fallback and safe_best_val <= trained_best_val + args.acceptance_tolerance:
        accepted_path = best_safe_path
        fallback_label = "epoch0_fallback"
        if checkpoint_extra:
            model_role = str(checkpoint_extra.get("model_role", "")).strip()
            if model_role:
                fallback_label = f"{model_role}_epoch0_fallback"
        accepted_role = "best_safe" if safe_best_epoch != 0 else fallback_label
        accepted_epoch = safe_best_epoch
        accepted_val = safe_best_val
        accepted_new_stage = safe_best_epoch != 0
    shutil.copy2(accepted_path, best_accepted_path)
    shutil.copy2(best_accepted_path, best_alias_path)
    accepted = torch.load(best_accepted_path, map_location=device, weights_only=False)
    accepted["metadata"] = {
        **accepted.get("metadata", {}),
        "accepted_role": accepted_role,
        "accepted_epoch": accepted_epoch,
        "accepted_val_real_MSE": accepted_val,
        "accepted_new_stage": accepted_new_stage,
    }
    torch.save(accepted, best_accepted_path)
    torch.save(accepted, best_alias_path)
    baseline_candidate_pool = None
    if isinstance(model, NumericalSolarBaseline) and bool(
        getattr(args, "use_numerical_candidate_pool", True)
    ):
        accepted = torch.load(best_accepted_path, map_location=device, weights_only=False)
        model.load_state_dict(accepted["model"], strict=True)
        candidate_arrays = collect_numerical_candidate_arrays(model, val_loader, device, amp)
        baseline_candidate_pool = fit_numerical_candidate_pool(candidate_arrays)
        if baseline_candidate_pool is not None:
            apply_numerical_candidate_pool(model, baseline_candidate_pool)
            accepted["model"] = model.state_dict()
            accepted["numerical_candidate_pool"] = {
                **baseline_candidate_pool,
                "weights": np.asarray(baseline_candidate_pool["weights"], dtype=np.float32),
            }
            accepted["metadata"] = {
                **accepted.get("metadata", {}),
                "numerical_candidate_pool": {
                    key: (
                        np.asarray(value).tolist()
                        if key == "weights"
                        else _json_safe(value)
                    )
                    for key, value in baseline_candidate_pool.items()
                },
            }
            torch.save(accepted, best_accepted_path)
            torch.save(accepted, best_alias_path)
            write_numerical_candidate_pool_files(out_dir, baseline_candidate_pool)
            print(
                "numerical candidate pool fitted: "
                f"val_gain={baseline_candidate_pool.get('val_scaled_improve_MSE', 0.0):.6f}, "
                f"fold_min={baseline_candidate_pool.get('val_fold_gain_min', 0.0):.6f}, "
                f"variant={baseline_candidate_pool.get('variant', '')}"
            )

    rows = []
    for role, path in [
        ("trained", best_trained_path),
        ("safe", best_safe_path),
        ("accepted", best_accepted_path),
    ]:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        calibration_alpha = None
        residual_calibration = None
        skip_posthoc_text2_release = (
            role == "accepted"
            and bool(getattr(args, "safe_text2_fallback", True))
            and not bool(accepted_new_stage)
        )
        use_text2_posthoc = (
            role == "accepted"
            and hasattr(model, "use_text2_correction")
            and bool(getattr(model, "use_text2_correction"))
            and not skip_posthoc_text2_release
        )
        if use_text2_posthoc:
            calibration_batches = int(getattr(args, "calibration_max_batches", 0) or 0)
            val_arrays = collect_text2_calibration_arrays(
                model,
                val_loader,
                device,
                amp,
                max_batches=calibration_batches,
            )
            if val_arrays is not None:
                stage1_text_val, correction_val, target_val = val_arrays
                calibration_alpha = fit_text2_residual_alpha(
                    stage1_text_val, correction_val, target_val
                )
                np.save(os.path.join(out_dir, f"text2_residual_alpha_{role}.npy"), calibration_alpha)
            residual_arrays = collect_residual_calibration_arrays(
                model,
                val_loader,
                device,
                amp,
                max_batches=calibration_batches,
            )
            residual_calibration = fit_residual_calibration(
                residual_arrays,
                search_all_specs=bool(getattr(args, "calibration_search", False)),
                fast=bool(getattr(args, "calibration_fast", False)),
                safe_fallback=bool(getattr(args, "safe_text2_fallback", True)),
                min_global_gain=float(getattr(args, "safe_text2_min_global_gain", 1e-4)),
                min_extreme_gain=float(getattr(args, "safe_text2_min_extreme_gain", 5e-4)),
                force_candidate_name=str(getattr(args, "force_calibration_name", "") or ""),
                force_candidate_variant=str(getattr(args, "force_calibration_variant", "") or ""),
                force_release_scale=float(getattr(args, "force_release_scale", -1.0)),
            )
            if residual_calibration is not None:
                write_residual_calibration_files(
                    out_dir,
                    f"residual_calibration_{role}",
                    residual_calibration,
                )
        elif skip_posthoc_text2_release:
            print(
                "accepted checkpoint is an epoch0 fallback; "
                "skipping post-hoc Text2 residual release"
            )
        metrics, prediction, target, baseline, correction = evaluate(
            model,
            test_loader,
            device,
            y_scaler,
            amp,
            text2_residual_alpha=calibration_alpha,
            residual_calibration=residual_calibration,
        )
        if args.save_arrays:
            np.save(os.path.join(out_dir, f"prediction_{role}_normalized.npy"), prediction)
            np.save(os.path.join(out_dir, f"correction_{role}_scaled.npy"), correction)
            if role == "accepted":
                np.save(os.path.join(out_dir, "target_normalized.npy"), target)
                np.save(os.path.join(out_dir, "baseline_prediction_normalized.npy"), baseline)
                if calibration_alpha is not None:
                    np.save(os.path.join(out_dir, "text2_residual_alpha.npy"), calibration_alpha)
                if residual_calibration is not None:
                    write_residual_calibration_files(out_dir, "residual_calibration", residual_calibration)
        pd.DataFrame([metrics]).to_csv(
            os.path.join(out_dir, f"test_metrics_{role}.csv"), index=False
        )
        rows.append((role, metrics))

    accepted_metrics = dict(rows[-1][1])
    metadata_out = {
        **metadata,
        "code_version": CODE_VERSION,
        "method_name": METHOD_NAME,
        "method_full_name": METHOD_FULL_NAME,
        "method_core_claim": METHOD_CORE_CLAIM,
        "method_pipeline": list(METHOD_PIPELINE),
        "method_card": METHOD_CARD,
        "experiment_name": experiment_name,
        "best_trained_epoch": trained_best_epoch,
        "best_trained_val_real_MSE": trained_best_val,
        "best_safe_epoch": safe_best_epoch,
        "best_safe_val_real_MSE": safe_best_val,
        "best_accepted_epoch": accepted_epoch,
        "best_accepted_val_real_MSE": accepted_val,
        "accepted_role": accepted_role,
        "accepted_new_stage": accepted_new_stage,
        "paper_checkpoint": ACCEPTED_CHECKPOINT_NAME,
        "args": _args_metadata(args),
        **(checkpoint_extra or {}),
    }
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata_out, handle, ensure_ascii=False, indent=2)

    result = {
        "experiment": experiment_name,
        "checkpoint": best_alias_path,
        "accepted_role": accepted_role,
        "accepted_new_stage": accepted_new_stage,
        "accepted_val_real_MSE": accepted_val,
        **accepted_metrics,
    }
    pd.DataFrame([result]).to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    paper_core = {
        "method_name": METHOD_NAME,
        "method_full_name": METHOD_FULL_NAME,
        "paper_mainline_name": PAPER_MAINLINE_NAME,
        "code_version": CODE_VERSION,
        "experiment": experiment_name,
        "accepted_role": accepted_role,
        "accepted_new_stage": accepted_new_stage,
        "checkpoint": best_alias_path,
    }
    for key in PAPER_CORE_METRIC_KEYS:
        if key in accepted_metrics:
            paper_core[key] = accepted_metrics[key]
    pd.DataFrame([paper_core]).to_csv(os.path.join(out_dir, "paper_core_metrics.csv"), index=False)
    with open(os.path.join(out_dir, "paper_core_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(paper_core, handle, ensure_ascii=False, indent=2)
    print(
        f"TEST accepted MSE={accepted_metrics['real_MSE']:.6f}, "
        f"MAE={accepted_metrics['real_MAE']:.6f}, "
        f"base_MSE={accepted_metrics['stage1_real_MSE']:.6f}, "
        f"text1_gain={accepted_metrics.get('numeric_to_text1_gain_MSE', 0.0):.6f}, "
        f"text2_gain={accepted_metrics.get('text1_to_final_gain_MSE', 0.0):.6f}, "
        f"extreme_gain={accepted_metrics.get('extreme_union_text2_residual_improve_MSE', 0.0):.6f}, "
        f"normal_degrade={accepted_metrics.get('normal_daylight_degrade_MSE', 0.0):.6f}, "
        f"accepted_role={accepted_role}"
    )
    return result


def run_baseline(args):
    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    amp = bool(args.amp and device.type == "cuda")
    loaders = make_solar_loaders(args)
    model = build_baseline_model(args, device)
    return train_and_select(
        args,
        model,
        loaders,
        device,
        amp,
        args.baseline_out_dir,
        "InternalNumericalFoundation_iTransformerResidualChain",
        checkpoint_extra={
            "model_role": "internal_numerical_foundation",
            "paper_stage": "numerical_foundation",
            "paper_visibility": "internal_pretraining;not_main_ablation_result",
            "numeric_foundation_profile": getattr(args, "numeric_foundation_profile", "auto"),
            "numeric_foundation_profile_effective": getattr(
                args, "numeric_foundation_profile_effective", "unknown"
            ),
            "numeric_foundation_policy": (
                "robust history/NWP/periodic numerical candidate pool"
                if getattr(args, "numeric_foundation_profile_effective", "") == "robust"
                else "compact residual-chain numerical candidate pool"
            ),
            "method_card": METHOD_CARD,
        },
    )


def build_fusion_model(
    args,
    baseline,
    text1_dim: int,
    text2_dim: int,
    use_text2_correction: bool,
):
    return SolarUnifiedTextFusionModel(
        baseline_model=baseline,
        input_dim=len(solar_feature_cols()),
        text1_dim=text1_dim,
        text2_dim=text2_dim,
        nwp_dim=len(SOLAR_NWP_COLS),
        pred_len=args.pred_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        text_layers=args.text_layers,
        nwp_layers=args.nwp_layers,
        experts=args.experts,
        basis_rank=args.basis_rank,
        dropout=args.dropout,
        cutoff_index=args.cutoff_index,
        max_delta=args.max_delta,
        text2_max_delta=args.text2_max_delta,
        risk_horizon_hours=args.risk_horizon_hours,
        realtime_slots_per_field=args.realtime_slots_per_field,
        freeze_baseline=args.freeze_baseline,
        use_realtime_condition=args.use_realtime_condition,
        use_fuzzy_extreme=args.use_fuzzy_extreme,
        use_shock_evidence=args.use_shock_evidence,
        use_low_text2=args.use_low_text2,
        use_high_text2=args.use_high_text2,
        use_text_router=args.use_text_router,
        use_text2_correction=use_text2_correction,
    )


def run_text1(args):
    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    amp = bool(args.amp and device.type == "cuda")
    _require_checkpoint_matches_current_data(args.baseline_ckpt, args, "baseline")
    baseline, baseline_checkpoint = load_baseline_model(args.baseline_ckpt, device)
    if bool(getattr(args, "multi_dataset", False)):
        text1_cache = _build_text1_caches(args, device)
        text1_dim = int(_first_cache(text1_cache)["state_prompt_sentence"].shape[1])
    else:
        text1_cache = build_text1_cache(
            args.text1_path,
            args.text_model,
            str(device),
            args.cache_dir,
            batch_size=args.text_batch_size,
            max_tokens=args.text_max_tokens,
        )
        text1_dim = int(text1_cache["state_prompt_sentence"].shape[1])
    loaders = make_solar_loaders(args, text1_cache=text1_cache, text2_cache=None)
    model = build_fusion_model(
        args,
        baseline,
        text1_dim=text1_dim,
        text2_dim=1,
        use_text2_correction=False,
    ).to(device)
    return train_and_select(
        args,
        model,
        loaders,
        device,
        amp,
        args.text1_out_dir,
        "InternalRealtimeTextResidualIntervention",
        checkpoint_extra={
            "model_role": "internal_realtime_text_residual_intervention",
            "method_name": "FRTC-Net-Text1",
            "paper_stage": "realtime_textual_residual_intervention",
            "paper_visibility": "internal_pretraining;supports_final_frtc_ber",
            "stage1_module": "regime_conditioned_deformable_asynchronous_realtime_text_residual_injection",
            "baseline_checkpoint": args.baseline_ckpt,
            "baseline_code_version": baseline_checkpoint.get("code_version", "unknown"),
            "text1_dim": text1_dim,
            "text2_dim": 1,
            "use_text2_correction": False,
            "method_card": METHOD_CARD,
        },
        require_zero_initial_correction=True,
        allow_epoch0_fallback=True,
    )


def load_fusion_model(
    checkpoint_path: str,
    baseline,
    device: torch.device,
    text1_dim_override: Optional[int] = None,
    text2_dim_override: Optional[int] = None,
    args_override=None,
    load_text2_modules: bool = True,
):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved = AttrDict(**checkpoint.get("args", {}))
    if args_override is not None:
        for key, value in vars(args_override).items():
            setattr(saved, key, value)
    defaults = {
        "seq_len": 96,
        "pred_len": 45,
        "d_model": 128,
        "n_heads": 4,
        "d_ff": 512,
        "text_layers": 3,
        "nwp_layers": 2,
        "experts": 6,
        "basis_rank": 16,
        "dropout": 0.1,
        "cutoff_index": 8.0,
        "max_delta": 0.20,
        "text2_max_delta": 0.20,
        "risk_horizon_hours": 3.0,
        "realtime_slots_per_field": 4,
        "freeze_baseline": True,
        "use_realtime_condition": True,
        "use_fuzzy_extreme": True,
        "use_shock_evidence": True,
        "use_low_text2": True,
        "use_high_text2": True,
        "w_stage_scale": 0.01,
        "w_short_residual": 0.08,
        "risk_horizon_hours": 1.5,
    }
    for key, value in defaults.items():
        if not hasattr(saved, key):
            setattr(saved, key, value)
    text1_dim = int(
        text1_dim_override
        if text1_dim_override is not None
        else checkpoint.get("metadata", {}).get("text1_dim", checkpoint.get("text1_dim", 1))
    )
    text2_dim = int(
        text2_dim_override
        if text2_dim_override is not None
        else checkpoint.get("metadata", {}).get("text2_dim", checkpoint.get("text2_dim", 1))
    )
    model = build_fusion_model(
        saved,
        baseline,
        text1_dim=text1_dim,
        text2_dim=text2_dim,
        use_text2_correction=True,
    ).to(device)
    current = model.state_dict()
    compatible = {}
    skipped = []
    text2_prefixes = (
        "text_encoder.",
        "low_resampler.",
        "high_resampler.",
        "low_innovation.",
        "high_innovation.",
        "residual_decoder.",
        "candidate_calibrator.",
        "text_router.",
        "release_gate.",
        "textual_intervention.",
        "risk_controller.",
        "extreme_shock.",
        "fuzzy_extreme.",
        "text2_stage_scale",
        "router_stage_scale",
        "text_ramp_stage_scale",
    )
    for name, value in checkpoint["model"].items():
        if not load_text2_modules and any(name.startswith(prefix) for prefix in text2_prefixes):
            skipped.append(name)
            continue
        if name in current and current[name].shape == value.shape:
            compatible[name] = value
        else:
            skipped.append(name)
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected keys when loading fusion model: {unexpected}")
    if skipped or missing:
        benign_missing_prefixes = (
            "candidate_calibrator.disagreement_",
            "candidate_calibrator.direction_",
        )
        missing = [
            name for name in missing
            if not any(name.startswith(prefix) for prefix in benign_missing_prefixes)
        ]
        print(
            f"loaded compatible fusion weights: {len(compatible)}, "
            f"skipped={len(skipped)}, newly_initialized={len(missing)}"
        )
    return model, checkpoint


def freeze_for_text2_stage(model: SolarUnifiedTextFusionModel) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    trainable_modules = [
        model.text_encoder,
        model.low_resampler,
        model.high_resampler,
        model.low_innovation,
        model.high_innovation,
        model.residual_decoder,
        model.risk_controller,
        model.extreme_shock,
        model.fuzzy_extreme,
        model.release_gate,
        model.textual_intervention,
        model.condition_norm,
    ]
    if model.use_text_router:
        trainable_modules.extend([model.candidate_calibrator, model.text_router])
    for module in trainable_modules:
        for parameter in module.parameters():
            parameter.requires_grad = True
    model.text2_stage_scale.requires_grad = True
    model.text_ramp_stage_scale.requires_grad = True
    model.router_stage_scale.requires_grad = bool(model.use_text_router)


def freeze_for_text2_high_stage(model: SolarUnifiedTextFusionModel) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    trainable_modules = [
        model.high_resampler,
        model.high_innovation,
        model.residual_decoder,
        model.risk_controller,
        model.extreme_shock,
        model.fuzzy_extreme,
        model.release_gate,
        model.textual_intervention,
        model.condition_norm,
    ]
    if model.use_text_router:
        trainable_modules.extend([model.candidate_calibrator, model.text_router])
    for module in trainable_modules:
        for parameter in module.parameters():
            parameter.requires_grad = True
    model.text2_stage_scale.requires_grad = True
    model.text_ramp_stage_scale.requires_grad = True
    model.router_stage_scale.requires_grad = bool(model.use_text_router)


def _smooth_1d_horizon(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size < 3:
        return np.clip(values, 0.0, 1.0)
    padded = np.pad(values, (1, 1), mode="edge")
    kernel = np.asarray([0.25, 0.5, 0.25], dtype=np.float32)
    smoothed = np.convolve(padded, kernel, mode="valid")
    return np.clip(smoothed.astype(np.float32, copy=False), 0.0, 1.0)


@torch.inference_mode()
def collect_text2_calibration_arrays(model, loader, device, amp, max_batches: int = 0):
    model.eval()
    stage1_texts, interventions, targets = [], [], []
    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= int(max_batches):
            break
        batch = move_batch(batch, device)
        with _autocast_context(device, amp):
            prediction, parts = _forward_with_parts(model, batch)
        stage1_text = parts.get("stage1_text_prediction", prediction)
        intervention = parts.get("text2_intervention", parts.get("correction", torch.zeros_like(prediction)))
        stage1_texts.append(stage1_text.float().cpu().numpy())
        interventions.append(intervention.float().cpu().numpy())
        targets.append(batch["y"].float().cpu().numpy())
    if not stage1_texts:
        return None
    return (
        np.concatenate(stage1_texts),
        np.concatenate(interventions),
        np.concatenate(targets),
    )


@torch.inference_mode()
def collect_residual_calibration_arrays(model, loader, device, amp, max_batches: int = 0):
    model.eval()
    stage1_texts, targets = [], []
    features: Dict[str, List[np.ndarray]] = {key: [] for key in RESIDUAL_CALIBRATION_FEATURES}
    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= int(max_batches):
            break
        batch = move_batch(batch, device)
        with _autocast_context(device, amp):
            prediction, parts = _forward_with_parts(model, batch)
        stage1_text = parts.get("stage1_text_prediction", prediction)
        stage1_texts.append(stage1_text.float().cpu().numpy())
        targets.append(batch["y"].float().cpu().numpy())
        for key in RESIDUAL_CALIBRATION_FEATURES:
            value = parts.get(key)
            if value is None or value.shape != prediction.shape:
                value = torch.zeros_like(prediction)
            features[key].append(value.float().cpu().numpy())
    if not stage1_texts:
        return None
    return {
        "stage1_text": np.concatenate(stage1_texts, axis=0),
        "target": np.concatenate(targets, axis=0),
        "features": {
            key: np.concatenate(values, axis=0)
            for key, values in features.items()
            if values
        },
    }


def fit_text2_residual_alpha(
    stage1_text_scaled: np.ndarray,
    correction_scaled: np.ndarray,
    target_scaled: np.ndarray,
) -> np.ndarray:
    residual_scaled = target_scaled - stage1_text_scaled
    numerator = np.sum(correction_scaled * residual_scaled, axis=0)
    denominator = np.sum(correction_scaled ** 2, axis=0)
    alpha = numerator / np.maximum(denominator, 1e-8)
    alpha = np.clip(alpha, 0.0, 1.0)
    return _smooth_1d_horizon(alpha)


def _fit_horizon_ridge_weights(
    feature_stack: np.ndarray,
    residual: np.ndarray,
    ridge_lambda: float,
    weight_clip: float,
    sample_weight: Optional[np.ndarray] = None,
) -> np.ndarray:
    feature_stack = np.asarray(feature_stack, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64)
    if feature_stack.ndim != 3:
        raise ValueError(f"feature_stack must be [N,H,F], got {feature_stack.shape}")
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
        if sample_weight.shape != residual.shape:
            raise ValueError(
                f"sample_weight shape mismatch: got {sample_weight.shape}, expected {residual.shape}"
            )
    horizon = feature_stack.shape[1]
    feature_count = feature_stack.shape[2]
    weights = np.zeros((horizon, feature_count), dtype=np.float32)
    eye = np.eye(feature_count, dtype=np.float64)
    lam = max(float(ridge_lambda), 0.0)
    for index in range(horizon):
        x_h = feature_stack[:, index, :]
        y_h = residual[:, index]
        if sample_weight is not None:
            w_h = np.sqrt(np.clip(sample_weight[:, index], 1e-6, None)).reshape(-1, 1)
            x_h = x_h * w_h
            y_h = y_h * w_h[:, 0]
        system = x_h.T @ x_h + lam * eye
        rhs = x_h.T @ y_h
        try:
            weight = np.linalg.solve(system, rhs)
        except np.linalg.LinAlgError:
            weight = np.linalg.lstsq(system, rhs, rcond=None)[0]
        weights[index] = np.clip(weight, -weight_clip, weight_clip).astype(np.float32, copy=False)
    return weights


def _scaled_mse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((np.asarray(prediction) - np.asarray(target)) ** 2))


def _weighted_scaled_mse(prediction: np.ndarray, target: np.ndarray, weight: np.ndarray) -> float:
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    weight = np.asarray(weight)
    return float(np.sum(weight * (prediction - target) ** 2) / np.maximum(np.sum(weight), 1e-8))


def _scaled_mae(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(prediction) - np.asarray(target))))


def _weighted_scaled_mae(prediction: np.ndarray, target: np.ndarray, weight: np.ndarray) -> float:
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    weight = np.asarray(weight)
    return float(np.sum(weight * np.abs(prediction - target)) / np.maximum(np.sum(weight), 1e-8))


def _observable_release_budget(
    feature_map: Dict[str, np.ndarray],
    shape: Tuple[int, int],
    profile: str = "balanced",
) -> np.ndarray:
    """Forecast-time release budget from observable text/NWP evidence only.

    Profiles are selected on the validation split and then fixed for test.  None
    of the features may depend on future target residuals.
    """
    zero = np.zeros(shape, dtype=np.float32)

    def feature(name: str) -> np.ndarray:
        value = feature_map.get(name)
        if value is None or np.asarray(value).shape != shape:
            return zero
        return np.asarray(value, dtype=np.float32)

    extreme = np.clip(feature("observable_extreme_gate"), 0.0, 1.0)
    stable = np.clip(feature("observable_stable_gate"), 0.0, 1.0)
    competitive_event = np.clip(feature("competitive_event_gate"), 0.0, 1.0)
    competitive_stable = np.clip(feature("competitive_stable_gate"), 0.0, 1.0)
    event_margin = np.clip(feature("event_stability_margin_gate"), 0.0, 1.0)
    dataset = np.clip(feature("dataset_evidence_release_gate"), 0.0, 1.0)
    semantic = np.abs(np.clip(feature("semantic_signed_risk_gate"), -1.0, 1.0))
    ramp = np.abs(np.clip(feature("ramp_signed_risk_gate"), -1.0, 1.0))
    shock = np.abs(np.clip(feature("shock_signed_risk_gate"), -1.0, 1.0))
    scalar = np.abs(np.clip(feature("text_scalar_signed_risk_gate"), -1.0, 1.0))
    counterfactual = np.tanh(np.abs(feature("extreme_counterfactual_delta")) / 0.035)
    proposal = np.tanh(np.abs(feature("extreme_proposal_delta")) / 0.045)
    nwp_disagreement = np.tanh(np.abs(feature("extreme_nwp_disagreement_delta")) / 0.020)
    event_basis = np.tanh(
        (
            np.abs(feature("event_counterfactual_basis_delta"))
            + np.abs(feature("event_proposal_basis_delta"))
            + np.abs(feature("event_ramp_basis_delta"))
            + np.abs(feature("event_realtime_basis_delta"))
            + np.abs(feature("event_nwp_disagreement_basis_delta"))
            + np.abs(feature("competitive_event_treatment_delta"))
        )
        / 0.055
    )
    normal_trend = np.tanh(
        (
            np.abs(feature("normal_trend_basis_delta"))
            + np.abs(feature("competitive_trend_treatment_delta"))
        )
        / 0.060
    )
    risk = (
        0.23 * extreme
        + 0.16 * competitive_event
        + 0.13 * event_margin
        + 0.13 * dataset
        + 0.13 * semantic
        + 0.13 * ramp
        + 0.10 * shock
        + 0.06 * event_basis
        + 0.04 * scalar
        + 0.03 * counterfactual
        + 0.02 * proposal
        + 0.02 * nwp_disagreement
    )
    risk = np.clip(risk, 0.0, 1.0)

    profile = str(profile or "balanced")
    if profile == "none":
        return np.ones(shape, dtype=np.float32)
    if profile == "normal_safe":
        event_pressure = np.clip(
            0.30 * extreme
            + 0.20 * competitive_event
            + 0.18 * event_margin
            + 0.14 * semantic
            + 0.12 * ramp
            + 0.08 * shock
            + 0.05 * event_basis
            + 0.04 * nwp_disagreement,
            0.0,
            1.0,
        )
        stable_trend = np.clip(
            0.36 * stable
            + 0.22 * competitive_stable
            + 0.25 * normal_trend
            + 0.11 * (1.0 - event_pressure)
            + 0.08 * np.tanh(np.abs(feature("stable_proposal_delta")) / 0.040)
            + 0.06 * np.tanh(np.abs(feature("stable_dataset_delta")) / 0.035),
            0.0,
            1.0,
        )
        budget = np.clip((stable_trend - 0.12) / 0.46, 0.0, 1.0)
        budget = budget * np.clip(1.0 - 0.72 * event_pressure, 0.05, 1.0)
        budget = np.maximum(
            budget,
            0.18 * normal_trend * stable * np.clip(1.0 - 0.82 * event_pressure, 0.0, 1.0),
        )
        budget = np.where((stable_trend > 0.20) | (normal_trend > 0.18), budget, 0.0)
        budget = np.where(budget >= 0.05, budget, 0.0)
        return np.clip(budget.astype(np.float32, copy=False), 0.0, 0.70)
    if profile == "soft":
        normal_suppression = np.clip(1.0 - 0.70 * stable, 0.18, 1.0)
        budget = np.clip((risk * normal_suppression - 0.06) / 0.26, 0.0, 1.0)
        budget = np.maximum(budget, 0.20 * normal_trend * (1.0 - 0.45 * extreme))
        budget = np.maximum(budget, 0.18 * extreme * (1.0 - 0.55 * stable))
        return np.clip(budget.astype(np.float32, copy=False), 0.0, 1.0)
    if profile == "extreme":
        normal_suppression = np.clip(1.0 - 0.82 * stable, 0.05, 1.0)
        event_risk = np.clip(
            0.28 * extreme
            + 0.20 * competitive_event
            + 0.16 * event_margin
            + 0.13 * semantic
            + 0.13 * ramp
            + 0.09 * shock
            + 0.07 * event_basis
            + 0.06 * nwp_disagreement,
            0.0,
            1.0,
        )
        budget = np.clip((event_risk * normal_suppression - 0.10) / 0.25, 0.0, 1.0)
        budget = np.maximum(
            budget,
            0.18 * np.maximum(extreme, competitive_event) * (1.0 - 0.65 * stable),
        )
        budget = np.where(
            (event_risk > 0.10)
            | (extreme > 0.28)
            | (competitive_event > 0.18)
            | (event_margin > 0.55)
            | (semantic > 0.14),
            budget,
            0.0,
        )
        budget = np.where(budget >= 0.08, budget, 0.0)
        return np.clip(budget.astype(np.float32, copy=False), 0.0, 1.0)
    if profile == "event_sparse":
        event_pressure = np.clip(
            0.26 * extreme
            + 0.22 * competitive_event
            + 0.16 * event_margin
            + 0.15 * ramp
            + 0.12 * semantic
            + 0.08 * shock
            + 0.07 * event_basis
            + 0.06 * nwp_disagreement,
            0.0,
            1.0,
        )
        normal_suppression = np.clip(
            1.0 - 0.88 * stable - 0.24 * competitive_stable - 0.18 * normal_trend,
            0.02,
            1.0,
        )
        budget = np.clip((event_pressure * normal_suppression - 0.12) / 0.21, 0.0, 1.0)
        budget = np.maximum(
            budget,
            0.22 * np.maximum(extreme, competitive_event) * np.clip(1.0 - 0.76 * stable, 0.0, 1.0),
        )
        budget = np.maximum(budget, 0.16 * ramp * np.clip(1.0 - 0.70 * stable, 0.0, 1.0))
        budget = np.maximum(budget, 0.12 * semantic * np.clip(1.0 - 0.70 * stable, 0.0, 1.0))
        trigger = (
            (event_pressure > 0.12)
            | (extreme > 0.30)
            | (competitive_event > 0.18)
            | (event_margin > 0.56)
            | (ramp > 0.20)
            | (semantic > 0.18)
            | (shock > 0.20)
            | (nwp_disagreement > 0.28)
        )
        budget = np.where(trigger, budget, 0.0)
        budget = np.where(budget >= 0.07, budget, 0.0)
        return np.clip(budget.astype(np.float32, copy=False), 0.0, 0.88)
    if profile == "sparse":
        event_pressure = np.clip(
            0.27 * extreme
            + 0.23 * competitive_event
            + 0.16 * event_margin
            + 0.15 * ramp
            + 0.11 * semantic
            + 0.08 * shock
            + 0.06 * event_basis
            + 0.05 * nwp_disagreement,
            0.0,
            1.0,
        )
        normal_suppression = np.clip(
            1.0 - 0.90 * stable - 0.25 * competitive_stable - 0.14 * normal_trend,
            0.02,
            1.0,
        )
        budget = np.clip((event_pressure * normal_suppression - 0.16) / 0.22, 0.0, 1.0)
        budget = np.maximum(
            budget,
            0.12 * np.maximum(extreme, competitive_event) * np.clip(1.0 - 0.76 * stable, 0.0, 1.0),
        )
        trigger = (
            (event_pressure > 0.16)
            | (extreme > 0.38)
            | (competitive_event > 0.22)
            | (event_margin > 0.60)
            | (ramp > 0.22)
            | (semantic > 0.22)
            | (shock > 0.24)
            | (nwp_disagreement > 0.32)
        )
        budget = np.where(trigger, budget, 0.0)
        budget = np.where(budget >= 0.08, budget, 0.0)
        return np.clip(budget.astype(np.float32, copy=False), 0.0, 0.88)

    normal_suppression = np.clip(
        1.0 - 0.80 * stable - 0.20 * competitive_stable,
        0.08,
        1.0,
    )
    budget = np.clip((risk * normal_suppression - 0.11) / 0.28, 0.0, 1.0)
    budget = np.maximum(
        budget,
        0.16 * np.maximum(extreme, competitive_event) * (1.0 - 0.65 * stable),
    )
    budget = np.maximum(budget, 0.10 * normal_trend * (1.0 - 0.55 * extreme))
    budget = np.where(
        (risk > 0.10)
        | (extreme > 0.30)
        | (competitive_event > 0.16)
        | (event_margin > 0.55)
        | (semantic > 0.14)
        | (normal_trend > 0.25),
        budget,
        0.0,
    )
    budget = np.where(budget >= 0.08, budget, 0.0)
    return np.clip(budget.astype(np.float32, copy=False), 0.0, 1.0)


def _observable_regime_mix(
    feature_map: Dict[str, np.ndarray],
    shape: Tuple[int, int],
    *,
    sharpness: float = 1.25,
    floor: float = 0.03,
    bias: float = 0.0,
) -> np.ndarray:
    """Observable event-vs-normal mixture used at inference.

    The mixture is fitted on validation through fixed hyper-parameters, but the
    test-time value itself depends only on forecast-time NWP/text evidence.
    """
    zero = np.zeros(shape, dtype=np.float32)

    def feature(name: str) -> np.ndarray:
        value = feature_map.get(name)
        if value is None or np.asarray(value).shape != shape:
            return zero
        return np.asarray(value, dtype=np.float32)

    extreme = np.clip(feature("observable_extreme_gate"), 0.0, 1.0)
    stable = np.clip(feature("observable_stable_gate"), 0.0, 1.0)
    competitive_event = np.clip(feature("competitive_event_gate"), 0.0, 1.0)
    competitive_stable = np.clip(feature("competitive_stable_gate"), 0.0, 1.0)
    event_margin = np.clip(feature("event_stability_margin_gate"), 0.0, 1.0)
    dataset = np.clip(feature("dataset_evidence_release_gate"), 0.0, 1.0)
    semantic = np.abs(np.clip(feature("semantic_signed_risk_gate"), -1.0, 1.0))
    ramp = np.abs(np.clip(feature("ramp_signed_risk_gate"), -1.0, 1.0))
    shock = np.abs(np.clip(feature("shock_signed_risk_gate"), -1.0, 1.0))
    scalar = np.abs(np.clip(feature("text_scalar_signed_risk_gate"), -1.0, 1.0))
    nwp_disagreement = np.tanh(np.abs(feature("extreme_nwp_disagreement_delta")) / 0.020)
    counterfactual = np.tanh(np.abs(feature("extreme_counterfactual_delta")) / 0.035)
    event_basis = np.tanh(
        (
            np.abs(feature("event_counterfactual_basis_delta"))
            + np.abs(feature("event_proposal_basis_delta"))
            + np.abs(feature("event_ramp_basis_delta"))
            + np.abs(feature("event_nwp_disagreement_basis_delta"))
            + np.abs(feature("competitive_event_treatment_delta"))
        )
        / 0.055
    )
    normal_trend = np.tanh(
        (
            np.abs(feature("normal_trend_basis_delta"))
            + np.abs(feature("competitive_trend_treatment_delta"))
        )
        / 0.060
    )
    stable_basis = np.tanh(
        (np.abs(feature("stable_proposal_delta")) + np.abs(feature("stable_dataset_delta")))
        / 0.055
    )
    event_evidence = np.clip(
        0.24 * extreme
        + 0.18 * competitive_event
        + 0.13 * event_margin
        + 0.13 * semantic
        + 0.12 * ramp
        + 0.09 * shock
        + 0.08 * dataset
        + 0.07 * nwp_disagreement
        + 0.05 * counterfactual
        + 0.03 * event_basis,
        0.0,
        1.0,
    )
    normal_evidence = np.clip(
        0.34 * stable
        + 0.20 * competitive_stable
        + 0.23 * normal_trend
        + 0.15 * stable_basis
        + 0.10 * (1.0 - event_evidence)
        + 0.08 * (1.0 - scalar),
        0.0,
        1.0,
    )
    logits = float(sharpness) * (event_evidence - normal_evidence + float(bias))
    event_mix = 1.0 / (1.0 + np.exp(-np.clip(logits, -12.0, 12.0)))
    floor = float(np.clip(floor, 0.0, 0.30))
    event_mix = floor + (1.0 - 2.0 * floor) * event_mix
    return np.clip(event_mix.astype(np.float32, copy=False), 0.0, 1.0)


def _event_calibration_weight(stage1: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Validation-only event weights.

    This function is used only when fitting residual release weights on the
    validation split.  It may look at target residuals to decide which validation
    horizons are high-ramp/high-error cases.  At inference/test time, the fitted
    weights are fixed and no future target or true residual is accessed.
    """
    stage1 = np.asarray(stage1, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    residual_abs = np.abs(target - stage1)
    residual_scale = np.mean(residual_abs, axis=1, keepdims=True).clip(min=1e-3)
    residual_focus = 1.0 / (1.0 + np.exp(-(residual_abs - 0.85 * residual_scale) / 0.05))
    ramp = np.zeros_like(target, dtype=np.float32)
    if target.shape[1] > 1:
        ramp[:, 1:] = target[:, 1:] - target[:, :-1]
    ramp_abs = np.abs(ramp)
    ramp_scale = np.mean(ramp_abs, axis=1, keepdims=True).clip(min=1e-3)
    ramp_focus = 1.0 / (1.0 + np.exp(-(ramp_abs - 0.70 * ramp_scale) / 0.04))
    event_weight = 1.0 + 2.6 * ramp_focus + 1.4 * residual_focus
    return np.clip(event_weight.astype(np.float32, copy=False), 1.0, 5.0)


def _calibration_event_masks(stage1: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validation-only masks for reporting and selecting fixed release weights."""
    stage1 = np.asarray(stage1, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    residual_err = (target - stage1) ** 2
    high_threshold = np.nanpercentile(residual_err, 80.0)
    high_residual = residual_err >= high_threshold
    ramp = np.zeros_like(target, dtype=np.float32)
    if target.shape[1] > 1:
        ramp[:, 1:] = target[:, 1:] - target[:, :-1]
    ramp_abs = np.abs(ramp)
    ramp_threshold = np.nanpercentile(ramp_abs, 85.0)
    ramp_event = ramp_abs >= max(float(ramp_threshold), 1e-6)
    extreme_union = high_residual | ramp_event
    return high_residual, ramp_event, extreme_union


def _masked_scaled_mse(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return _scaled_mse(prediction, target)
    return float(np.mean((np.asarray(prediction)[mask] - np.asarray(target)[mask]) ** 2))


def _masked_scaled_mae(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return _scaled_mae(prediction, target)
    return float(np.mean(np.abs(np.asarray(prediction)[mask] - np.asarray(target)[mask])))


def _residual_delta_from_weights(
    stack: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float32)
    return np.sum(stack * weights.reshape(1, weights.shape[0], weights.shape[1]), axis=-1)


def _candidate_calibration_delta(
    candidate: dict,
    feature_map: Dict[str, np.ndarray],
    shape: Tuple[int, int],
    *,
    dtype=np.float32,
    apply_release_scale: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Apply a fitted residual-release candidate using observable features only."""
    variant = str(candidate.get("variant", "single_event_release"))
    if variant == "no_text2_release":
        zeros = np.zeros(shape, dtype=dtype)
        return zeros, zeros.copy(), np.ones(shape, dtype=dtype), zeros.copy()

    feature_names = tuple(candidate.get("features", ()))
    if not feature_names:
        raise ValueError("residual calibration candidate has no features")
    missing = [name for name in feature_names if name not in feature_map]
    if missing:
        raise ValueError(f"residual calibration features are missing: {missing}")
    stack = np.stack(
        [np.asarray(feature_map[name], dtype=dtype) for name in feature_names],
        axis=-1,
    )
    weights = np.asarray(candidate.get("weights"), dtype=dtype)
    expected_shape = (shape[1], len(feature_names))
    if weights.ndim != 2 or weights.shape != expected_shape:
        raise ValueError(
            "residual calibration weight shape mismatch: "
            f"got {weights.shape}, expected {expected_shape}"
        )

    cap = _safe_float(candidate.get("cap", 0.0), 0.0)
    release_budget = None
    normal_release_budget = None
    event_mix = None
    if variant == "semantic_route_event_switch":
        event_weights = np.asarray(candidate.get("event_weights", weights), dtype=dtype)
        if event_weights.shape != expected_shape:
            raise ValueError(
                "semantic route/event switch weight shape mismatch: "
                f"event={event_weights.shape}, expected={expected_shape}"
            )
        route_delta = _residual_delta_from_weights(stack, weights)
        event_delta = _residual_delta_from_weights(stack, event_weights)
        release_budget = _observable_release_budget(
            feature_map,
            shape,
            profile=str(candidate.get("budget_profile", "extreme")),
        ).astype(dtype, copy=False)
        event_mix_raw = _observable_regime_mix(
            feature_map,
            shape,
            sharpness=_safe_float(candidate.get("regime_mix_sharpness", 1.25), 1.25),
            floor=_safe_float(candidate.get("regime_mix_floor", 0.03), 0.03),
            bias=_safe_float(candidate.get("regime_mix_bias", 0.0), 0.0),
        ).astype(dtype, copy=False)
        switch_transform = str(candidate.get("switch_transform", "budget_mix"))
        if switch_transform == "budget_only":
            event_mix = release_budget
        elif switch_transform == "sharp_budget":
            event_mix = np.clip((release_budget - 0.06) / 0.34, 0.0, 1.0)
            event_mix = np.maximum(event_mix, release_budget * event_mix_raw)
        else:
            event_mix = np.clip(release_budget * (0.35 + 0.65 * event_mix_raw), 0.0, 1.0)
        normal_release_budget = 1.0 - event_mix
        delta = normal_release_budget * route_delta + event_mix * event_delta
    elif variant == "pareto_regime":
        event_weights = np.asarray(candidate.get("event_weights", weights), dtype=dtype)
        normal_weights = np.asarray(candidate.get("normal_weights", weights), dtype=dtype)
        if event_weights.shape != expected_shape or normal_weights.shape != expected_shape:
            raise ValueError(
                "pareto residual calibration weight shape mismatch: "
                f"event={event_weights.shape}, normal={normal_weights.shape}, expected={expected_shape}"
            )
        event_delta = _residual_delta_from_weights(stack, event_weights)
        normal_delta = _residual_delta_from_weights(stack, normal_weights)
        release_budget = _observable_release_budget(
            feature_map,
            shape,
            profile=str(candidate.get("budget_profile", "balanced")),
        ).astype(dtype, copy=False)
        normal_release_budget = _observable_release_budget(
            feature_map,
            shape,
            profile=str(candidate.get("normal_budget_profile", "normal_safe")),
        ).astype(dtype, copy=False)
        event_mix = _observable_regime_mix(
            feature_map,
            shape,
            sharpness=_safe_float(candidate.get("regime_mix_sharpness", 1.25), 1.25),
            floor=_safe_float(candidate.get("regime_mix_floor", 0.03), 0.03),
            bias=_safe_float(candidate.get("regime_mix_bias", 0.0), 0.0),
        ).astype(dtype, copy=False)
        delta = event_mix * event_delta * release_budget + (1.0 - event_mix) * normal_delta * normal_release_budget
    else:
        delta = _residual_delta_from_weights(stack, weights)
        if bool(candidate.get("use_observable_release_budget", False)):
            release_budget = _observable_release_budget(
                feature_map,
                shape,
                profile=str(candidate.get("budget_profile", "balanced")),
            ).astype(dtype, copy=False)
            delta = delta * release_budget

    if cap > 0:
        delta = np.clip(delta, -cap, cap)
    if apply_release_scale:
        release_scale = _safe_float(
            candidate.get("style_release_scale", candidate.get("release_scale", 1.0)),
            1.0,
        )
        delta = delta * float(np.clip(release_scale, 0.0, 1.0))
    return delta.astype(dtype, copy=False), release_budget, normal_release_budget, event_mix


def _fold_slices(length: int, folds: int = 4) -> List[slice]:
    length = int(length)
    folds = int(max(folds, 1))
    if length <= 0:
        return []
    boundaries = np.linspace(0, length, num=min(folds, length) + 1, dtype=int)
    slices: List[slice] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end > start:
            slices.append(slice(int(start), int(end)))
    return slices


def _release_scale_candidate_metrics(
    stage1: np.ndarray,
    target: np.ndarray,
    delta: np.ndarray,
    *,
    event_weight: np.ndarray,
    high_mask: np.ndarray,
    ramp_mask: np.ndarray,
    extreme_mask: np.ndarray,
    baseline_mse: float,
    baseline_mae: float,
    baseline_weighted_mse: float,
    baseline_weighted_mae: float,
    baseline_high_mse: float,
    baseline_ramp_mse: float,
    baseline_extreme_mse: float,
    baseline_normal_mse: float,
    scale: float,
) -> Dict[str, object]:
    scale = float(np.clip(scale, 0.0, 1.0))
    scaled_delta = np.asarray(delta, dtype=np.float32) * scale
    calibrated = np.asarray(stage1, dtype=np.float32) + scaled_delta
    mse = _scaled_mse(calibrated, target)
    mae = _scaled_mae(calibrated, target)
    weighted_mse = _weighted_scaled_mse(calibrated, target, event_weight)
    weighted_mae = _weighted_scaled_mae(calibrated, target, event_weight)
    high_mse = _masked_scaled_mse(calibrated, target, high_mask)
    ramp_mse = _masked_scaled_mse(calibrated, target, ramp_mask)
    extreme_mse = _masked_scaled_mse(calibrated, target, extreme_mask)
    normal_mse = _masked_scaled_mse(calibrated, target, ~extreme_mask)
    high_gain = baseline_high_mse - high_mse
    ramp_gain = baseline_ramp_mse - ramp_mse
    extreme_gain = baseline_extreme_mse - extreme_mse
    normal_gain = baseline_normal_mse - normal_mse
    global_gain = baseline_mse - mse
    weighted_gain = baseline_weighted_mse - weighted_mse
    fold_global_gains: List[float] = []
    fold_event_gains: List[float] = []
    fold_normal_degrades: List[float] = []
    for fold in _fold_slices(stage1.shape[0], folds=4):
        fold_stage1 = stage1[fold]
        fold_target = target[fold]
        fold_calibrated = calibrated[fold]
        fold_extreme = extreme_mask[fold]
        fold_baseline = _scaled_mse(fold_stage1, fold_target)
        fold_global_gains.append(fold_baseline - _scaled_mse(fold_calibrated, fold_target))
        if np.any(fold_extreme):
            fold_event_gains.append(
                _masked_scaled_mse(fold_stage1, fold_target, fold_extreme)
                - _masked_scaled_mse(fold_calibrated, fold_target, fold_extreme)
            )
        else:
            fold_event_gains.append(0.0)
        if np.any(~fold_extreme):
            fold_normal_gain = (
                _masked_scaled_mse(fold_stage1, fold_target, ~fold_extreme)
                - _masked_scaled_mse(fold_calibrated, fold_target, ~fold_extreme)
            )
            fold_normal_degrades.append(max(0.0, -fold_normal_gain))
        else:
            fold_normal_degrades.append(0.0)
    fold_global = np.asarray(fold_global_gains, dtype=np.float32)
    fold_event = np.asarray(fold_event_gains, dtype=np.float32)
    fold_normal_degrade = np.asarray(fold_normal_degrades, dtype=np.float32)
    release = np.abs(scaled_delta) > 1e-5
    release_coverage = float(np.mean(release))
    normal_release_coverage = float(np.mean(release[~extreme_mask])) if np.any(~extreme_mask) else 0.0
    extreme_release_coverage = float(np.mean(release[extreme_mask])) if np.any(extreme_mask) else 0.0
    negative_fold_penalty = float(np.mean(np.maximum(0.0, -fold_global)))
    event_negative_fold_penalty = float(np.mean(np.maximum(0.0, -fold_event)))
    fold_instability = float(np.std(fold_global)) if fold_global.size else 0.0
    normal_degrade = max(0.0, -normal_gain)
    delta_rms = float(np.sqrt(np.mean(scaled_delta ** 2)))
    robust_score = (
        weighted_mse
        + 0.38 * mse
        + 0.80 * max(0.0, -global_gain)
        + 0.55 * max(0.0, -weighted_gain)
        + 6.00 * normal_degrade
        + 0.60 * negative_fold_penalty
        + 0.45 * event_negative_fold_penalty
        + 0.25 * max(0.0, fold_instability - max(abs(global_gain), 1e-4))
        + 0.10 * max(0.0, delta_rms - 0.10)
        + 0.32 * max(0.0, normal_release_coverage - 0.60) * max(0.0, normal_degrade)
        - 0.35 * max(0.0, global_gain)
        - 0.25 * max(0.0, weighted_gain)
        - 0.28 * max(0.0, high_gain)
        - 0.35 * max(0.0, ramp_gain)
        - 0.24 * max(0.0, extreme_gain)
    )
    return {
        "scale": scale,
        "val_scaled_MSE": float(mse),
        "val_scaled_MAE": float(mae),
        "val_scaled_improve_MSE": float(global_gain),
        "val_scaled_improve_MAE": float(baseline_mae - mae),
        "val_weighted_scaled_MSE": float(weighted_mse),
        "val_weighted_scaled_MAE": float(weighted_mae),
        "val_weighted_scaled_improve_MSE": float(weighted_gain),
        "val_weighted_scaled_improve_MAE": float(baseline_weighted_mae - weighted_mae),
        "val_high_residual_scaled_MSE": float(high_mse),
        "val_high_residual_improve_MSE": float(high_gain),
        "val_ramp_event_scaled_MSE": float(ramp_mse),
        "val_ramp_event_improve_MSE": float(ramp_gain),
        "val_extreme_union_scaled_MSE": float(extreme_mse),
        "val_extreme_union_improve_MSE": float(extreme_gain),
        "val_normal_scaled_MSE": float(normal_mse),
        "val_normal_improve_MSE": float(normal_gain),
        "val_normal_degrade_MSE": float(normal_degrade),
        "val_delta_rms_scaled": float(delta_rms),
        "val_release_coverage": release_coverage,
        "val_normal_release_coverage": normal_release_coverage,
        "val_extreme_release_coverage": extreme_release_coverage,
        "val_release_selectivity_gap": float(max(0.0, normal_release_coverage - 0.72 * extreme_release_coverage)),
        "val_fold_global_gain_min": float(fold_global.min()) if fold_global.size else 0.0,
        "val_fold_global_gain_mean": float(fold_global.mean()) if fold_global.size else 0.0,
        "val_fold_global_positive_rate": float(np.mean(fold_global >= -1e-8)) if fold_global.size else 0.0,
        "val_fold_event_gain_min": float(fold_event.min()) if fold_event.size else 0.0,
        "val_fold_event_gain_mean": float(fold_event.mean()) if fold_event.size else 0.0,
        "val_fold_event_positive_rate": float(np.mean(fold_event >= -1e-8)) if fold_event.size else 0.0,
        "val_fold_normal_degrade_max": float(fold_normal_degrade.max()) if fold_normal_degrade.size else 0.0,
        "val_fold_instability": fold_instability,
        "val_style_release_score": float(robust_score),
    }


def _attach_dataset_style_release(
    candidate: dict,
    stage1: np.ndarray,
    target: np.ndarray,
    delta: np.ndarray,
    *,
    event_weight: np.ndarray,
    high_mask: np.ndarray,
    ramp_mask: np.ndarray,
    extreme_mask: np.ndarray,
    baseline_mse: float,
    baseline_mae: float,
    baseline_weighted_mse: float,
    baseline_weighted_mae: float,
    baseline_high_mse: float,
    baseline_ramp_mse: float,
    baseline_extreme_mse: float,
    baseline_normal_mse: float,
) -> dict:
    scale_grid = (0.0, 0.10, 0.18, 0.25, 0.35, 0.50, 0.65, 0.80, 0.85, 0.90, 0.95, 1.0)
    metrics = [
        _release_scale_candidate_metrics(
            stage1,
            target,
            delta,
            event_weight=event_weight,
            high_mask=high_mask,
            ramp_mask=ramp_mask,
            extreme_mask=extreme_mask,
            baseline_mse=baseline_mse,
            baseline_mae=baseline_mae,
            baseline_weighted_mse=baseline_weighted_mse,
            baseline_weighted_mae=baseline_weighted_mae,
            baseline_high_mse=baseline_high_mse,
            baseline_ramp_mse=baseline_ramp_mse,
            baseline_extreme_mse=baseline_extreme_mse,
            baseline_normal_mse=baseline_normal_mse,
            scale=scale,
        )
        for scale in scale_grid
    ]
    best = min(metrics, key=lambda item: item["val_style_release_score"])
    for key in (
        "val_scaled_MSE",
        "val_scaled_MAE",
        "val_scaled_improve_MSE",
        "val_scaled_improve_MAE",
        "val_weighted_scaled_MSE",
        "val_weighted_scaled_MAE",
        "val_weighted_scaled_improve_MSE",
        "val_weighted_scaled_improve_MAE",
        "val_high_residual_scaled_MSE",
        "val_high_residual_improve_MSE",
        "val_ramp_event_scaled_MSE",
        "val_ramp_event_improve_MSE",
        "val_extreme_union_scaled_MSE",
        "val_extreme_union_improve_MSE",
        "val_normal_scaled_MSE",
        "val_normal_improve_MSE",
        "val_normal_degrade_MSE",
        "val_delta_rms_scaled",
        "val_release_coverage",
        "val_normal_release_coverage",
        "val_extreme_release_coverage",
        "val_release_selectivity_gap",
        "val_fold_global_gain_min",
        "val_fold_global_gain_mean",
        "val_fold_global_positive_rate",
        "val_fold_event_gain_min",
        "val_fold_event_gain_mean",
        "val_fold_event_positive_rate",
        "val_fold_normal_degrade_max",
        "val_fold_instability",
        "val_style_release_score",
    ):
        candidate[f"pre_style_{key}"] = candidate.get(key, 0.0)
        candidate[key] = best[key]
    candidate["style_release_scale"] = float(best["scale"])
    candidate["release_scale"] = float(best["scale"])
    candidate["style_release_grid"] = [
        {
            "scale": item["scale"],
            "val_scaled_improve_MSE": item["val_scaled_improve_MSE"],
            "val_extreme_union_improve_MSE": item["val_extreme_union_improve_MSE"],
            "val_normal_degrade_MSE": item["val_normal_degrade_MSE"],
            "val_fold_global_gain_min": item["val_fold_global_gain_min"],
            "val_fold_event_gain_min": item["val_fold_event_gain_min"],
            "val_style_release_score": item["val_style_release_score"],
        }
        for item in metrics
    ]
    candidate["val_event_score"] = float(best["val_style_release_score"])
    return candidate


def _candidate_metric_record(
    *,
    name: str,
    features: Sequence[str],
    ridge_lambda: float,
    cap: float,
    budget_profile: str,
    normal_budget_profile: str,
    variant: str,
    mse: float,
    mae: float,
    weighted_mse: float,
    weighted_mae: float,
    baseline_mse: float,
    baseline_mae: float,
    baseline_weighted_mse: float,
    baseline_weighted_mae: float,
    high_mse: float,
    ramp_mse: float,
    extreme_mse: float,
    normal_mse: float,
    baseline_high_mse: float,
    baseline_ramp_mse: float,
    baseline_extreme_mse: float,
    baseline_normal_mse: float,
    delta_rms: float,
    release_coverage: float,
    normal_release_coverage: float,
    extreme_release_coverage: float,
    budget_mean: float,
    budget_coverage: float,
    event_score: float,
) -> Dict[str, object]:
    high_gain = baseline_high_mse - high_mse
    ramp_gain = baseline_ramp_mse - ramp_mse
    extreme_gain = baseline_extreme_mse - extreme_mse
    normal_gain = baseline_normal_mse - normal_mse
    return {
        "name": name,
        "features": tuple(features),
        "lambda": float(ridge_lambda),
        "cap": float(cap),
        "budget_profile": str(budget_profile),
        "normal_budget_profile": str(normal_budget_profile),
        "variant": str(variant),
        "val_scaled_MSE": float(mse),
        "val_scaled_MAE": float(mae),
        "val_scaled_improve_MSE": float(baseline_mse - mse),
        "val_scaled_improve_MAE": float(baseline_mae - mae),
        "val_weighted_scaled_MSE": float(weighted_mse),
        "val_weighted_scaled_MAE": float(weighted_mae),
        "val_weighted_scaled_improve_MSE": float(baseline_weighted_mse - weighted_mse),
        "val_weighted_scaled_improve_MAE": float(baseline_weighted_mae - weighted_mae),
        "val_high_residual_scaled_MSE": float(high_mse),
        "val_high_residual_improve_MSE": float(high_gain),
        "val_ramp_event_scaled_MSE": float(ramp_mse),
        "val_ramp_event_improve_MSE": float(ramp_gain),
        "val_extreme_union_scaled_MSE": float(extreme_mse),
        "val_extreme_union_improve_MSE": float(extreme_gain),
        "val_normal_scaled_MSE": float(normal_mse),
        "val_normal_improve_MSE": float(normal_gain),
        "val_normal_degrade_MSE": float(max(0.0, -normal_gain)),
        "val_delta_rms_scaled": float(delta_rms),
        "val_release_coverage": float(release_coverage),
        "val_normal_release_coverage": float(normal_release_coverage),
        "val_extreme_release_coverage": float(extreme_release_coverage),
        "val_release_selectivity_gap": float(
            max(0.0, normal_release_coverage - 0.72 * extreme_release_coverage)
        ),
        "val_observable_budget_mean": float(budget_mean),
        "val_observable_budget_coverage": float(budget_coverage),
        "use_observable_release_budget": True,
        "val_event_score": float(event_score),
    }


def _no_text2_release_record(
    *,
    horizon: int,
    baseline_mse: float,
    baseline_mae: float,
    baseline_weighted_mse: float,
    baseline_weighted_mae: float,
    baseline_high_mse: float,
    baseline_ramp_mse: float,
    baseline_extreme_mse: float,
    baseline_normal_mse: float,
) -> Dict[str, object]:
    record = _candidate_metric_record(
        name="no_text2_release",
        features=("__zero_release__",),
        ridge_lambda=0.0,
        cap=0.0,
        budget_profile="none",
        normal_budget_profile="stage1_text",
        variant="no_text2_release",
        mse=baseline_mse,
        mae=baseline_mae,
        weighted_mse=baseline_weighted_mse,
        weighted_mae=baseline_weighted_mae,
        baseline_mse=baseline_mse,
        baseline_mae=baseline_mae,
        baseline_weighted_mse=baseline_weighted_mse,
        baseline_weighted_mae=baseline_weighted_mae,
        high_mse=baseline_high_mse,
        ramp_mse=baseline_ramp_mse,
        extreme_mse=baseline_extreme_mse,
        normal_mse=baseline_normal_mse,
        baseline_high_mse=baseline_high_mse,
        baseline_ramp_mse=baseline_ramp_mse,
        baseline_extreme_mse=baseline_extreme_mse,
        baseline_normal_mse=baseline_normal_mse,
        delta_rms=0.0,
        release_coverage=0.0,
        normal_release_coverage=0.0,
        extreme_release_coverage=0.0,
        budget_mean=0.0,
        budget_coverage=0.0,
        event_score=baseline_weighted_mse + 0.34 * baseline_mse,
    )
    record["weights"] = np.zeros((int(horizon), 1), dtype=np.float32)
    record["fallback_reason"] = "stage1_text_baseline"
    record["use_observable_release_budget"] = False
    record["style_release_scale"] = 0.0
    record["release_scale"] = 0.0
    record["val_fold_global_gain_min"] = 0.0
    record["val_fold_global_gain_mean"] = 0.0
    record["val_fold_global_positive_rate"] = 1.0
    record["val_fold_event_gain_min"] = 0.0
    record["val_fold_event_gain_mean"] = 0.0
    record["val_fold_event_positive_rate"] = 1.0
    record["val_fold_normal_degrade_max"] = 0.0
    record["val_fold_instability"] = 0.0
    record["val_style_release_score"] = record["val_event_score"]
    record["val_event_mix_mean"] = 0.0
    record["val_event_mix_event_mean"] = 0.0
    record["val_event_mix_normal_mean"] = 0.0
    return record


def _accept_text2_calibration(candidate: dict, *, min_global_gain: float, min_extreme_gain: float) -> Tuple[bool, str]:
    global_gain = float(candidate.get("val_scaled_improve_MSE", 0.0))
    weighted_gain = float(candidate.get("val_weighted_scaled_improve_MSE", 0.0))
    high_gain = float(candidate.get("val_high_residual_improve_MSE", 0.0))
    ramp_gain = float(candidate.get("val_ramp_event_improve_MSE", 0.0))
    extreme_gain = float(candidate.get("val_extreme_union_improve_MSE", 0.0))
    normal_degrade = float(candidate.get("val_normal_degrade_MSE", 0.0))
    release_scale = float(candidate.get("style_release_scale", candidate.get("release_scale", 1.0)))
    fold_global_min = float(candidate.get("val_fold_global_gain_min", 0.0))
    fold_event_min = float(candidate.get("val_fold_event_gain_min", 0.0))
    fold_global_positive = float(candidate.get("val_fold_global_positive_rate", 1.0))
    fold_event_positive = float(candidate.get("val_fold_event_positive_rate", 1.0))
    fold_normal_degrade = float(candidate.get("val_fold_normal_degrade_max", 0.0))
    event_gain = max(high_gain, ramp_gain, extreme_gain)
    strong_event_gain = max(ramp_gain, extreme_gain, high_gain)
    if release_scale <= 1e-6:
        return False, "dataset_style_release_scale_zero"
    if global_gain < -1e-8:
        return False, "validation_global_mse_worse_than_text1"
    if weighted_gain < -1e-8:
        return False, "validation_event_weighted_mse_worse_than_text1"
    if (
        global_gain >= max(float(min_global_gain), 5e-4)
        and strong_event_gain >= max(float(min_extreme_gain), 5e-3)
        and normal_degrade <= max(0.0045, 0.12 * max(strong_event_gain, 0.0))
        and fold_global_positive >= 0.50
        and fold_event_positive >= 0.50
        and fold_global_min >= -0.04
        and fold_event_min >= -0.16
    ):
        return True, "strong_event_gain_with_bounded_normal_tradeoff"
    if fold_normal_degrade > max(1e-4, 0.80 * max(event_gain, global_gain, 0.0)):
        return False, "temporal_fold_normal_weather_degrade"
    if (
        global_gain >= float(min_global_gain)
        and normal_degrade <= max(5e-5, 0.35 * max(extreme_gain, 0.0))
        and fold_global_positive >= 0.50
        and fold_global_min >= -max(1e-4, 0.75 * max(global_gain, 0.0))
    ):
        return True, "global_validation_gain"
    if (
        event_gain >= float(min_extreme_gain)
        and normal_degrade <= max(5e-5, 0.45 * max(event_gain, 0.0))
        and fold_event_positive >= 0.50
        and fold_event_min >= -max(2e-4, 0.80 * max(event_gain, 0.0))
    ):
        return True, "event_validation_gain"
    return False, "validation_gain_below_safe_release_threshold"


def _is_better_calibration(candidate: dict, current: Optional[dict]) -> bool:
    if current is None:
        return True
    return candidate["val_event_score"] < current["val_event_score"]


def _bicriteria_release_score(candidate: dict) -> float:
    """Validation-only score for selecting a fixed text2 release policy.

    The score prefers candidates that improve both extreme/ramp windows and
    ordinary daylight windows.  It is deliberately separated from the
    candidate-fitting objective: fitting estimates horizon-wise residual bases,
    while this score chooses the release policy that is most useful for the
    paper mainline without using test residuals.
    """
    global_gain = float(candidate.get("val_scaled_improve_MSE", 0.0))
    weighted_gain = float(candidate.get("val_weighted_scaled_improve_MSE", 0.0))
    high_gain = float(candidate.get("val_high_residual_improve_MSE", 0.0))
    ramp_gain = float(candidate.get("val_ramp_event_improve_MSE", 0.0))
    extreme_gain = float(candidate.get("val_extreme_union_improve_MSE", 0.0))
    normal_gain = float(candidate.get("val_normal_improve_MSE", 0.0))
    normal_degrade = float(candidate.get("val_normal_degrade_MSE", 0.0))
    fold_global_min = float(candidate.get("val_fold_global_gain_min", 0.0))
    fold_event_min = float(candidate.get("val_fold_event_gain_min", 0.0))
    fold_global_positive = float(candidate.get("val_fold_global_positive_rate", 0.0))
    fold_event_positive = float(candidate.get("val_fold_event_positive_rate", 0.0))
    fold_instability = float(candidate.get("val_fold_instability", 0.0))
    release_scale = float(candidate.get("style_release_scale", candidate.get("release_scale", 1.0)))
    variant = str(candidate.get("variant", ""))
    name = str(candidate.get("name", ""))
    event_gain = 0.30 * max(0.0, high_gain) + 0.34 * max(0.0, ramp_gain) + 0.36 * max(0.0, extreme_gain)
    robust_penalty = (
        0.70 * max(0.0, -fold_global_min)
        + 0.45 * max(0.0, -fold_event_min)
        + 0.20 * max(0.0, 0.75 - fold_global_positive)
        + 0.18 * max(0.0, 0.75 - fold_event_positive)
        + 0.10 * max(0.0, fold_instability - max(abs(global_gain), 1e-4))
    )
    structure_bonus = 0.0
    if variant == "pareto_regime":
        structure_bonus += 0.0025
    if name in {"frtc_ber", "causal_treatment_safe", "pareto_regime_basis", "extreme_weather_first"}:
        structure_bonus += 0.0010
    score = (
        1.18 * max(0.0, global_gain)
        + 0.20 * max(0.0, weighted_gain)
        + 0.72 * event_gain
        + 0.30 * max(0.0, normal_gain)
        + structure_bonus
        - 1.55 * normal_degrade
        - robust_penalty
        - 0.012 * max(0.0, release_scale <= 1e-6)
    )
    return float(score)


def _select_bicriteria_release_candidate(
    candidates: Sequence[dict],
    selected: dict,
    *,
    preferred: str,
    min_global_gain: float,
    min_extreme_gain: float,
) -> dict:
    """Choose a validation-stable candidate that balances normal and event regimes."""
    if not candidates:
        return selected
    selected_score = _bicriteria_release_score(selected)
    selected_event_gain = max(
        float(selected.get("val_ramp_event_improve_MSE", 0.0)),
        float(selected.get("val_extreme_union_improve_MSE", 0.0)),
        float(selected.get("val_high_residual_improve_MSE", 0.0)),
    )
    viable: List[dict] = []
    for item in candidates:
        if item.get("variant") == "no_text2_release":
            continue
        if float(item.get("style_release_scale", item.get("release_scale", 1.0))) <= 1e-6:
            continue
        global_gain = float(item.get("val_scaled_improve_MSE", 0.0))
        event_gain = max(
            float(item.get("val_ramp_event_improve_MSE", 0.0)),
            float(item.get("val_extreme_union_improve_MSE", 0.0)),
            float(item.get("val_high_residual_improve_MSE", 0.0)),
        )
        if global_gain < max(float(min_global_gain), 5e-4):
            continue
        if event_gain < max(float(min_extreme_gain), 5e-3):
            continue
        if float(item.get("val_fold_global_positive_rate", 0.0)) < 0.75:
            continue
        if float(item.get("val_fold_event_positive_rate", 0.0)) < 0.75:
            continue
        if float(item.get("val_fold_global_gain_min", 0.0)) < -0.012:
            continue
        if float(item.get("val_fold_event_gain_min", 0.0)) < -0.035:
            continue
        if float(item.get("val_normal_degrade_MSE", 0.0)) > max(0.006, 0.24 * event_gain):
            continue
        viable.append(item)
    if not viable:
        return selected
    viable.sort(
        key=lambda item: (
            -_bicriteria_release_score(item),
            0 if item.get("variant") == "pareto_regime" else 1,
            0 if item.get("name") in {"frtc_ber", preferred} else 1,
            -float(item.get("val_scaled_improve_MSE", 0.0)),
            float(item.get("val_normal_degrade_MSE", 0.0)),
            -max(
                float(item.get("val_ramp_event_improve_MSE", 0.0)),
                float(item.get("val_extreme_union_improve_MSE", 0.0)),
                float(item.get("val_high_residual_improve_MSE", 0.0)),
            ),
        )
    )
    best = viable[0]
    best_score = _bicriteria_release_score(best)
    best_event_gain = max(
        float(best.get("val_ramp_event_improve_MSE", 0.0)),
        float(best.get("val_extreme_union_improve_MSE", 0.0)),
        float(best.get("val_high_residual_improve_MSE", 0.0)),
    )
    selected_global = float(selected.get("val_scaled_improve_MSE", 0.0))
    best_global = float(best.get("val_scaled_improve_MSE", 0.0))
    selected_normal_degrade = float(selected.get("val_normal_degrade_MSE", 0.0))
    best_normal_degrade = float(best.get("val_normal_degrade_MSE", 0.0))
    preserves_event_gain = best_event_gain >= 0.92 * max(selected_event_gain, min_extreme_gain)
    improves_global = best_global >= selected_global + 0.0015
    improves_normal = best_normal_degrade <= max(0.0, selected_normal_degrade - 0.0007)
    if (
        best is not selected
        and preserves_event_gain
        and (
            best_score >= selected_score + 0.0040
            or (improves_global and best_normal_degrade <= selected_normal_degrade + 0.0005)
            or (improves_normal and best_global >= selected_global - 0.0002)
        )
    ):
        best["fallback_reason"] = "bicriteria_extreme_stable_release"
        best["bicriteria_release_score"] = best_score
        best["previous_selected_name"] = selected.get("name")
        best["previous_selected_variant"] = selected.get("variant")
        best["previous_bicriteria_release_score"] = selected_score
        return best
    selected["bicriteria_release_score"] = selected_score
    return selected


def _select_pareto_safe_extreme_candidate(
    candidates: Sequence[dict],
    selected: dict,
    *,
    preferred: str,
    min_global_gain: float,
    min_extreme_gain: float,
) -> dict:
    """Prefer a less harmful normal-weather policy when event gains are retained.

    This is a validation-only policy selector.  It does not weaken the trained
    text residual model; it only chooses which fixed release rule is exported for
    inference.  The chosen rule must keep most of the selected policy's global
    and extreme-regime gains while sharply reducing normal-weather degradation.
    """
    selected_global_gain = float(selected.get("val_scaled_improve_MSE", 0.0))
    selected_extreme_gain = float(selected.get("val_extreme_union_improve_MSE", 0.0))
    selected_event_gain = max(
        float(selected.get("val_ramp_event_improve_MSE", 0.0)),
        selected_extreme_gain,
        float(selected.get("val_high_residual_improve_MSE", 0.0)),
    )
    selected_normal_degrade = float(selected.get("val_normal_degrade_MSE", 0.0))
    if selected_global_gain <= 0.0 or selected_event_gain <= 0.0:
        return selected
    if selected_normal_degrade <= 0.0026:
        selected["fallback_reason"] = selected.get(
            "fallback_reason",
            "pareto_safe_extreme_release_low_normal_degrade",
        )
        return selected

    max_allowed_normal_degrade = min(
        0.0026,
        max(0.00035, 0.50 * max(selected_normal_degrade, 0.0)),
    )
    viable: List[dict] = []
    for item in candidates:
        if item.get("variant") == "no_text2_release":
            continue
        if float(item.get("style_release_scale", item.get("release_scale", 1.0))) <= 1e-6:
            continue
        item_global_gain = float(item.get("val_scaled_improve_MSE", 0.0))
        item_extreme_gain = float(item.get("val_extreme_union_improve_MSE", 0.0))
        item_event_gain = max(
            float(item.get("val_ramp_event_improve_MSE", 0.0)),
            item_extreme_gain,
            float(item.get("val_high_residual_improve_MSE", 0.0)),
        )
        item_normal_degrade = float(item.get("val_normal_degrade_MSE", 0.0))
        if item_normal_degrade > max_allowed_normal_degrade:
            continue
        if item_global_gain < max(min_global_gain, 0.88 * max(selected_global_gain, min_global_gain)):
            continue
        if item_extreme_gain < max(min_extreme_gain, 0.76 * max(selected_extreme_gain, min_extreme_gain)):
            continue
        if item_event_gain < max(min_extreme_gain, 0.78 * max(selected_event_gain, min_extreme_gain)):
            continue
        if float(item.get("val_fold_global_positive_rate", 0.0)) < 0.75:
            continue
        if float(item.get("val_fold_event_positive_rate", 0.0)) < 0.75:
            continue
        viable.append(item)
    if not viable:
        return selected

    viable.sort(
        key=lambda item: (
            -float(item.get("val_scaled_improve_MSE", 0.0)),
            -float(item.get("val_extreme_union_improve_MSE", 0.0)),
            -float(item.get("val_weighted_scaled_improve_MSE", 0.0)),
            float(item.get("val_normal_degrade_MSE", 0.0)),
            0 if item.get("variant") == "pareto_regime" else 1,
            0 if item.get("name") in {"frtc_ber", "causal_treatment_safe", preferred} else 1,
            float(item.get("val_event_score", 1e9)),
        )
    )
    best = viable[0]
    if best is not selected:
        best["fallback_reason"] = "pareto_safe_extreme_release"
        best["previous_selected_name"] = selected.get("name")
        best["previous_selected_variant"] = selected.get("variant")
        best["previous_val_scaled_improve_MSE"] = selected_global_gain
        best["previous_val_extreme_union_improve_MSE"] = selected_extreme_gain
        best["previous_val_normal_degrade_MSE"] = selected_normal_degrade
    return best


def _soft_pareto_candidate_score(candidate: dict) -> float:
    """Validation score for the paper-facing soft Pareto release selector."""
    global_gain = float(candidate.get("val_scaled_improve_MSE", 0.0))
    weighted_gain = float(candidate.get("val_weighted_scaled_improve_MSE", 0.0))
    high_gain = float(candidate.get("val_high_residual_improve_MSE", 0.0))
    ramp_gain = float(candidate.get("val_ramp_event_improve_MSE", 0.0))
    extreme_gain = float(candidate.get("val_extreme_union_improve_MSE", 0.0))
    normal_degrade = float(candidate.get("val_normal_degrade_MSE", 0.0))
    release_scale = float(candidate.get("style_release_scale", candidate.get("release_scale", 1.0)))
    event_gain = 0.36 * max(0.0, high_gain) + 0.30 * max(0.0, ramp_gain) + 0.34 * max(0.0, extreme_gain)
    fold_penalty = 0.0
    fold_penalty += 0.45 * max(0.0, -float(candidate.get("val_fold_global_gain_min", 0.0)))
    fold_penalty += 0.30 * max(0.0, -float(candidate.get("val_fold_event_gain_min", 0.0)))
    return float(
        -1.00 * max(0.0, global_gain)
        -0.34 * max(0.0, weighted_gain)
        -0.55 * event_gain
        +1.00 * normal_degrade
        +0.018 * max(0.0, release_scale <= 1e-6)
        +fold_penalty
        +0.020 * float(candidate.get("val_event_score", 0.0))
    )


def fit_residual_calibration(
    arrays: dict,
    *,
    preferred: str = "pareto_regime_basis",
    ridge_grid: Sequence[float] = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1),
    cap_grid: Sequence[float] = (0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20),
    weight_clip: float = 3.0,
    fast: bool = False,
    search_all_specs: bool = False,
    safe_fallback: bool = True,
    min_global_gain: float = 1e-4,
    min_extreme_gain: float = 5e-4,
    force_candidate_name: str = "",
    force_candidate_variant: str = "",
    force_release_scale: float = -1.0,
) -> Optional[dict]:
    if arrays is None:
        return None
    if fast:
        ridge_grid = (1e-3, 1e-2)
        cap_grid = (0.10, 0.14, 0.18, 0.22)
    stage1 = np.asarray(arrays["stage1_text"], dtype=np.float32)
    target = np.asarray(arrays["target"], dtype=np.float32)
    residual = target - stage1
    feature_map = arrays.get("features", {})
    baseline_mse = _scaled_mse(stage1, target)
    baseline_mae = _scaled_mae(stage1, target)
    event_weight = _event_calibration_weight(stage1, target)
    baseline_weighted_mse = _weighted_scaled_mse(stage1, target, event_weight)
    baseline_weighted_mae = _weighted_scaled_mae(stage1, target, event_weight)
    high_mask, ramp_mask, extreme_mask = _calibration_event_masks(stage1, target)
    baseline_high_mse = _masked_scaled_mse(stage1, target, high_mask)
    baseline_ramp_mse = _masked_scaled_mse(stage1, target, ramp_mask)
    baseline_extreme_mse = _masked_scaled_mse(stage1, target, extreme_mask)
    baseline_normal_mse = _masked_scaled_mse(stage1, target, ~extreme_mask)
    no_release = _no_text2_release_record(
        horizon=residual.shape[1],
        baseline_mse=baseline_mse,
        baseline_mae=baseline_mae,
        baseline_weighted_mse=baseline_weighted_mse,
        baseline_weighted_mae=baseline_weighted_mae,
        baseline_high_mse=baseline_high_mse,
        baseline_ramp_mse=baseline_ramp_mse,
        baseline_extreme_mse=baseline_extreme_mse,
        baseline_normal_mse=baseline_normal_mse,
    )
    observable_normal_budget = _observable_release_budget(
        feature_map, residual.shape, profile="normal_safe"
    )
    normal_weight = np.where(extreme_mask, 0.25, 1.0).astype(np.float32)
    normal_weight = normal_weight * (1.0 + 1.25 * observable_normal_budget)
    normal_weight = np.clip(normal_weight, 0.20, 3.0).astype(np.float32, copy=False)
    candidates = []
    specs = (
        RESIDUAL_CALIBRATION_SPECS.items()
        if search_all_specs
        else ((preferred, RESIDUAL_CALIBRATION_SPECS[preferred]),)
    )
    for name, features in specs:
        if name == "internal_probe":
            continue
        if fast and search_all_specs and name not in {
            "pareto_regime_basis",
            "event_adaptive_basis",
            "frtc_ber",
            "competitive_treatment_basis",
        }:
            continue
        if any(feature not in feature_map for feature in features):
            continue
        stack = np.stack([feature_map[feature] for feature in features], axis=-1)
        budget_profiles = {
            profile: _observable_release_budget(feature_map, residual.shape, profile=profile)
            for profile in ("none", "soft", "balanced", "extreme", "sparse", "event_sparse")
        }
        normal_budget_profiles = {
            profile: _observable_release_budget(feature_map, residual.shape, profile=profile)
            for profile in ("none", "soft", "balanced", "normal_safe")
        }
        if fast:
            regime_mix_profiles = [
                ("normal_protect", 1.35, 0.04, -0.05),
                ("event_recall", 1.20, 0.03, 0.05),
            ]
        else:
            regime_mix_profiles = [
                ("sharp", 1.55, 0.03, 0.00),
                ("normal_protect", 1.35, 0.04, -0.05),
                ("event_recall", 1.20, 0.03, 0.05),
                ("balanced", 1.05, 0.05, 0.00),
            ]
        regime_mix_maps = {
            mix_name: _observable_regime_mix(
                feature_map,
                residual.shape,
                sharpness=mix_sharpness,
                floor=mix_floor,
                bias=mix_bias,
            )
            for mix_name, mix_sharpness, mix_floor, mix_bias in regime_mix_profiles
        }
        allow_pareto_search = name in {
            "pareto_regime_basis",
            "event_adaptive_basis",
            "frtc_ber",
            "competitive_treatment_basis",
        }
        pareto_cap_values = {0.14, 0.18, 0.22} if fast else {0.08, 0.12, 0.16, 0.20}
        pareto_lambda_values = {1e-3, 1e-2} if fast else {1e-4, 1e-3, 1e-2, 1e-1}
        best = None
        for ridge_lambda in ridge_grid:
            if fast and float(ridge_lambda) not in {1e-3, 1e-2}:
                continue
            event_weights = _fit_horizon_ridge_weights(
                stack,
                residual,
                ridge_lambda=ridge_lambda,
                weight_clip=weight_clip,
                sample_weight=event_weight,
            )
            normal_weights = _fit_horizon_ridge_weights(
                stack,
                residual,
                ridge_lambda=ridge_lambda,
                weight_clip=weight_clip,
                sample_weight=normal_weight,
            )
            delta = _residual_delta_from_weights(stack, event_weights)
            normal_delta = _residual_delta_from_weights(stack, normal_weights)
            for cap in cap_grid:
                single_budget_items = (
                    ((profile, budget_profiles[profile]) for profile in ("balanced", "extreme", "event_sparse"))
                    if fast
                    else budget_profiles.items()
                )
                for budget_profile, observable_budget in single_budget_items:
                    capped = np.clip(delta * observable_budget, -float(cap), float(cap))
                    calibrated = stage1 + capped
                    mse = _scaled_mse(calibrated, target)
                    mae = _scaled_mae(calibrated, target)
                    weighted_mse = _weighted_scaled_mse(calibrated, target, event_weight)
                    weighted_mae = _weighted_scaled_mae(calibrated, target, event_weight)
                    high_mse = _masked_scaled_mse(calibrated, target, high_mask)
                    ramp_mse = _masked_scaled_mse(calibrated, target, ramp_mask)
                    extreme_mse = _masked_scaled_mse(calibrated, target, extreme_mask)
                    normal_mse = _masked_scaled_mse(calibrated, target, ~extreme_mask)
                    high_gain = baseline_high_mse - high_mse
                    ramp_gain = baseline_ramp_mse - ramp_mse
                    extreme_gain = baseline_extreme_mse - extreme_mse
                    normal_degrade = max(0.0, normal_mse - baseline_normal_mse)
                    delta_rms = float(np.sqrt(np.mean(capped ** 2)))
                    release = np.abs(capped) > 1e-5
                    release_coverage = float(np.mean(release))
                    normal_release_coverage = float(np.mean(release[~extreme_mask])) if np.any(~extreme_mask) else 0.0
                    extreme_release_coverage = float(np.mean(release[extreme_mask])) if np.any(extreme_mask) else 0.0
                    release_selectivity_gap = max(0.0, normal_release_coverage - 0.72 * extreme_release_coverage)
                    excessive_release = max(0.0, release_coverage - 0.56)
                    normal_release_over = max(0.0, normal_release_coverage - 0.42)
                    extreme_recall_shortfall = max(0.0, 0.88 - extreme_release_coverage)
                    event_selectivity_reward = max(0.0, extreme_release_coverage - normal_release_coverage)
                    budget_mean = float(np.mean(observable_budget))
                    budget_coverage = float(np.mean(observable_budget > 1e-4))
                    mae_penalty = max(0.0, mae - baseline_mae)
                    weighted_mae_penalty = max(0.0, weighted_mae - baseline_weighted_mae)
                    delta_penalty = max(0.0, delta_rms - 0.115)
                    global_mse_penalty = max(0.0, mse - baseline_mse)
                    global_mse_gain = max(0.0, baseline_mse - mse)
                    event_gain = max(0.0, 0.35 * high_gain + 0.40 * ramp_gain + 0.25 * extreme_gain)
                    release_profile_penalty = {
                        "none": 0.018,
                        "soft": 0.004,
                        "balanced": 0.000,
                        "extreme": 0.002,
                        "sparse": 0.004,
                        "event_sparse": -0.004,
                    }.get(budget_profile, 0.006)
                    event_score = (
                        weighted_mse
                        + 0.34 * mse
                        + 0.72 * global_mse_penalty
                        + 0.34 * mae_penalty
                        + 0.16 * weighted_mae_penalty
                        + 1.05 * normal_degrade
                        + 0.14 * release_selectivity_gap
                        + 0.24 * excessive_release
                        + 0.34 * normal_release_over
                        + 0.18 * extreme_recall_shortfall
                        + 0.12 * delta_penalty
                        + release_profile_penalty
                        - 0.28 * global_mse_gain
                        - 0.42 * high_gain
                        - 0.50 * ramp_gain
                        - 0.32 * extreme_gain
                        - 0.12 * event_gain
                        - 0.08 * event_selectivity_reward
                    )
                    if best is None or event_score < best["val_event_score"]:
                        best = _candidate_metric_record(
                            name=name,
                            features=features,
                            ridge_lambda=ridge_lambda,
                            cap=cap,
                            budget_profile=budget_profile,
                            normal_budget_profile="",
                            variant="single_event_release",
                            mse=mse,
                            mae=mae,
                            weighted_mse=weighted_mse,
                            weighted_mae=weighted_mae,
                            baseline_mse=baseline_mse,
                            baseline_mae=baseline_mae,
                            baseline_weighted_mse=baseline_weighted_mse,
                            baseline_weighted_mae=baseline_weighted_mae,
                            high_mse=high_mse,
                            ramp_mse=ramp_mse,
                            extreme_mse=extreme_mse,
                            normal_mse=normal_mse,
                            baseline_high_mse=baseline_high_mse,
                            baseline_ramp_mse=baseline_ramp_mse,
                            baseline_extreme_mse=baseline_extreme_mse,
                            baseline_normal_mse=baseline_normal_mse,
                            delta_rms=delta_rms,
                            release_coverage=release_coverage,
                            normal_release_coverage=normal_release_coverage,
                            extreme_release_coverage=extreme_release_coverage,
                            budget_mean=budget_mean,
                            budget_coverage=budget_coverage,
                            event_score=event_score,
                        )
                        best["weights"] = event_weights.copy()
                if (
                    not allow_pareto_search
                    or float(cap) not in pareto_cap_values
                    or float(ridge_lambda) not in pareto_lambda_values
                ):
                    continue
                event_profile_names = (
                    ("extreme", "balanced", "event_sparse")
                    if fast
                    else ("event_sparse", "extreme", "balanced", "sparse")
                )
                normal_profile_names = ("normal_safe", "soft") if fast else ("normal_safe", "soft", "balanced")
                for event_profile in event_profile_names:
                    event_budget = budget_profiles[event_profile]
                    event_component = delta * event_budget
                    for normal_profile in normal_profile_names:
                        normal_budget = normal_budget_profiles[normal_profile]
                        normal_component = normal_delta * normal_budget
                        for mix_name, mix_sharpness, mix_floor, mix_bias in regime_mix_profiles:
                            event_mix = regime_mix_maps[mix_name]
                            mixed = event_mix * event_component + (1.0 - event_mix) * normal_component
                            capped = np.clip(mixed, -float(cap), float(cap))
                            calibrated = stage1 + capped
                            mse = _scaled_mse(calibrated, target)
                            mae = _scaled_mae(calibrated, target)
                            weighted_mse = _weighted_scaled_mse(calibrated, target, event_weight)
                            weighted_mae = _weighted_scaled_mae(calibrated, target, event_weight)
                            high_mse = _masked_scaled_mse(calibrated, target, high_mask)
                            ramp_mse = _masked_scaled_mse(calibrated, target, ramp_mask)
                            extreme_mse = _masked_scaled_mse(calibrated, target, extreme_mask)
                            normal_mse = _masked_scaled_mse(calibrated, target, ~extreme_mask)
                            high_gain = baseline_high_mse - high_mse
                            ramp_gain = baseline_ramp_mse - ramp_mse
                            extreme_gain = baseline_extreme_mse - extreme_mse
                            normal_gain = baseline_normal_mse - normal_mse
                            normal_degrade = max(0.0, -normal_gain)
                            delta_rms = float(np.sqrt(np.mean(capped ** 2)))
                            release = np.abs(capped) > 1e-5
                            release_coverage = float(np.mean(release))
                            normal_release_coverage = (
                                float(np.mean(release[~extreme_mask])) if np.any(~extreme_mask) else 0.0
                            )
                            extreme_release_coverage = (
                                float(np.mean(release[extreme_mask])) if np.any(extreme_mask) else 0.0
                            )
                            release_selectivity_gap = max(
                                0.0,
                                normal_release_coverage - 0.76 * extreme_release_coverage,
                            )
                            excessive_release = max(0.0, release_coverage - 0.58)
                            normal_release_over = max(0.0, normal_release_coverage - 0.44)
                            extreme_recall_shortfall = max(0.0, 0.88 - extreme_release_coverage)
                            event_selectivity_reward = max(0.0, extreme_release_coverage - normal_release_coverage)
                            budget_mean = float(np.mean(event_mix * event_budget + (1.0 - event_mix) * normal_budget))
                            budget_coverage = float(
                                np.mean((event_mix * event_budget + (1.0 - event_mix) * normal_budget) > 1e-4)
                            )
                            mae_penalty = max(0.0, mae - baseline_mae)
                            weighted_mae_penalty = max(0.0, weighted_mae - baseline_weighted_mae)
                            delta_penalty = max(0.0, delta_rms - 0.105)
                            global_mse_penalty = max(0.0, mse - baseline_mse)
                            global_mse_gain = max(0.0, baseline_mse - mse)
                            event_gain = max(0.0, 0.34 * high_gain + 0.36 * ramp_gain + 0.30 * extreme_gain)
                            normal_reward = max(0.0, normal_gain)
                            event_floor_penalty = max(0.0, 0.012 - event_gain)
                            profile_penalty = {
                                ("extreme", "normal_safe"): 0.000,
                                ("balanced", "normal_safe"): 0.001,
                                ("sparse", "normal_safe"): 0.002,
                                ("event_sparse", "normal_safe"): -0.005,
                                ("extreme", "soft"): 0.003,
                                ("balanced", "soft"): 0.004,
                                ("event_sparse", "soft"): -0.002,
                                ("extreme", "balanced"): 0.006,
                                ("balanced", "balanced"): 0.007,
                                ("event_sparse", "balanced"): 0.000,
                                ("sparse", "soft"): 0.005,
                                ("sparse", "balanced"): 0.008,
                            }.get((event_profile, normal_profile), 0.006)
                            event_score = (
                                weighted_mse
                                + 0.42 * mse
                                + 0.90 * global_mse_penalty
                                + 0.42 * mae_penalty
                                + 0.16 * weighted_mae_penalty
                                + 1.65 * normal_degrade
                                + 0.16 * release_selectivity_gap
                                + 0.24 * excessive_release
                                + 0.34 * normal_release_over
                                + 0.18 * extreme_recall_shortfall
                                + 0.12 * delta_penalty
                                + 0.18 * event_floor_penalty
                                + profile_penalty
                                - 0.055
                                - 0.42 * global_mse_gain
                                - 0.34 * high_gain
                                - 0.40 * ramp_gain
                                - 0.30 * extreme_gain
                                - 0.18 * event_gain
                                - 0.18 * normal_reward
                                - 0.08 * event_selectivity_reward
                            )
                            if best is None or event_score < best["val_event_score"]:
                                best = _candidate_metric_record(
                                    name=name,
                                    features=features,
                                    ridge_lambda=ridge_lambda,
                                    cap=cap,
                                    budget_profile=event_profile,
                                    normal_budget_profile=normal_profile,
                                    variant="pareto_regime",
                                    mse=mse,
                                    mae=mae,
                                    weighted_mse=weighted_mse,
                                    weighted_mae=weighted_mae,
                                    baseline_mse=baseline_mse,
                                    baseline_mae=baseline_mae,
                                    baseline_weighted_mse=baseline_weighted_mse,
                                    baseline_weighted_mae=baseline_weighted_mae,
                                    high_mse=high_mse,
                                    ramp_mse=ramp_mse,
                                    extreme_mse=extreme_mse,
                                    normal_mse=normal_mse,
                                    baseline_high_mse=baseline_high_mse,
                                    baseline_ramp_mse=baseline_ramp_mse,
                                    baseline_extreme_mse=baseline_extreme_mse,
                                    baseline_normal_mse=baseline_normal_mse,
                                    delta_rms=delta_rms,
                                    release_coverage=release_coverage,
                                    normal_release_coverage=normal_release_coverage,
                                    extreme_release_coverage=extreme_release_coverage,
                                    budget_mean=budget_mean,
                                    budget_coverage=budget_coverage,
                                    event_score=event_score,
                                )
                                best["weights"] = event_weights.copy()
                                best["event_weights"] = event_weights.copy()
                                best["normal_weights"] = normal_weights.copy()
                                best["regime_mix_profile"] = mix_name
                                best["regime_mix_sharpness"] = float(mix_sharpness)
                                best["regime_mix_floor"] = float(mix_floor)
                                best["regime_mix_bias"] = float(mix_bias)
                                best["val_event_mix_mean"] = float(np.mean(event_mix))
                                best["val_event_mix_event_mean"] = (
                                    float(np.mean(event_mix[extreme_mask])) if np.any(extreme_mask) else 0.0
                                )
                                best["val_event_mix_normal_mean"] = (
                                    float(np.mean(event_mix[~extreme_mask])) if np.any(~extreme_mask) else 0.0
                                )
        if best is not None:
            candidates.append(best)
    if "route_prediction_delta" in feature_map:
        route_features = ("route_prediction_delta",)
        route_stack = np.stack([feature_map[feature] for feature in route_features], axis=-1)
        route_best = None
        for ridge_lambda in ridge_grid:
            if fast and float(ridge_lambda) not in {1e-3, 1e-2}:
                continue
            route_weights = _fit_horizon_ridge_weights(
                route_stack,
                residual,
                ridge_lambda=ridge_lambda,
                weight_clip=weight_clip,
                sample_weight=event_weight,
            )
            route_delta = _residual_delta_from_weights(route_stack, route_weights)
            for cap in cap_grid:
                capped = np.clip(route_delta, -float(cap), float(cap))
                calibrated = stage1 + capped
                mse = _scaled_mse(calibrated, target)
                mae = _scaled_mae(calibrated, target)
                weighted_mse = _weighted_scaled_mse(calibrated, target, event_weight)
                weighted_mae = _weighted_scaled_mae(calibrated, target, event_weight)
                high_mse = _masked_scaled_mse(calibrated, target, high_mask)
                ramp_mse = _masked_scaled_mse(calibrated, target, ramp_mask)
                extreme_mse = _masked_scaled_mse(calibrated, target, extreme_mask)
                normal_mse = _masked_scaled_mse(calibrated, target, ~extreme_mask)
                high_gain = baseline_high_mse - high_mse
                ramp_gain = baseline_ramp_mse - ramp_mse
                extreme_gain = baseline_extreme_mse - extreme_mse
                normal_gain = baseline_normal_mse - normal_mse
                normal_degrade = max(0.0, -normal_gain)
                delta_rms = float(np.sqrt(np.mean(capped ** 2)))
                release = np.abs(capped) > 1e-5
                release_coverage = float(np.mean(release))
                normal_release_coverage = float(np.mean(release[~extreme_mask])) if np.any(~extreme_mask) else 0.0
                extreme_release_coverage = float(np.mean(release[extreme_mask])) if np.any(extreme_mask) else 0.0
                mae_penalty = max(0.0, mae - baseline_mae)
                weighted_mae_penalty = max(0.0, weighted_mae - baseline_weighted_mae)
                global_mse_penalty = max(0.0, mse - baseline_mse)
                global_mse_gain = max(0.0, baseline_mse - mse)
                event_gain = max(0.0, 0.34 * high_gain + 0.38 * ramp_gain + 0.28 * extreme_gain)
                route_score = (
                    weighted_mse
                    + 0.36 * mse
                    + 0.80 * global_mse_penalty
                    + 0.30 * mae_penalty
                    + 0.14 * weighted_mae_penalty
                    + 1.10 * normal_degrade
                    + 0.10 * max(0.0, delta_rms - 0.12)
                    - 0.38 * global_mse_gain
                    - 0.32 * high_gain
                    - 0.38 * ramp_gain
                    - 0.28 * extreme_gain
                    - 0.16 * event_gain
                )
                record = _candidate_metric_record(
                    name="semantic_route_direct",
                    features=route_features,
                    ridge_lambda=ridge_lambda,
                    cap=cap,
                    budget_profile="route_direct",
                    normal_budget_profile="",
                    variant="semantic_route_direct",
                    mse=mse,
                    mae=mae,
                    weighted_mse=weighted_mse,
                    weighted_mae=weighted_mae,
                    baseline_mse=baseline_mse,
                    baseline_mae=baseline_mae,
                    baseline_weighted_mse=baseline_weighted_mse,
                    baseline_weighted_mae=baseline_weighted_mae,
                    high_mse=high_mse,
                    ramp_mse=ramp_mse,
                    extreme_mse=extreme_mse,
                    normal_mse=normal_mse,
                    baseline_high_mse=baseline_high_mse,
                    baseline_ramp_mse=baseline_ramp_mse,
                    baseline_extreme_mse=baseline_extreme_mse,
                    baseline_normal_mse=baseline_normal_mse,
                    delta_rms=delta_rms,
                    release_coverage=release_coverage,
                    normal_release_coverage=normal_release_coverage,
                    extreme_release_coverage=extreme_release_coverage,
                    budget_mean=1.0,
                    budget_coverage=release_coverage,
                    event_score=route_score,
                )
                record["weights"] = route_weights.copy()
                record["use_observable_release_budget"] = False
                if route_best is None or record["val_event_score"] < route_best["val_event_score"]:
                    route_best = record
        if route_best is not None:
            candidates.append(route_best)
    route_event_features = tuple(
        feature
        for feature in (
            "route_prediction_delta",
            "event_counterfactual_basis_delta",
            "event_proposal_basis_delta",
            "event_dataset_basis_delta",
            "event_ramp_basis_delta",
            "event_realtime_basis_delta",
            "event_nwp_disagreement_basis_delta",
            "competitive_event_treatment_delta",
            "competitive_trend_treatment_delta",
            "event_stability_margin_gate",
            "normal_trend_basis_delta",
            "text2_treatment_effect",
            "counterfactual_gap_delta",
            "text_ramp_prior",
            "shock_delta",
            "nwp_disagreement_delta",
        )
        if feature in feature_map
    )
    if "route_prediction_delta" in route_event_features and len(route_event_features) >= 4:
        route_event_stack = np.stack([feature_map[feature] for feature in route_event_features], axis=-1)
        route_event_best = None
        for ridge_lambda in ridge_grid:
            if fast and float(ridge_lambda) not in {1e-3, 1e-2}:
                continue
            route_event_weights = _fit_horizon_ridge_weights(
                route_event_stack,
                residual,
                ridge_lambda=ridge_lambda,
                weight_clip=weight_clip,
                sample_weight=event_weight,
            )
            route_event_delta = _residual_delta_from_weights(route_event_stack, route_event_weights)
            for cap in cap_grid:
                capped = np.clip(route_event_delta, -float(cap), float(cap))
                calibrated = stage1 + capped
                mse = _scaled_mse(calibrated, target)
                mae = _scaled_mae(calibrated, target)
                weighted_mse = _weighted_scaled_mse(calibrated, target, event_weight)
                weighted_mae = _weighted_scaled_mae(calibrated, target, event_weight)
                high_mse = _masked_scaled_mse(calibrated, target, high_mask)
                ramp_mse = _masked_scaled_mse(calibrated, target, ramp_mask)
                extreme_mse = _masked_scaled_mse(calibrated, target, extreme_mask)
                normal_mse = _masked_scaled_mse(calibrated, target, ~extreme_mask)
                high_gain = baseline_high_mse - high_mse
                ramp_gain = baseline_ramp_mse - ramp_mse
                extreme_gain = baseline_extreme_mse - extreme_mse
                normal_gain = baseline_normal_mse - normal_mse
                normal_degrade = max(0.0, -normal_gain)
                delta_rms = float(np.sqrt(np.mean(capped ** 2)))
                release = np.abs(capped) > 1e-5
                release_coverage = float(np.mean(release))
                normal_release_coverage = float(np.mean(release[~extreme_mask])) if np.any(~extreme_mask) else 0.0
                extreme_release_coverage = float(np.mean(release[extreme_mask])) if np.any(extreme_mask) else 0.0
                mae_penalty = max(0.0, mae - baseline_mae)
                weighted_mae_penalty = max(0.0, weighted_mae - baseline_weighted_mae)
                global_mse_penalty = max(0.0, mse - baseline_mse)
                global_mse_gain = max(0.0, baseline_mse - mse)
                event_gain = max(0.0, 0.34 * high_gain + 0.38 * ramp_gain + 0.28 * extreme_gain)
                route_event_score = (
                    weighted_mse
                    + 0.34 * mse
                    + 0.80 * global_mse_penalty
                    + 0.30 * mae_penalty
                    + 0.14 * weighted_mae_penalty
                    + 1.18 * normal_degrade
                    + 0.007 * max(0, len(route_event_features) - 4)
                    + 0.12 * max(0.0, delta_rms - 0.12)
                    - 0.40 * global_mse_gain
                    - 0.34 * high_gain
                    - 0.42 * ramp_gain
                    - 0.30 * extreme_gain
                    - 0.18 * event_gain
                )
                record = _candidate_metric_record(
                    name="semantic_route_event_direct",
                    features=route_event_features,
                    ridge_lambda=ridge_lambda,
                    cap=cap,
                    budget_profile="route_event_direct",
                    normal_budget_profile="",
                    variant="semantic_route_event_direct",
                    mse=mse,
                    mae=mae,
                    weighted_mse=weighted_mse,
                    weighted_mae=weighted_mae,
                    baseline_mse=baseline_mse,
                    baseline_mae=baseline_mae,
                    baseline_weighted_mse=baseline_weighted_mse,
                    baseline_weighted_mae=baseline_weighted_mae,
                    high_mse=high_mse,
                    ramp_mse=ramp_mse,
                    extreme_mse=extreme_mse,
                    normal_mse=normal_mse,
                    baseline_high_mse=baseline_high_mse,
                    baseline_ramp_mse=baseline_ramp_mse,
                    baseline_extreme_mse=baseline_extreme_mse,
                    baseline_normal_mse=baseline_normal_mse,
                    delta_rms=delta_rms,
                    release_coverage=release_coverage,
                    normal_release_coverage=normal_release_coverage,
                    extreme_release_coverage=extreme_release_coverage,
                    budget_mean=1.0,
                    budget_coverage=release_coverage,
                    event_score=route_event_score,
                )
                record["weights"] = route_event_weights.copy()
                record["use_observable_release_budget"] = False
                if route_event_best is None or record["val_event_score"] < route_event_best["val_event_score"]:
                    route_event_best = record
        if route_event_best is not None:
            candidates.append(route_event_best)
    switch_event_features = tuple(
        feature
        for feature in (
            "event_counterfactual_basis_delta",
            "event_proposal_basis_delta",
            "event_dataset_basis_delta",
            "event_ramp_basis_delta",
            "event_realtime_basis_delta",
            "event_nwp_disagreement_basis_delta",
            "extreme_weather_evidence_delta",
            "fuzzy_ramp_evidence_delta",
            "shock_direction_delta",
            "text_scalar_risk_delta",
            "nwp_disagreement_delta",
            "text_ramp_prior",
            "shock_delta",
        )
        if feature in feature_map
    )
    if "route_prediction_delta" in feature_map and len(switch_event_features) >= 4:
        switch_features = ("route_prediction_delta",) + switch_event_features
        route_stack = np.stack([feature_map["route_prediction_delta"]], axis=-1)
        event_stack = np.stack([feature_map[feature] for feature in switch_event_features], axis=-1)
        switch_budget_names = ("extreme", "event_sparse") if fast else (
            "extreme",
            "event_sparse",
            "sparse",
            "balanced",
        )
        event_budgets = {
            profile: _observable_release_budget(feature_map, residual.shape, profile=profile)
            for profile in switch_budget_names
        }
        switch_mix_profiles = (
            [("event_recall", 1.20, 0.03, 0.05)]
            if fast
            else [
                ("sharp", 1.55, 0.03, 0.00),
                ("normal_protect", 1.35, 0.04, -0.05),
                ("event_recall", 1.20, 0.03, 0.05),
            ]
        )
        switch_mix_maps = {
            mix_name: _observable_regime_mix(
                feature_map,
                residual.shape,
                sharpness=mix_sharpness,
                floor=mix_floor,
                bias=mix_bias,
            )
            for mix_name, mix_sharpness, mix_floor, mix_bias in switch_mix_profiles
        }
        switch_best = None
        route_lambdas = (1e-4, 1e-3, 1e-2) if not fast else (1e-3, 1e-2)
        event_lambdas = (1e-4, 1e-3, 1e-2, 1e-1) if not fast else (1e-3, 1e-2)
        switch_cap_values = (
            0.08,
            0.12,
            0.16,
            0.20,
            0.24,
            0.28,
            0.32,
            0.36,
            0.40,
            0.44,
            0.48,
        ) if not fast else (0.14, 0.18, 0.22, 0.28)
        switch_transforms = ("budget_mix",) if fast else ("budget_mix", "budget_only", "sharp_budget")
        for route_lambda in route_lambdas:
            route_weights_small = _fit_horizon_ridge_weights(
                route_stack,
                residual,
                ridge_lambda=route_lambda,
                weight_clip=weight_clip,
                sample_weight=normal_weight,
            )
            route_delta = _residual_delta_from_weights(route_stack, route_weights_small)
            for event_lambda in event_lambdas:
                event_weights_small = _fit_horizon_ridge_weights(
                    event_stack,
                    residual,
                    ridge_lambda=event_lambda,
                    weight_clip=weight_clip,
                    sample_weight=event_weight,
                )
                event_delta = _residual_delta_from_weights(event_stack, event_weights_small)
                for cap in switch_cap_values:
                    route_component = np.clip(route_delta, -float(cap), float(cap))
                    event_component = np.clip(event_delta, -float(cap), float(cap))
                    for budget_profile, event_budget in event_budgets.items():
                        for mix_name, mix_sharpness, mix_floor, mix_bias in switch_mix_profiles:
                            event_mix = switch_mix_maps[mix_name]
                            for switch_transform in switch_transforms:
                                if switch_transform == "budget_only":
                                    switch = event_budget
                                elif switch_transform == "sharp_budget":
                                    switch = np.clip((event_budget - 0.06) / 0.34, 0.0, 1.0)
                                    switch = np.maximum(switch, event_budget * event_mix)
                                else:
                                    switch = np.clip(event_budget * (0.35 + 0.65 * event_mix), 0.0, 1.0)
                                mixed = (1.0 - switch) * route_component + switch * event_component
                                capped = np.clip(mixed, -float(cap), float(cap))
                                calibrated = stage1 + capped
                                mse = _scaled_mse(calibrated, target)
                                mae = _scaled_mae(calibrated, target)
                                weighted_mse = _weighted_scaled_mse(calibrated, target, event_weight)
                                weighted_mae = _weighted_scaled_mae(calibrated, target, event_weight)
                                high_mse = _masked_scaled_mse(calibrated, target, high_mask)
                                ramp_mse = _masked_scaled_mse(calibrated, target, ramp_mask)
                                extreme_mse = _masked_scaled_mse(calibrated, target, extreme_mask)
                                normal_mse = _masked_scaled_mse(calibrated, target, ~extreme_mask)
                                high_gain = baseline_high_mse - high_mse
                                ramp_gain = baseline_ramp_mse - ramp_mse
                                extreme_gain = baseline_extreme_mse - extreme_mse
                                normal_gain = baseline_normal_mse - normal_mse
                                normal_degrade = max(0.0, -normal_gain)
                                delta_rms = float(np.sqrt(np.mean(capped ** 2)))
                                release = np.abs(capped) > 1e-5
                                release_coverage = float(np.mean(release))
                                normal_release_coverage = (
                                    float(np.mean(release[~extreme_mask])) if np.any(~extreme_mask) else 0.0
                                )
                                extreme_release_coverage = (
                                    float(np.mean(release[extreme_mask])) if np.any(extreme_mask) else 0.0
                                )
                                switch_event_mean = float(np.mean(switch[extreme_mask])) if np.any(extreme_mask) else 0.0
                                switch_normal_mean = float(np.mean(switch[~extreme_mask])) if np.any(~extreme_mask) else 0.0
                                switch_selectivity = max(0.0, switch_event_mean - switch_normal_mean)
                                mae_penalty = max(0.0, mae - baseline_mae)
                                weighted_mae_penalty = max(0.0, weighted_mae - baseline_weighted_mae)
                                global_mse_penalty = max(0.0, mse - baseline_mse)
                                global_mse_gain = max(0.0, baseline_mse - mse)
                                event_gain = max(0.0, 0.34 * high_gain + 0.40 * ramp_gain + 0.26 * extreme_gain)
                                switch_score = (
                                    weighted_mse
                                    + 0.30 * mse
                                    + 0.70 * global_mse_penalty
                                    + 0.26 * mae_penalty
                                    + 0.12 * weighted_mae_penalty
                                    + 1.10 * normal_degrade
                                    + 0.10 * max(0.0, delta_rms - 0.12)
                                    + 0.04 * max(0.0, switch_normal_mean - 0.38)
                                    - 0.46 * global_mse_gain
                                    - 0.38 * high_gain
                                    - 0.46 * ramp_gain
                                    - 0.34 * extreme_gain
                                    - 0.20 * event_gain
                                    - 0.05 * switch_selectivity
                                )
                                route_weights_full = np.zeros(
                                    (residual.shape[1], len(switch_features)),
                                    dtype=np.float32,
                                )
                                event_weights_full = np.zeros_like(route_weights_full)
                                route_weights_full[:, 0] = route_weights_small[:, 0]
                                event_weights_full[:, 1:] = event_weights_small
                                record = _candidate_metric_record(
                                    name="semantic_route_event_switch",
                                    features=switch_features,
                                    ridge_lambda=event_lambda,
                                    cap=cap,
                                    budget_profile=budget_profile,
                                    normal_budget_profile="route_direct",
                                    variant="semantic_route_event_switch",
                                    mse=mse,
                                    mae=mae,
                                    weighted_mse=weighted_mse,
                                    weighted_mae=weighted_mae,
                                    baseline_mse=baseline_mse,
                                    baseline_mae=baseline_mae,
                                    baseline_weighted_mse=baseline_weighted_mse,
                                    baseline_weighted_mae=baseline_weighted_mae,
                                    high_mse=high_mse,
                                    ramp_mse=ramp_mse,
                                    extreme_mse=extreme_mse,
                                    normal_mse=normal_mse,
                                    baseline_high_mse=baseline_high_mse,
                                    baseline_ramp_mse=baseline_ramp_mse,
                                    baseline_extreme_mse=baseline_extreme_mse,
                                    baseline_normal_mse=baseline_normal_mse,
                                    delta_rms=delta_rms,
                                    release_coverage=release_coverage,
                                    normal_release_coverage=normal_release_coverage,
                                    extreme_release_coverage=extreme_release_coverage,
                                    budget_mean=float(np.mean(switch)),
                                    budget_coverage=float(np.mean(switch > 1e-4)),
                                    event_score=switch_score,
                                )
                                record["weights"] = route_weights_full
                                record["event_weights"] = event_weights_full
                                record["route_lambda"] = float(route_lambda)
                                record["event_lambda"] = float(event_lambda)
                                record["regime_mix_profile"] = mix_name
                                record["regime_mix_sharpness"] = float(mix_sharpness)
                                record["regime_mix_floor"] = float(mix_floor)
                                record["regime_mix_bias"] = float(mix_bias)
                                record["switch_transform"] = switch_transform
                                record["val_event_mix_mean"] = float(np.mean(switch))
                                record["val_event_mix_event_mean"] = switch_event_mean
                                record["val_event_mix_normal_mean"] = switch_normal_mean
                                if switch_best is None or record["val_event_score"] < switch_best["val_event_score"]:
                                    switch_best = record
        if switch_best is not None:
            candidates.append(switch_best)
    if not candidates:
        return no_release if safe_fallback else None
    candidate_scores_source = list(candidates)
    if safe_fallback:
        styled_candidates = []
        for candidate in candidate_scores_source:
            try:
                raw_delta, _, _, _ = _candidate_calibration_delta(
                    candidate,
                    feature_map,
                    residual.shape,
                    dtype=np.float32,
                    apply_release_scale=False,
                )
                styled_candidates.append(
                    _attach_dataset_style_release(
                        candidate,
                        stage1,
                        target,
                        raw_delta,
                        event_weight=event_weight,
                        high_mask=high_mask,
                        ramp_mask=ramp_mask,
                        extreme_mask=extreme_mask,
                        baseline_mse=baseline_mse,
                        baseline_mae=baseline_mae,
                        baseline_weighted_mse=baseline_weighted_mse,
                        baseline_weighted_mae=baseline_weighted_mae,
                        baseline_high_mse=baseline_high_mse,
                        baseline_ramp_mse=baseline_ramp_mse,
                        baseline_extreme_mse=baseline_extreme_mse,
                        baseline_normal_mse=baseline_normal_mse,
                    )
                )
            except Exception as exc:
                candidate["style_release_error"] = str(exc)
                continue
        candidate_scores_source = styled_candidates
    if not candidate_scores_source:
        selected = no_release
    else:
        candidate_scores_source.sort(
            key=lambda item: (
                _soft_pareto_candidate_score(item),
                0 if item.get("style_release_scale", 1.0) > 1e-6 else 1,
                0 if item.get("variant") == "pareto_regime" else 1,
                0 if item["name"] == preferred else 1,
                -item.get("val_ramp_event_improve_MSE", 0.0),
                -item.get("val_high_residual_improve_MSE", 0.0),
                len(item["features"]),
            )
        )
        selected = candidate_scores_source[0]
    all_candidate_scores = list(candidate_scores_source) + [no_release]
    all_candidate_scores.sort(
        key=lambda item: (
            _soft_pareto_candidate_score(item),
            0 if item.get("style_release_scale", 1.0) > 1e-6 else 1,
            0 if item.get("variant") == "pareto_regime" else 1,
            0 if item["name"] == preferred else 1,
            -item.get("val_ramp_event_improve_MSE", 0.0),
            -item.get("val_high_residual_improve_MSE", 0.0),
            len(item["features"]),
        )
    )
    force_candidate_name = str(force_candidate_name or "").strip()
    force_candidate_variant = str(force_candidate_variant or "").strip()
    forced_selection = False
    if force_candidate_name or force_candidate_variant:
        forced_matches = [
            item
            for item in candidate_scores_source
            if (not force_candidate_name or item.get("name") == force_candidate_name)
            and (not force_candidate_variant or item.get("variant") == force_candidate_variant)
            and item.get("variant") != "no_text2_release"
            and item.get("style_release_scale", 1.0) > 1e-6
        ]
        if forced_matches:
            forced_matches.sort(
                key=lambda item: (
                    -float(item.get("val_scaled_improve_MSE", 0.0)),
                    -float(item.get("val_extreme_union_improve_MSE", 0.0)),
                    float(item.get("val_event_score", 1e9)),
                )
            )
            selected = forced_matches[0]
            selected["fallback_reason"] = "forced_validation_candidate_for_eval"
            forced_selection = True
    if not forced_selection:
        preferred_slack = 4e-3
        for item in candidate_scores_source:
            if (
                item["name"] == preferred
                and item["val_event_score"] <= selected["val_event_score"] + preferred_slack
                and item.get("variant") != "no_text2_release"
                and item.get("style_release_scale", 1.0) > 1e-6
            ):
                selected = item
                break
        normal_safe_slack = 4.0e-2
        for item in candidate_scores_source:
            selected_event_gain = max(
                selected.get("val_ramp_event_improve_MSE", 0.0),
                selected.get("val_extreme_union_improve_MSE", 0.0),
                selected.get("val_high_residual_improve_MSE", 0.0),
            )
            item_event_gain = max(
                item.get("val_ramp_event_improve_MSE", 0.0),
                item.get("val_extreme_union_improve_MSE", 0.0),
                item.get("val_high_residual_improve_MSE", 0.0),
            )
            if (
                item.get("style_release_scale", 1.0) > 1e-6
                and item.get("val_event_score", 1e9) <= selected.get("val_event_score", 1e9) + normal_safe_slack
                and item.get("val_scaled_improve_MSE", 0.0) >= min_global_gain
                and item.get("val_extreme_union_improve_MSE", 0.0) >= min_extreme_gain
                and item.get("val_normal_degrade_MSE", 0.0) <= 1e-5
                and item_event_gain >= 0.65 * max(selected_event_gain, min_extreme_gain)
                and item.get("val_fold_global_positive_rate", 0.0) >= 0.75
                and item.get("val_fold_event_positive_rate", 0.0) >= 0.75
            ):
                selected = item
                selected["fallback_reason"] = "normal_safe_pareto_release"
                break
        selected = _select_bicriteria_release_candidate(
            candidate_scores_source,
            selected,
            preferred=preferred,
            min_global_gain=min_global_gain,
            min_extreme_gain=min_extreme_gain,
        )
        selected = _select_pareto_safe_extreme_candidate(
            candidate_scores_source,
            selected,
            preferred=preferred,
            min_global_gain=min_global_gain,
            min_extreme_gain=min_extreme_gain,
        )
    if float(force_release_scale) >= 0.0:
        forced_scale = float(np.clip(float(force_release_scale), 0.0, 1.0))
        selected["style_release_scale"] = forced_scale
        selected["release_scale"] = forced_scale
        selected["forced_release_scale"] = forced_scale
        selected["fallback_reason"] = "forced_release_scale_for_eval"
    if safe_fallback and selected.get("variant") != "no_text2_release":
        if float(selected.get("style_release_scale", 1.0)) <= 1e-6:
            no_release["fallback_reason"] = "dataset_style_release_scale_zero"
            no_release["rejected_candidate"] = {
                "name": selected.get("name"),
                "variant": selected.get("variant"),
                "val_scaled_improve_MSE": selected.get("val_scaled_improve_MSE", 0.0),
                "val_extreme_union_improve_MSE": selected.get("val_extreme_union_improve_MSE", 0.0),
                "val_normal_degrade_MSE": selected.get("val_normal_degrade_MSE", 0.0),
                "style_release_scale": selected.get("style_release_scale", 0.0),
                "val_fold_global_gain_min": selected.get("val_fold_global_gain_min", 0.0),
                "val_fold_event_gain_min": selected.get("val_fold_event_gain_min", 0.0),
            }
            selected = no_release
        else:
            selected["fallback_reason"] = "dataset_style_release_scale"
    if safe_fallback and selected.get("variant") != "no_text2_release":
        accepted, reason = _accept_text2_calibration(
            selected,
            min_global_gain=min_global_gain,
            min_extreme_gain=min_extreme_gain,
        )
        if not accepted:
            no_release["fallback_reason"] = reason
            no_release["rejected_candidate"] = {
                "name": selected.get("name"),
                "variant": selected.get("variant"),
                "val_scaled_improve_MSE": selected.get("val_scaled_improve_MSE", 0.0),
                "val_weighted_scaled_improve_MSE": selected.get("val_weighted_scaled_improve_MSE", 0.0),
                "val_high_residual_improve_MSE": selected.get("val_high_residual_improve_MSE", 0.0),
                "val_ramp_event_improve_MSE": selected.get("val_ramp_event_improve_MSE", 0.0),
                "val_extreme_union_improve_MSE": selected.get("val_extreme_union_improve_MSE", 0.0),
                "val_normal_degrade_MSE": selected.get("val_normal_degrade_MSE", 0.0),
                "style_release_scale": selected.get("style_release_scale", 1.0),
                "val_fold_global_gain_min": selected.get("val_fold_global_gain_min", 0.0),
                "val_fold_event_gain_min": selected.get("val_fold_event_gain_min", 0.0),
            }
            selected = no_release
        else:
            selected["fallback_reason"] = reason
    selected["candidate_scores"] = [
        {
            "name": item["name"],
            "features": list(item["features"]),
            "lambda": item["lambda"],
            "cap": item["cap"],
            "variant": item.get("variant", "single_event_release"),
            "budget_profile": item.get("budget_profile", "balanced"),
            "normal_budget_profile": item.get("normal_budget_profile", ""),
            "regime_mix_profile": item.get("regime_mix_profile", ""),
            "val_scaled_MSE": item["val_scaled_MSE"],
            "val_scaled_MAE": item["val_scaled_MAE"],
            "val_scaled_improve_MSE": item["val_scaled_improve_MSE"],
            "val_scaled_improve_MAE": item["val_scaled_improve_MAE"],
            "val_weighted_scaled_MSE": item["val_weighted_scaled_MSE"],
            "val_weighted_scaled_MAE": item["val_weighted_scaled_MAE"],
            "val_weighted_scaled_improve_MSE": item["val_weighted_scaled_improve_MSE"],
            "val_weighted_scaled_improve_MAE": item["val_weighted_scaled_improve_MAE"],
            "val_high_residual_scaled_MSE": item["val_high_residual_scaled_MSE"],
            "val_high_residual_improve_MSE": item["val_high_residual_improve_MSE"],
            "val_ramp_event_scaled_MSE": item["val_ramp_event_scaled_MSE"],
            "val_ramp_event_improve_MSE": item["val_ramp_event_improve_MSE"],
            "val_extreme_union_scaled_MSE": item["val_extreme_union_scaled_MSE"],
            "val_extreme_union_improve_MSE": item["val_extreme_union_improve_MSE"],
            "val_normal_scaled_MSE": item["val_normal_scaled_MSE"],
            "val_normal_improve_MSE": item.get("val_normal_improve_MSE", 0.0),
            "val_normal_degrade_MSE": item["val_normal_degrade_MSE"],
            "val_delta_rms_scaled": item["val_delta_rms_scaled"],
            "val_event_mix_mean": item.get("val_event_mix_mean", 0.0),
            "val_event_mix_event_mean": item.get("val_event_mix_event_mean", 0.0),
            "val_event_mix_normal_mean": item.get("val_event_mix_normal_mean", 0.0),
            "style_release_scale": item.get("style_release_scale", item.get("release_scale", 1.0)),
            "val_fold_global_gain_min": item.get("val_fold_global_gain_min", 0.0),
            "val_fold_global_gain_mean": item.get("val_fold_global_gain_mean", 0.0),
            "val_fold_global_positive_rate": item.get("val_fold_global_positive_rate", 0.0),
            "val_fold_event_gain_min": item.get("val_fold_event_gain_min", 0.0),
            "val_fold_event_gain_mean": item.get("val_fold_event_gain_mean", 0.0),
            "val_fold_event_positive_rate": item.get("val_fold_event_positive_rate", 0.0),
            "val_fold_normal_degrade_max": item.get("val_fold_normal_degrade_max", 0.0),
            "val_fold_instability": item.get("val_fold_instability", 0.0),
            "val_style_release_score": item.get("val_style_release_score", item.get("val_event_score", 0.0)),
            "val_event_score": item["val_event_score"],
        }
        for item in all_candidate_scores
    ]
    return selected


def run_text2(args):
    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    amp = bool(args.amp and device.type == "cuda")
    _require_checkpoint_matches_current_data(args.baseline_ckpt, args, "baseline")
    _require_checkpoint_matches_current_data(args.text1_ckpt, args, "text1")
    baseline, baseline_checkpoint = load_baseline_model(args.baseline_ckpt, device)
    if bool(getattr(args, "multi_dataset", False)):
        text1_cache = _build_text1_caches(args, device)
        text2_cache = _build_text2_caches(args, device)
        first_text1 = _first_cache(text1_cache)
        first_text2 = _first_cache(text2_cache)
        text1_dim = int(first_text1["state_prompt_sentence"].shape[1])
        text2_dim = int(first_text2["low_frequency_trend_prompt"].shape[1])
        text2_scalar_dim = int(
            first_text2.get("text2_scalar", np.zeros((1, TEXT2_SCALAR_DIM), dtype=np.float32)).shape[1]
        )
    else:
        _assert_safe_text2_path(args.text2_path)
        text1_cache = build_text1_cache(
            args.text1_path,
            args.text_model,
            str(device),
            args.cache_dir,
            batch_size=args.text_batch_size,
            max_tokens=args.text_max_tokens,
        )
        text2_cache = build_text2_cache(
            args.text2_path,
            args.text_model,
            str(device),
            args.cache_dir,
            batch_size=args.text_batch_size,
        )
        text1_dim = int(text1_cache["state_prompt_sentence"].shape[1])
        text2_dim = int(text2_cache["low_frequency_trend_prompt"].shape[1])
        text2_scalar_dim = int(
            text2_cache.get("text2_scalar", np.zeros((1, TEXT2_SCALAR_DIM), dtype=np.float32)).shape[1]
        )
    loaders = make_solar_loaders(args, text1_cache=text1_cache, text2_cache=text2_cache)
    if args.text1_ckpt:
        model, text1_checkpoint = load_fusion_model(
            args.text1_ckpt,
            baseline,
            device,
            text1_dim_override=text1_dim,
            text2_dim_override=text2_dim,
            args_override=args,
            load_text2_modules=bool(args.reuse_text2_modules),
        )
        model.use_text2_correction = True
        if args.freeze_text1_for_text2:
            if args.freeze_text2_low_for_high:
                freeze_for_text2_high_stage(model)
            else:
                freeze_for_text2_stage(model)
    else:
        model = build_fusion_model(
            args,
            baseline,
            text1_dim=text1_dim,
            text2_dim=text2_dim,
            use_text2_correction=True,
        ).to(device)
        text1_checkpoint = {}
    return train_and_select(
        args,
        model,
        loaders,
        device,
        amp,
        args.text2_out_dir,
        PAPER_MAINLINE_NAME,
        checkpoint_extra={
            "model_role": "paper_main_frtc_ber",
            "method_name": METHOD_NAME,
            "method_full_name": METHOD_FULL_NAME,
            "method_core_claim": METHOD_CORE_CLAIM,
            "method_pipeline": list(METHOD_PIPELINE),
            "method_card": METHOD_CARD,
            "paper_stage": "final_forecast_time_residual_intervention",
            "paper_visibility": "main_result",
            "contribution_1": PAPER_MAIN_CONTRIBUTIONS[0],
            "contribution_2": PAPER_MAIN_CONTRIBUTIONS[1],
            "contribution_3": PAPER_MAIN_CONTRIBUTIONS[2],
            "stage1_module": "iTransformer_numeric_foundation_plus_asynchronous_realtime_text_residual_intervention",
            "stage2_module": "forecast_time_available_text2_counterfactual_fuzzy_regime_pareto_residual_release",
            "baseline_checkpoint": args.baseline_ckpt,
            "baseline_code_version": baseline_checkpoint.get("code_version", "unknown"),
            "text1_checkpoint": args.text1_ckpt,
            "reuse_text2_modules": bool(args.reuse_text2_modules),
            "text1_code_version": text1_checkpoint.get("code_version", "unknown"),
            **_single_or_multi_path_metadata(args),
            "text1_dim": text1_dim,
            "text2_dim": text2_dim,
            "text2_scalar_dim": text2_scalar_dim,
            "text2_availability_constraint": (
                "Text2 must be available at forecast issue time; default Text2 "
                "is NWP-derived horizon prompt aligned by forecast valid time, "
                "not post-event observed text."
            ),
            "paper_formulas": PAPER_METHOD_FORMULAS,
            "inference_constraints": list(PAPER_INFERENCE_CONSTRAINTS),
            "inference_residual_policy": (
                "Final forecast = stage1_text_prediction plus a budgeted "
                "released residual. Text2 proposes counterfactual residual "
                "treatment effects, fuzzy extreme/ramp evidence decides where "
                "residual capacity should be spent, and validation labels are "
                "used only to fit fixed release weights."
            ),
            "release_gate_module": (
                "Budgeted extreme-aware residual release: compares factual "
                "Text2 residual proposals with neutral-text counterfactuals, "
                "uses fuzzy extreme/ramp semantics and NWP disagreement as "
                "observable release evidence, and caps horizon-wise released "
                "residuals to protect normal-weather forecasts."
            ),
            "causal_intervention_module": (
                "Textual residual intervention: defines forecast-time Text2 "
                "contribution through a factual-vs-neutral residual treatment "
                "effect, separates low-frequency trend treatment from "
                "high-frequency event treatment, and releases the event part "
                "more actively under observable extreme regimes."
            ),
            "dataset_evidence_release_module": (
                "Forecast-time evidence basis: exposes "
                "evidence from realtime Text1 confidence, Text2 scalar "
                "direction/risk/confidence, fuzzy extreme semantics, NWP "
                "candidate disagreement, and text coverage as residual-release "
                "basis functions. It increases information utilization without "
                "using future true residuals at inference."
            ),
            "text_contribution_objective": "forecast_time_available_text_guides_budgeted_residual_release_under_extreme_weather_and_ramp_regimes",
            "text2_short_branch": (
                "fuzzy_extreme_and_ramp_semantics_as_high_frequency_treatment_effect"
            ),
            "text2_long_branch": "low_frequency_trend_prompt_as_invariant_residual_treatment",
            "fusion_policy": "text2_is_counterfactual_residual_evidence;final_forecast_is_stage1_plus_budgeted_released_residual_not_direct_feature_fusion",
            "router_candidate_names": ROUTER_CANDIDATE_NAMES,
            "use_realtime_condition": args.use_realtime_condition,
            "use_fuzzy_extreme": args.use_fuzzy_extreme,
            "use_shock_evidence": args.use_shock_evidence,
            "use_low_text2": args.use_low_text2,
            "use_high_text2": args.use_high_text2,
            "use_text_router": args.use_text_router,
        },
        require_zero_initial_correction=False,
        allow_epoch0_fallback=True,
    )


def run_text2_eval(args):
    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    amp = bool(args.amp and device.type == "cuda")
    if not args.text2_ckpt:
        raise ValueError("--text2_ckpt is required for mode=eval_text2")
    ensure_dir(args.text2_out_dir)
    _require_checkpoint_matches_current_data(args.baseline_ckpt, args, "baseline")
    _require_checkpoint_matches_current_data(args.text2_ckpt, args, "text2")
    baseline, baseline_checkpoint = load_baseline_model(args.baseline_ckpt, device)
    if bool(getattr(args, "multi_dataset", False)):
        text1_cache = _build_text1_caches(args, device)
        text2_cache = _build_text2_caches(args, device)
        first_text1 = _first_cache(text1_cache)
        first_text2 = _first_cache(text2_cache)
        text1_dim = int(first_text1["state_prompt_sentence"].shape[1])
        text2_dim = int(first_text2["low_frequency_trend_prompt"].shape[1])
        text2_scalar_dim = int(
            first_text2.get("text2_scalar", np.zeros((1, TEXT2_SCALAR_DIM), dtype=np.float32)).shape[1]
        )
    else:
        _assert_safe_text2_path(args.text2_path)
        text1_cache = build_text1_cache(
            args.text1_path,
            args.text_model,
            str(device),
            args.cache_dir,
            batch_size=args.text_batch_size,
            max_tokens=args.text_max_tokens,
        )
        text2_cache = build_text2_cache(
            args.text2_path,
            args.text_model,
            str(device),
            args.cache_dir,
            batch_size=args.text_batch_size,
        )
        text1_dim = int(text1_cache["state_prompt_sentence"].shape[1])
        text2_dim = int(text2_cache["low_frequency_trend_prompt"].shape[1])
        text2_scalar_dim = int(
            text2_cache.get("text2_scalar", np.zeros((1, TEXT2_SCALAR_DIM), dtype=np.float32)).shape[1]
        )
    loaders = make_solar_loaders(args, text1_cache=text1_cache, text2_cache=text2_cache)
    _, val_loader, test_loader, _, y_scaler, _, metadata = loaders
    model, checkpoint = load_fusion_model(
        args.text2_ckpt,
        baseline,
        device,
        text1_dim_override=text1_dim,
        text2_dim_override=text2_dim,
        args_override=args,
        load_text2_modules=True,
    )
    model.use_text2_correction = True
    model.eval()
    checkpoint_metadata = checkpoint.get("metadata", {})
    skip_posthoc_text2_release = (
        bool(getattr(args, "safe_text2_fallback", True))
        and checkpoint_metadata.get("accepted_new_stage") is False
    )
    residual_calibration = None
    if not skip_posthoc_text2_release:
        residual_arrays = collect_residual_calibration_arrays(
            model,
            val_loader,
            device,
            amp,
            max_batches=int(getattr(args, "calibration_max_batches", 0) or 0),
        )
        residual_calibration = fit_residual_calibration(
            residual_arrays,
            fast=bool(getattr(args, "calibration_fast", False)),
            search_all_specs=bool(getattr(args, "calibration_search", False)),
            safe_fallback=bool(getattr(args, "safe_text2_fallback", True)),
            min_global_gain=float(getattr(args, "safe_text2_min_global_gain", 1e-4)),
            min_extreme_gain=float(getattr(args, "safe_text2_min_extreme_gain", 5e-4)),
            force_candidate_name=str(getattr(args, "force_calibration_name", "") or ""),
            force_candidate_variant=str(getattr(args, "force_calibration_variant", "") or ""),
            force_release_scale=float(getattr(args, "force_release_scale", -1.0)),
        )
    else:
        print(
            "source checkpoint is an epoch0 fallback; "
            "skipping post-hoc Text2 residual release"
        )
    if residual_calibration is not None:
        write_residual_calibration_files(args.text2_out_dir, "residual_calibration_accepted", residual_calibration)
        write_residual_calibration_files(args.text2_out_dir, "residual_calibration", residual_calibration)
    metrics, prediction, target, baseline_prediction, correction = evaluate(
        model,
        test_loader,
        device,
        y_scaler,
        amp,
        residual_calibration=residual_calibration,
        max_batches=int(getattr(args, "eval_max_batches", 0) or 0),
    )
    pd.DataFrame([metrics]).to_csv(
        os.path.join(args.text2_out_dir, "test_metrics_accepted.csv"),
        index=False,
    )
    best_accepted_path = os.path.join(args.text2_out_dir, ACCEPTED_CHECKPOINT_NAME)
    best_alias_path = os.path.join(args.text2_out_dir, "best.pt")
    if os.path.abspath(args.text2_ckpt) != os.path.abspath(best_accepted_path):
        shutil.copy2(args.text2_ckpt, best_accepted_path)
    if os.path.abspath(best_accepted_path) != os.path.abspath(best_alias_path):
        shutil.copy2(best_accepted_path, best_alias_path)
    if args.save_arrays:
        np.save(os.path.join(args.text2_out_dir, "prediction_accepted_normalized.npy"), prediction)
        np.save(os.path.join(args.text2_out_dir, "target_normalized.npy"), target)
        np.save(os.path.join(args.text2_out_dir, "baseline_prediction_normalized.npy"), baseline_prediction)
        np.save(os.path.join(args.text2_out_dir, "correction_accepted_scaled.npy"), correction)
    experiment_name = "FRTC_BER_ParetoRegimeResidualRelease"
    result = {
        "experiment": experiment_name,
        "checkpoint": best_alias_path,
        "accepted_role": "eval_text2_checkpoint",
        "accepted_new_stage": True,
        "accepted_val_real_MSE": np.nan,
        **metrics,
    }
    pd.DataFrame([result]).to_csv(os.path.join(args.text2_out_dir, "summary.csv"), index=False)
    paper_core = {
        "method_name": METHOD_NAME,
        "method_full_name": METHOD_FULL_NAME,
        "code_version": CODE_VERSION,
        "experiment": experiment_name,
        "accepted_role": "eval_text2_checkpoint",
        "accepted_new_stage": True,
        "checkpoint": best_alias_path,
    }
    for key in PAPER_CORE_METRIC_KEYS:
        if key in metrics:
            paper_core[key] = metrics[key]
    pd.DataFrame([paper_core]).to_csv(
        os.path.join(args.text2_out_dir, "paper_core_metrics.csv"),
        index=False,
    )
    with open(os.path.join(args.text2_out_dir, "paper_core_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(paper_core, handle, ensure_ascii=False, indent=2)
    metadata_out = {
        **metadata,
        "code_version": CODE_VERSION,
        "method_name": METHOD_NAME,
        "method_full_name": METHOD_FULL_NAME,
        "method_core_claim": METHOD_CORE_CLAIM,
        "method_pipeline": list(METHOD_PIPELINE),
        "experiment_name": experiment_name,
        "accepted_role": "eval_text2_checkpoint",
        "paper_checkpoint": ACCEPTED_CHECKPOINT_NAME,
        "source_text2_checkpoint": args.text2_ckpt,
        "source_checkpoint_code_version": checkpoint.get("code_version", "unknown"),
        "baseline_checkpoint": args.baseline_ckpt,
        "baseline_code_version": baseline_checkpoint.get("code_version", "unknown"),
        **_single_or_multi_path_metadata(args),
        "args": _args_metadata(args),
        "text1_dim": text1_dim,
        "text2_dim": text2_dim,
        "text2_scalar_dim": text2_scalar_dim,
        "residual_release_variant": (
            residual_calibration.get("variant", "none") if residual_calibration is not None else "none"
        ),
    }
    with open(os.path.join(args.text2_out_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata_out, handle, ensure_ascii=False, indent=2)
    print(
        f"TEST accepted MSE={metrics['real_MSE']:.6f}, "
        f"MAE={metrics['real_MAE']:.6f}, "
        f"text2_gain={metrics.get('text1_to_final_gain_MSE', 0.0):.6f}, "
        f"extreme_gain={metrics.get('extreme_union_text2_residual_improve_MSE', 0.0):.6f}, "
        f"normal_degrade={metrics.get('normal_daylight_degrade_MSE', 0.0):.6f}, "
        f"variant={metrics.get('residual_calibration_variant', 'none')}"
    )
    return result


def run_fusion(args):
    text1_result = run_text1(args)
    args.text1_ckpt = text1_result["checkpoint"]
    return run_text2(args)


def write_method_card(out_dir: str, extra: Optional[dict] = None) -> None:
    ensure_dir(out_dir)
    payload = dict(METHOD_CARD)
    if extra:
        payload.update(extra)
    with open(os.path.join(out_dir, "method_card.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def run_paper_mainline(args):
    """Paper-facing FRTC-BER mainline.

    The internal dependency order is kept for training practicality, but the
    exported result is a single method: numerical foundation plus textual
    residual intervention and Regime-Pareto release.
    """
    ensure_dir(args.out_dir)
    if _arg_was_provided("baseline_ckpt"):
        args.paper_retrain_baseline = False
    if _arg_was_provided("text1_ckpt"):
        args.paper_retrain_text1 = False
    if bool(getattr(args, "paper_retrain_baseline", False)):
        args.baseline_ckpt = None
    if bool(getattr(args, "paper_retrain_text1", False)):
        args.text1_ckpt = None
    _require_checkpoint_matches_current_data(args.baseline_ckpt, args, "baseline")
    _require_checkpoint_matches_current_data(args.text1_ckpt, args, "text1")
    if not args.baseline_ckpt and not bool(getattr(args, "paper_retrain_baseline", False)):
        baseline_candidates = [
            os.path.join(args.baseline_out_dir, ACCEPTED_CHECKPOINT_NAME),
        ]
        for candidate in baseline_candidates:
            if _checkpoint_matches_current_data(candidate, args):
                args.baseline_ckpt = candidate
                print(f"using validated numerical foundation: {candidate}")
                break
    if not args.text1_ckpt and not bool(getattr(args, "paper_retrain_text1", False)):
        text1_candidates = [
            os.path.join(args.text1_out_dir, ACCEPTED_CHECKPOINT_NAME),
        ]
        for candidate in text1_candidates:
            if _checkpoint_matches_current_data(candidate, args):
                args.text1_ckpt = candidate
                print(f"using validated realtime text intervention: {candidate}")
                break
    write_method_card(
        args.out_dir,
        {
            "execution_mode": "paper",
            "main_result_dir": args.text2_out_dir,
            "internal_numerical_foundation_dir": args.baseline_out_dir,
            "internal_text1_intervention_dir": args.text1_out_dir,
            "numerical_foundation_policy": (
                "reuse_validated_checkpoint"
                if args.baseline_ckpt and not bool(getattr(args, "paper_retrain_baseline", False))
                else "train_from_scratch"
            ),
            "realtime_text1_policy": (
                "reuse_validated_checkpoint"
                if args.text1_ckpt and not bool(getattr(args, "paper_retrain_text1", False))
                else "train_from_scratch"
            ),
        },
    )
    if not args.baseline_ckpt or not os.path.exists(args.baseline_ckpt):
        baseline_result = run_baseline(args)
        args.baseline_ckpt = baseline_result["checkpoint"]
    if not args.text1_ckpt or not os.path.exists(args.text1_ckpt):
        text1_result = run_text1(args)
        args.text1_ckpt = text1_result["checkpoint"]
    result = run_text2(args)
    final_summary = os.path.join(args.text2_out_dir, "summary.csv")
    final_core_csv = os.path.join(args.text2_out_dir, "paper_core_metrics.csv")
    final_core_json = os.path.join(args.text2_out_dir, "paper_core_metrics.json")
    if os.path.exists(final_summary):
        shutil.copy2(final_summary, os.path.join(args.out_dir, "paper_main_summary.csv"))
    if os.path.exists(final_core_csv):
        shutil.copy2(final_core_csv, os.path.join(args.out_dir, "paper_core_metrics.csv"))
    if os.path.exists(final_core_json):
        shutil.copy2(final_core_json, os.path.join(args.out_dir, "paper_core_metrics.json"))
    write_method_card(
        args.out_dir,
        {
            "execution_mode": "paper",
            "main_result_dir": args.text2_out_dir,
            "main_checkpoint": result.get("checkpoint"),
            "internal_numerical_foundation_dir": args.baseline_out_dir,
            "internal_text1_intervention_dir": args.text1_out_dir,
            "result": {
                key: result[key]
                for key in (
                    "real_MSE",
                    "real_MAE",
                    "stage1_real_MSE",
                    "stage1_text_real_MSE",
                    "numeric_to_text1_gain_MSE",
                    "text1_to_final_gain_MSE",
                    "extreme_union_text2_residual_improve_MSE",
                    "normal_daylight_degrade_MSE",
                    "residual_calibration_variant",
                )
                if key in result
            },
        },
    )
    return result


def run(args):
    ensure_dir(args.out_dir)
    if getattr(args, "mode", None) == "build_text2":
        return build_residual_oriented_text2(args)
    if getattr(args, "mode", None) == "eval_text2":
        return run_text2_eval(args)
    if getattr(args, "mode", None) == "paper":
        return run_paper_mainline(args)
    if args.mode in {"baseline", "full"}:
        baseline_result = run_baseline(args)
        if args.mode == "baseline":
            return baseline_result
        args.baseline_ckpt = baseline_result["checkpoint"]
    if args.mode == "text1":
        return run_text1(args)
    if args.mode == "text2":
        return run_text2(args)
    if args.mode in {"fusion", "full"}:
        return run_fusion(args)
    raise ValueError(f"unknown mode: {args.mode}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["paper", "baseline", "text1", "text2", "eval_text2", "fusion", "full", "build_text2"],
        default="paper",
    )
    parser.add_argument("--dataset_root", default="./datasets")
    parser.add_argument("--dataset_name", default="kongzhaopu")
    parser.add_argument(
        "--datasets",
        default=None,
        help="Comma-separated dataset names under dataset_root. Example: kongzhaopu,adong,aping",
    )
    parser.add_argument("--solar_csv", default=None)
    parser.add_argument("--solar_aux_csv", default=None)
    parser.add_argument("--solar_nwp_csv", default=None)
    parser.add_argument("--context_csv", default=None)
    parser.add_argument("--text1_path", default=None)
    parser.add_argument("--text2_path", default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--build_neutral_text2", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--text_model", default="./encoder")
    parser.add_argument("--cache_dir", default="./cache/text")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--baseline_out_dir", default=None)
    parser.add_argument("--fusion_out_dir", default=None)
    parser.add_argument("--text1_out_dir", default=None)
    parser.add_argument("--text2_out_dir", default=None)
    parser.add_argument("--baseline_ckpt", default=None)
    parser.add_argument("--text1_ckpt", default=None)
    parser.add_argument("--text2_ckpt", default=None)

    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--pred_len", type=int, default=45)
    parser.add_argument("--prediction_start_hour", type=float, default=8.0)
    parser.add_argument("--prediction_end_hour", type=float, default=19.0)
    parser.add_argument("--prediction_include_end", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--init_tolerance", type=float, default=1e-7)
    parser.add_argument("--acceptance_tolerance", type=float, default=0.0)
    parser.add_argument("--use_weight_ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ema_decay", type=float, default=0.98)
    parser.add_argument("--ema_start_epoch", type=int, default=2)
    parser.add_argument("--use_weight_swa", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--swa_start_epoch", type=int, default=0)

    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--d_layers", type=int, default=1)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--ma_kernel", type=int, default=25)
    parser.add_argument("--text_layers", type=int, default=3)
    parser.add_argument("--nwp_layers", type=int, default=2)
    parser.add_argument("--experts", type=int, default=6)
    parser.add_argument("--basis_rank", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--cutoff_index", type=float, default=8.0)
    parser.add_argument("--max_delta", type=float, default=0.20)
    parser.add_argument("--text2_max_delta", type=float, default=0.20)
    parser.add_argument("--baseline_max_delta", type=float, default=0.25)
    parser.add_argument("--risk_horizon_hours", type=float, default=3.0)
    parser.add_argument("--realtime_slots_per_field", type=int, default=4)

    parser.add_argument("--daytime_weight", type=float, default=1.5)
    parser.add_argument("--w_numeric_event", type=float, default=0.35)
    parser.add_argument("--w_realtime_residual", type=float, default=0.16)
    parser.add_argument("--w_residual", type=float, default=0.18)
    parser.add_argument("--w_history_prior", type=float, default=0.08)
    parser.add_argument("--w_history_trust", type=float, default=0.01)
    parser.add_argument("--w_nwp_prior", type=float, default=0.25)
    parser.add_argument("--w_nwp_trust", type=float, default=0.02)
    parser.add_argument("--w_periodic_prior", type=float, default=0.18)
    parser.add_argument("--w_periodic_trust", type=float, default=0.02)
    parser.add_argument("--w_candidate_trust_sparse", type=float, default=0.04)
    parser.add_argument("--w_branch", type=float, default=0.10)
    parser.add_argument("--w_event", type=float, default=0.26)
    parser.add_argument("--w_shock", type=float, default=0.10)
    parser.add_argument("--w_short_residual", type=float, default=0.11)
    parser.add_argument("--w_risk_sparse", type=float, default=0.01)
    parser.add_argument("--w_risk_tv", type=float, default=0.002)
    parser.add_argument("--w_prototype_entropy", type=float, default=0.0005)
    parser.add_argument("--w_prototype_balance", type=float, default=0.001)
    parser.add_argument("--w_confidence_sparse", type=float, default=0.01)
    parser.add_argument("--w_fuzzy_ambiguity", type=float, default=0.002)
    parser.add_argument("--w_non_degradation", type=float, default=3.0)
    parser.add_argument("--w_residual_alignment", type=float, default=0.03)
    parser.add_argument("--w_opportunity", type=float, default=0.04)
    parser.add_argument("--w_selective_energy", type=float, default=0.05)
    parser.add_argument("--w_direction", type=float, default=0.08)
    parser.add_argument("--w_stage_scale", type=float, default=0.002)
    parser.add_argument("--w_energy", type=float, default=0.01)
    parser.add_argument("--w_smooth", type=float, default=0.01)
    parser.add_argument("--w_orthogonal", type=float, default=0.05)
    parser.add_argument("--w_neutral", type=float, default=0.05)
    parser.add_argument("--w_transport_cost", type=float, default=0.005)
    parser.add_argument("--w_transport_entropy", type=float, default=0.001)
    parser.add_argument("--w_async_lag", type=float, default=0.001)
    parser.add_argument("--w_async_entropy", type=float, default=0.0005)
    parser.add_argument("--w_counterfactual", type=float, default=0.10)
    parser.add_argument("--w_text_contrast", type=float, default=0.08)
    parser.add_argument("--w_calibration", type=float, default=0.10)
    parser.add_argument("--w_router", type=float, default=0.08)
    parser.add_argument("--w_router_distill", type=float, default=0.0)
    parser.add_argument("--w_router_safe", type=float, default=0.60)
    parser.add_argument("--w_router_gain", type=float, default=0.04)
    parser.add_argument("--w_router_entropy", type=float, default=0.002)
    parser.add_argument("--improvement_margin", type=float, default=0.0)

    parser.add_argument("--freeze_baseline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--numeric_foundation_profile",
        choices=["auto", "compact", "robust"],
        default="auto",
        help=(
            "Numerical backbone profile. auto uses the dataset-agnostic robust "
            "history/NWP/periodic prior configuration."
        ),
    )
    parser.add_argument("--use_history_backbone", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_nwp_prior", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_periodic_prior", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_numerical_candidate_pool", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_dataset_conditioning", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--balanced_dataset_sampling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--event_balanced_sampling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--event_sample_quantile", type=float, default=0.70)
    parser.add_argument("--event_sample_alpha", type=float, default=1.50)
    parser.add_argument("--event_sample_max_weight", type=float, default=4.00)
    parser.add_argument("--use_aux_residual", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_past_nwp_residual", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_future_nwp_residual", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--aux_gate_bias", type=float, default=-2.8)
    parser.add_argument("--past_nwp_gate_bias", type=float, default=-3.6)
    parser.add_argument("--future_nwp_gate_bias", type=float, default=-3.4)
    parser.add_argument("--aux_gate_max", type=float, default=0.55)
    parser.add_argument("--past_nwp_gate_max", type=float, default=0.45)
    parser.add_argument("--future_nwp_gate_max", type=float, default=0.55)
    parser.add_argument("--use_realtime_condition", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_fuzzy_extreme", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_shock_evidence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_low_text2", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_high_text2", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_text_router", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze_text1_for_text2", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze_text2_low_for_high", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reuse_text2_modules", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--allow_text2_initial_correction", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--text_batch_size", type=int, default=16)
    parser.add_argument("--text_max_tokens", type=int, default=48)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_arrays", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--calibration_fast", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--calibration_search", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--calibration_max_batches", type=int, default=24)
    parser.add_argument("--eval_max_batches", type=int, default=0)
    parser.add_argument("--safe_text2_fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--safe_text2_min_global_gain", type=float, default=1e-4)
    parser.add_argument("--safe_text2_min_extreme_gain", type=float, default=5e-4)
    parser.add_argument("--force_calibration_name", default="")
    parser.add_argument("--force_calibration_variant", default="")
    parser.add_argument("--force_release_scale", type=float, default=-1.0)
    parser.add_argument("--paper_retrain_baseline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--paper_retrain_text1", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    _apply_dataset_defaults(args)
    if args.prediction_start_hour is not None and args.prediction_end_hour is not None:
        expected_steps = int(
            round(
                (float(args.prediction_end_hour) - float(args.prediction_start_hour))
                * 60.0
                / 15.0
            )
        )
        if bool(args.prediction_include_end):
            expected_steps += 1
        if int(args.pred_len) != expected_steps:
            raise ValueError(
                "pred_len does not match the configured prediction window: "
                f"pred_len={args.pred_len}, expected={expected_steps} for "
                f"[{args.prediction_start_hour}, {args.prediction_end_hour}"
                f"{']' if args.prediction_include_end else ')'} at 15-minute resolution"
            )
    _apply_numeric_foundation_profile(args)
    if args.out_dir is None:
        default_result_name = "multi_dataset" if args.multi_dataset else str(args.dataset_names[0])
        args.out_dir = os.path.join("./results", "main_results", default_result_name)
    ensure_dir(args.out_dir)
    if args.baseline_out_dir is None:
        args.baseline_out_dir = os.path.join(args.out_dir, "00_numeric_baseline")
    if args.fusion_out_dir is None:
        args.fusion_out_dir = os.path.join(args.out_dir, "02_text2_soft_pareto_best")
    if args.text1_out_dir is None:
        args.text1_out_dir = os.path.join(args.out_dir, "01_realtime_text1")
    if args.text2_out_dir is None:
        args.text2_out_dir = os.path.join(args.out_dir, "02_text2_soft_pareto_best")
    if args.mode == "paper" and bool(args.multi_dataset):
        args.paper_retrain_baseline = True
        args.paper_retrain_text1 = True
    if args.mode in {"text1", "text2", "eval_text2", "fusion"} and not args.baseline_ckpt:
        preferred = os.path.join(args.baseline_out_dir, ACCEPTED_CHECKPOINT_NAME)
        args.baseline_ckpt = preferred if os.path.exists(preferred) else os.path.join(
            args.baseline_out_dir, ACCEPTED_CHECKPOINT_NAME
        )
    if args.mode == "text2" and not args.text1_ckpt:
        args.text1_ckpt = os.path.join(args.text1_out_dir, ACCEPTED_CHECKPOINT_NAME)
    run(args)


if __name__ == "__main__":
    main()
