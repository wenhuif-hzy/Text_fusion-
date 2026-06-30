"""Text prompt generation rules for GuangFu photovoltaic forecasting.

This file only contains the text-generation logic used for rt_text1.jsonl and
rt_text2.jsonl. It assumes the input dataframe already follows the standard
solar.csv schema:

date,temp,pressure,wind_speed,wind_dir,sun_radiation,sca_radiation,
nwp_tmp,nwp_rh,nwp_surfacepres,nwp_windspeed,nwp_winddir,
nwp_shortwaveirrad,nwp_scatterirrad,nwp_directirrad,capacity,OT

Design:
- text1: real-time description, aligned with the provided Kongzhaopu example.
  It uses current/history station state, observed radiation, NWP background,
  and recent variability.
- text2: future description, strictly generated from future NWP only.
  It is split into low-frequency trend and high-frequency disturbance prompts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def prompt_time_range(df: pd.DataFrame) -> pd.DatetimeIndex:
    start = pd.to_datetime(df["date"]).min().floor("h")
    end = pd.to_datetime(df["date"]).max().floor("h")
    return pd.date_range(start=start, end=end, freq="1h")


def _mean(frame: pd.DataFrame, column: str, default: float = 0.0) -> float:
    if frame.empty or column not in frame:
        return default
    value = pd.to_numeric(frame[column], errors="coerce").mean()
    return default if not np.isfinite(value) else float(value)


def _std(frame: pd.DataFrame, column: str, default: float = 0.0) -> float:
    if frame.empty or column not in frame:
        return default
    value = pd.to_numeric(frame[column], errors="coerce").std()
    return default if not np.isfinite(value) else float(value)


def _trend_phrase(delta: float, scale: float, subject: str) -> str:
    threshold = max(scale * 0.05, 1e-6)
    if delta > threshold:
        return f"{subject} has increased over the recent hour."
    if delta < -threshold:
        return f"{subject} has decreased over the recent hour."
    return f"{subject} has stayed broadly steady over the recent hour."


def _wind_phrase(wind_speed: float) -> str:
    if wind_speed < 1.5:
        return "wind is light."
    if wind_speed < 4.0:
        return "wind is moderate."
    return "wind is strong."


def make_text1_prompts(df: pd.DataFrame) -> list[dict[str, str]]:
    """Generate real-time text1 prompts.

    text1 describes the current and recent station state. It is allowed to use
    current/history OT and observed radiation because these are already
    available at forecast time. It does not use future OT.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    y_scale = max(float(df["OT"].quantile(0.95)), 1e-6)
    obs_rad_scale = max(float(df["sun_radiation"].quantile(0.95)), 1e-6)
    nwp_rad_scale = max(float(df["nwp_shortwaveirrad"].quantile(0.95)), 1e-6)

    prompts: list[dict[str, str]] = []
    for timestamp in prompt_time_range(df):
        recent = df[(df["date"] > timestamp - pd.Timedelta(hours=1)) & (df["date"] <= timestamp)]
        previous = df[
            (df["date"] > timestamp - pd.Timedelta(hours=2))
            & (df["date"] <= timestamp - pd.Timedelta(hours=1))
        ]

        output = _mean(recent, "OT")
        observed_radiation = _mean(recent, "sun_radiation")
        nwp_shortwave = _mean(recent, "nwp_shortwaveirrad")
        wind = _mean(recent, "wind_speed")

        output_delta = output - _mean(previous, "OT", output)
        observed_radiation_delta = observed_radiation - _mean(
            previous, "sun_radiation", observed_radiation
        )
        nwp_shortwave_delta = nwp_shortwave - _mean(
            previous, "nwp_shortwaveirrad", nwp_shortwave
        )
        variability = _std(recent, "OT")

        if observed_radiation < 0.03 * obs_rad_scale and output < 0.03 * y_scale:
            state = (
                "The station is under nighttime or very weak radiation conditions, "
                "and photovoltaic output remains near zero."
            )
        elif output < 0.25 * y_scale:
            state = "The station is in a low-output photovoltaic regime with limited effective radiation."
        elif output < 0.65 * y_scale:
            state = "The station is in a moderate photovoltaic output regime with usable radiation."
        else:
            state = "The station is in a high-output photovoltaic regime under strong radiation support."
        state = f"{state} {_wind_phrase(wind)}"

        trend = " ".join(
            [
                _trend_phrase(output_delta, y_scale, "Output"),
                _trend_phrase(observed_radiation_delta, obs_rad_scale, "Observed radiation"),
                _trend_phrase(nwp_shortwave_delta, nwp_rad_scale, "The NWP shortwave background"),
            ]
        )

        if variability < 0.03 * y_scale:
            variability_text = "Recent variability is low, and the series remains locally smooth."
        elif variability < 0.12 * y_scale:
            variability_text = "Recent variability is moderate, with visible but controlled fluctuation."
        else:
            variability_text = "Recent variability is high, suggesting rapid cloud-driven power changes."

        prompts.append(
            {
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "state_prompt": state,
                "recent_trend_prompt": trend,
                "statistical_variability_prompt": variability_text,
            }
        )
    return prompts


def make_text2_prompts(df: pd.DataFrame) -> list[dict[str, str]]:
    """Generate future text2 prompts.

    text2 is strictly generated from future NWP fields. It does not use future
    OT/power or future observed radiation. It is split into low-frequency trend
    and high-frequency disturbance descriptions.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    rad_scale = max(float(df["nwp_shortwaveirrad"].quantile(0.95)), 1e-6)
    prompts: list[dict[str, str]] = []
    for timestamp in prompt_time_range(df):
        horizon = df[
            (df["date"] >= timestamp)
            & (df["date"] < timestamp + pd.Timedelta(hours=24))
        ]
        if horizon.empty:
            horizon = df[df["date"] >= timestamp]

        nwp = pd.to_numeric(horizon["nwp_shortwaveirrad"], errors="coerce").fillna(0.0)
        direct = pd.to_numeric(horizon["nwp_directirrad"], errors="coerce").fillna(0.0)
        diffuse = pd.to_numeric(horizon["nwp_scatterirrad"], errors="coerce").fillna(0.0)

        first = float(nwp.iloc[:4].mean()) if len(nwp) else 0.0
        last = float(nwp.iloc[-4:].mean()) if len(nwp) >= 4 else first
        mean_nwp = float(nwp.mean()) if len(nwp) else 0.0
        variability = float(nwp.diff().abs().quantile(0.9)) if len(nwp) > 1 else 0.0
        direct_ratio = float(direct.mean() / max(direct.mean() + diffuse.mean(), 1e-6))

        if mean_nwp < 0.03 * rad_scale:
            low = "The low-frequency background remains near a nighttime baseline through the coming horizon."
        elif last > first + 0.10 * rad_scale:
            low = "The low-frequency background is expected to strengthen gradually through the coming horizon."
        elif last < first - 0.10 * rad_scale:
            low = "The low-frequency background is expected to weaken gradually through the coming horizon."
        else:
            low = "The low-frequency background remains broadly stable through the coming horizon."

        if mean_nwp >= 0.03 * rad_scale:
            if direct_ratio > 0.65:
                low += " The radiation regime leans toward a direct-beam-dominant background."
            elif direct_ratio < 0.35:
                low += " The radiation regime leans toward a diffuse-cloudy background."
            else:
                low += " The radiation regime leans toward a mixed direct and diffuse background."

        if variability < 0.03 * rad_scale:
            high = "The high-frequency component is very weak, and obvious mutation-like jumps are unlikely."
        elif variability < 0.12 * rad_scale:
            high = "The high-frequency component remains limited, with only mild short-lived fluctuation expected."
        else:
            high = (
                "The high-frequency component is active, and sharp cloud-edge-like jumps "
                "or rapid reversals are likely."
            )

        prompts.append(
            {
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "low_frequency_trend_prompt": low,
                "high_frequency_component_prompt": high,
            }
        )
    return prompts


def write_jsonl(records: list[dict[str, str]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

