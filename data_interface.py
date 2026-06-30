# -*- coding: utf-8 -*-
"""
Unified, leakage-safe data interface for Exp1--Exp7.

Important properties
--------------------
* Chronological 7/1/2 split with targets strictly confined to each split.
* Text remains hourly and is returned with timestamps and validity masks.
* Missing text is masked instead of clipping to the first/last available row.
* Circular wind direction is encoded as sine/cosine.
* Text embedding caches are fingerprinted by source file and encoder path.
* No random-seed manipulation.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler


TARGET = "OT"

STATION_BASE_AUX_COLS = [
    "lmd_totalirrad",
    "lmd_diffuseirrad",
    "lmd_temperature",
    "lmd_pressure",
    "lmd_windspeed",
]
STATION_AUX_COLS = STATION_BASE_AUX_COLS + [
    "lmd_winddirection_sin",
    "lmd_winddirection_cos",
]

SOLAR_BASE_AUX_COLS = [
    "temp",
    "pressure",
    "wind_speed",
    "sun_radiation",
    "sca_radiation",
    "capacity",
]
SOLAR_AUX_COLS = SOLAR_BASE_AUX_COLS + ["wind_dir_sin", "wind_dir_cos"]

SOLAR_BASE_NWP_COLS = [
    "nwp_tmp",
    "nwp_rh",
    "nwp_surfacepres",
    "nwp_windspeed",
    "nwp_shortwaveirrad",
    "nwp_scatterirrad",
    "nwp_directirrad",
]
SOLAR_NWP_COLS = SOLAR_BASE_NWP_COLS + ["nwp_winddir_sin", "nwp_winddir_cos"]

TEXT1_FIELDS = ["state_prompt", "recent_trend_prompt", "statistical_variability_prompt"]
TEXT2_FIELDS = ["low_frequency_trend_prompt", "high_frequency_component_prompt"]
TEXT2_SCALAR_DIM = 11


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def parse_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def add_circular_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mapping = {
        "wind_dir": "wind_dir",
        "nwp_winddir": "nwp_winddir",
        "lmd_winddirection": "lmd_winddirection",
    }
    for source, prefix in mapping.items():
        if source in df.columns:
            radians = np.deg2rad(pd.to_numeric(df[source], errors="coerce"))
            df[f"{prefix}_sin"] = np.sin(radians)
            df[f"{prefix}_cos"] = np.cos(radians)
    return df


def add_time_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    t = parse_datetime(df[date_col])
    hour = t.dt.hour + t.dt.minute / 60.0
    dow = t.dt.dayofweek.astype(float)
    doy = t.dt.dayofyear.astype(float)
    out = pd.DataFrame(index=df.index)
    out["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    out["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)
    out["doy_sin"] = np.sin(2.0 * np.pi * doy / 366.0)
    out["doy_cos"] = np.cos(2.0 * np.pi * doy / 366.0)
    return out.astype(np.float32)


class StandardScaler:
    """NaN-safe feature scaler fitted on the training partition only."""

    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> "StandardScaler":
        x = np.asarray(x, dtype=np.float64)
        self.mean = np.nanmean(x, axis=0, keepdims=True)
        self.std = np.nanstd(x, axis=0, keepdims=True)
        self.mean = np.where(np.isfinite(self.mean), self.mean, 0.0)
        self.std = np.where(np.isfinite(self.std) & (self.std >= 1e-6), self.std, 1.0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("scaler has not been fitted")
        x = np.asarray(x, dtype=np.float32)
        x = np.where(np.isfinite(x), x, self.mean.astype(np.float32))
        return ((x - self.mean) / self.std).astype(np.float32)

    def inverse_target(self, y: np.ndarray, target_index: int = 0) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("scaler has not been fitted")
        return y * self.std[0, target_index] + self.mean[0, target_index]

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {"mean": self.mean, "std": self.std}


class IndexedTargetScaler(StandardScaler):
    """Target scaler with one scale per dataset id.

    The transform/inverse methods inherited from StandardScaler keep the object
    compatible with single-scale call sites. Multi-dataset datasets use
    transform_with_index and inverse_target_with_index to avoid one large station
    dominating the target scale.
    """

    def __init__(self):
        super().__init__()
        self.dataset_names: List[str] = []

    def fit_by_frame(
        self,
        frames: Sequence[pd.DataFrame],
        column: str,
        train_ends: Sequence[int],
        dataset_names: Sequence[str],
    ) -> "IndexedTargetScaler":
        means, stds = [], []
        for frame, train_end in zip(frames, train_ends):
            values = frame.iloc[: int(train_end)][[column]].to_numpy(np.float64)
            mean = np.nanmean(values, axis=0)
            std = np.nanstd(values, axis=0)
            mean = np.where(np.isfinite(mean), mean, 0.0)
            std = np.where(np.isfinite(std) & (std >= 1e-6), std, 1.0)
            means.append(mean)
            stds.append(std)
        if not means:
            raise ValueError("cannot fit indexed target scaler from an empty dataset list")
        self.mean = np.asarray(means, dtype=np.float64)
        self.std = np.asarray(stds, dtype=np.float64)
        self.dataset_names = [str(name) for name in dataset_names]
        return self

    def fit_by_arrays(
        self,
        arrays: Sequence[np.ndarray],
        dataset_names: Sequence[str],
    ) -> "IndexedTargetScaler":
        means, stds = [], []
        for values in arrays:
            values = np.asarray(values, dtype=np.float64).reshape(-1, 1)
            mean = np.nanmean(values, axis=0)
            std = np.nanstd(values, axis=0)
            mean = np.where(np.isfinite(mean), mean, 0.0)
            std = np.where(np.isfinite(std) & (std >= 1e-6), std, 1.0)
            means.append(mean)
            stds.append(std)
        if not means:
            raise ValueError("cannot fit indexed target scaler from an empty array list")
        self.mean = np.asarray(means, dtype=np.float64)
        self.std = np.asarray(stds, dtype=np.float64)
        self.dataset_names = [str(name) for name in dataset_names]
        return self

    def transform_with_index(self, y: np.ndarray, dataset_index: int) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("scaler has not been fitted")
        y = np.asarray(y, dtype=np.float32)
        idx = int(dataset_index)
        mean = self.mean[idx].astype(np.float32)
        std = self.std[idx].astype(np.float32)
        y = np.where(np.isfinite(y), y, mean)
        return ((y - mean) / std).astype(np.float32)

    def inverse_target_with_index(
        self,
        y: np.ndarray,
        dataset_index: np.ndarray,
        target_index: int = 0,
    ) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("scaler has not been fitted")
        y = np.asarray(y, dtype=np.float32)
        idx = np.asarray(dataset_index, dtype=np.int64).reshape(-1)
        mean = self.mean[idx, target_index].reshape((-1,) + (1,) * (y.ndim - 1))
        std = self.std[idx, target_index].reshape((-1,) + (1,) * (y.ndim - 1))
        return y * std + mean

    def state_dict(self) -> Dict[str, np.ndarray]:
        payload = super().state_dict()
        payload["dataset_names"] = np.asarray(self.dataset_names)
        return payload


TARGET_VALUE_COL = "__target_value"
TARGET_AVAILABLE_COL = "__target_available"


def load_numeric_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError(f"{path} is missing the 'date' column")
    df["date"] = parse_datetime(df["date"])
    df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
    return add_circular_features(df.reset_index(drop=True))


def load_and_merge_numeric(
    primary_csv: str,
    aux_csv: Optional[str] = None,
    nwp_csv: Optional[str] = None,
) -> pd.DataFrame:
    df = load_numeric_csv(primary_csv)
    for path in [aux_csv, nwp_csv]:
        if path:
            other = load_numeric_csv(path)
            new_cols = [c for c in other.columns if c == "date" or c not in df.columns]
            df = df.merge(other[new_cols], on="date", how="inner", validate="one_to_one")
    return add_circular_features(df.sort_values("date").reset_index(drop=True))


def load_context_target_numeric(
    primary_csv: str,
    aux_csv: Optional[str] = None,
    nwp_csv: Optional[str] = None,
    context_csv: Optional[str] = None,
    target_col: str = TARGET,
) -> pd.DataFrame:
    """Load full context rows while keeping the supervised target separate."""
    if context_csv:
        df = load_and_merge_numeric(context_csv, aux_csv, nwp_csv)
        target_df = load_numeric_csv(primary_csv)
        validate_columns(target_df, [target_col], primary_csv)
        target_df = target_df[["date", target_col]].rename(columns={target_col: TARGET_VALUE_COL})
        df = df.merge(target_df, on="date", how="left", validate="one_to_one")
        df[TARGET_AVAILABLE_COL] = np.isfinite(pd.to_numeric(df[TARGET_VALUE_COL], errors="coerce"))
        return add_circular_features(df.sort_values("date").reset_index(drop=True))
    df = load_and_merge_numeric(primary_csv, aux_csv, nwp_csv)
    df[TARGET_VALUE_COL] = pd.to_numeric(df[target_col], errors="coerce")
    df[TARGET_AVAILABLE_COL] = np.isfinite(df[TARGET_VALUE_COL])
    return df


def _fit_scaler_from_frames(frames: Sequence[pd.DataFrame], columns: Sequence[str], train_ends: Sequence[int]) -> StandardScaler:
    values = [
        frame.iloc[: int(train_end)][list(columns)].to_numpy(np.float32)
        for frame, train_end in zip(frames, train_ends)
    ]
    if not values:
        raise ValueError("cannot fit scaler from an empty dataset list")
    return StandardScaler().fit(np.concatenate(values, axis=0))


def validate_columns(df: pd.DataFrame, columns: Sequence[str], source: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def split_points(n: int, train_ratio: float = 0.7, val_ratio: float = 0.1) -> Tuple[int, int]:
    if not (0.0 < train_ratio < 1.0 and 0.0 < val_ratio < 1.0):
        raise ValueError("invalid split ratios")
    n_train = int(n * train_ratio)
    n_val_end = int(n * (train_ratio + val_ratio))
    return n_train, n_val_end


def _window_hour_mask(
    times: pd.DatetimeIndex,
    start_hour: Optional[float],
    end_hour: Optional[float],
    include_end: bool,
) -> np.ndarray:
    if start_hour is None or end_hour is None:
        return np.ones(len(times), dtype=bool)
    hours = (
        times.hour.to_numpy(np.float32)
        + times.minute.to_numpy(np.float32) / 60.0
        + times.second.to_numpy(np.float32) / 3600.0
    )
    if include_end:
        return (hours >= float(start_hour) - 1e-6) & (hours <= float(end_hour) + 1e-6)
    return (hours >= float(start_hour) - 1e-6) & (hours < float(end_hour) - 1e-6)


def _target_values_for_scaler(
    frame: pd.DataFrame,
    train_end: int,
    start_hour: Optional[float],
    end_hour: Optional[float],
    include_end: bool,
) -> np.ndarray:
    train = frame.iloc[: int(train_end)].copy()
    values = pd.to_numeric(train[TARGET_VALUE_COL], errors="coerce").to_numpy(np.float64)
    available = np.isfinite(values)
    if TARGET_AVAILABLE_COL in train.columns:
        available &= train[TARGET_AVAILABLE_COL].to_numpy(bool)
    window = _window_hour_mask(
        pd.DatetimeIndex(train["date"]),
        start_hour,
        end_hour,
        include_end,
    )
    selected = values[available & window]
    if selected.size == 0:
        selected = values[available]
    return selected.reshape(-1, 1)


def load_json_or_jsonl(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    rows: List[dict] = []
    if path.lower().endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    else:
        with open(path, "r", encoding="utf-8") as handle:
            obj = json.load(handle)
        rows = obj if isinstance(obj, list) else obj.get("data", [])
    df = pd.DataFrame(rows)
    if "timestamp" not in df.columns:
        if "date" not in df.columns:
            raise ValueError(f"{path} is missing the 'timestamp' field")
        df["timestamp"] = df["date"]
    df["timestamp"] = parse_datetime(df["timestamp"])
    return (
        df.dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )


class FrozenBGEEncoder:
    """Local BGE encoder used to build sentence- and token-level caches.

    Stage-1 Exp6 uses the full token hidden states.  Special tokens are masked
    from word-level cross-attention while the first-token BGE sentence vector is
    retained for Stage-2 and global diagnostics.
    """

    def __init__(self, model_path: str, device: str, max_length: int = 192):
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers is required to build text caches. Install it or reuse an existing cache."
            ) from exc
        self.device = torch.device(device)
        self.max_length = int(max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode_sentences(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        outputs: List[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            tokens = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            hidden = self.model(**tokens).last_hidden_state
            pooled = F.normalize(hidden[:, 0], p=2, dim=-1)
            outputs.append(pooled.cpu().numpy().astype(np.float32))
        if not outputs:
            raise ValueError("cannot encode an empty text collection")
        return np.concatenate(outputs, axis=0)

    @torch.inference_mode()
    def encode_tokens(
        self,
        texts: Sequence[str],
        batch_size: int = 16,
        max_tokens: int = 48,
        storage_dtype: str = "float16",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        token_chunks: List[np.ndarray] = []
        mask_chunks: List[np.ndarray] = []
        sentence_chunks: List[np.ndarray] = []
        target_dtype = np.float16 if storage_dtype == "float16" else np.float32
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            encoded = self.tokenizer(
                batch,
                padding="max_length",
                truncation=True,
                max_length=int(max_tokens),
                return_special_tokens_mask=True,
                return_tensors="pt",
            )
            special = encoded.pop("special_tokens_mask")
            encoded = encoded.to(self.device)
            hidden = self.model(**encoded).last_hidden_state
            attention_mask = encoded["attention_mask"].bool()
            lexical_mask = attention_mask & (~special.to(self.device).bool())
            nonempty_text = torch.tensor(
                [bool(str(text).strip()) for text in batch],
                dtype=torch.bool,
                device=self.device,
            )
            lexical_mask = lexical_mask & nonempty_text.unsqueeze(1)
            # A non-empty string can occasionally be reduced to special tokens
            # only. Keep one token in that rare case, but preserve genuinely
            # empty strings as an all-false modality mask.
            fallback_needed = nonempty_text & (~lexical_mask.any(dim=1))
            if fallback_needed.any():
                fallback = attention_mask.float().argmax(dim=1)
                lexical_mask[fallback_needed, fallback[fallback_needed]] = True
            sentence = F.normalize(hidden[:, 0], p=2, dim=-1)
            token_chunks.append(hidden.cpu().numpy().astype(target_dtype))
            mask_chunks.append(lexical_mask.cpu().numpy().astype(np.uint8))
            sentence_chunks.append(sentence.cpu().numpy().astype(np.float32))
        if not token_chunks:
            raise ValueError("cannot encode an empty text collection")
        return (
            np.concatenate(token_chunks, axis=0),
            np.concatenate(mask_chunks, axis=0),
            np.concatenate(sentence_chunks, axis=0),
        )

def _cache_fingerprint(source_path: str, model_path: str, fields: Sequence[str], version: str) -> str:
    stat = os.stat(source_path)
    payload = "|".join(
        [
            os.path.abspath(source_path),
            str(stat.st_size),
            str(stat.st_mtime_ns),
            os.path.abspath(model_path),
            ",".join(fields),
            version,
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_sentence_text_cache(
    text_path: str,
    model_path: str,
    device: str,
    cache_dir: str,
    fields: Sequence[str],
    cache_prefix: str,
    batch_size: int = 64,
) -> Dict[str, np.ndarray]:
    ensure_dir(cache_dir)
    fingerprint = _cache_fingerprint(text_path, model_path, fields, "main-v6-sentence")
    cache_path = os.path.join(cache_dir, f"{cache_prefix}_{fingerprint}.npz")
    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=False)
        return {key: data[key] for key in data.files}

    df = load_json_or_jsonl(text_path)
    encoder = FrozenBGEEncoder(model_path, device)
    arrays: Dict[str, np.ndarray] = {
        "timestamp_ns": df["timestamp"].astype("int64").to_numpy(dtype=np.int64)
    }
    for field in fields:
        values = df[field].fillna("").astype(str).tolist() if field in df.columns else [""] * len(df)
        arrays[field] = encoder.encode_sentences(values, batch_size=batch_size)
    np.savez_compressed(cache_path, **arrays)
    return arrays


def build_token_text1_cache(
    text_path: str,
    model_path: str,
    device: str,
    cache_dir: str,
    batch_size: int = 16,
    max_tokens: int = 48,
    storage_dtype: str = "float16",
) -> Dict[str, np.ndarray]:
    """Build full word-token BGE caches for the three real-time text fields."""
    ensure_dir(cache_dir)
    version = f"main-v6-token-max{int(max_tokens)}-{storage_dtype}"
    fingerprint = _cache_fingerprint(text_path, model_path, TEXT1_FIELDS, version)
    cache_path = os.path.join(cache_dir, f"text1_token_{fingerprint}.npz")
    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=False)
        return {key: data[key] for key in data.files}

    df = load_json_or_jsonl(text_path)
    encoder = FrozenBGEEncoder(model_path, device, max_length=max_tokens)
    arrays: Dict[str, np.ndarray] = {
        "timestamp_ns": df["timestamp"].astype("int64").to_numpy(dtype=np.int64),
        "max_tokens": np.asarray([int(max_tokens)], dtype=np.int32),
    }
    for field in TEXT1_FIELDS:
        values = df[field].fillna("").astype(str).tolist() if field in df.columns else [""] * len(df)
        tokens, mask, sentence = encoder.encode_tokens(
            values,
            batch_size=batch_size,
            max_tokens=max_tokens,
            storage_dtype=storage_dtype,
        )
        arrays[f"{field}_tokens"] = tokens
        arrays[f"{field}_token_mask"] = mask
        arrays[f"{field}_sentence"] = sentence
    np.savez_compressed(cache_path, **arrays)
    return arrays


def build_text1_cache(
    text_path: str,
    model_path: str,
    device: str,
    cache_dir: str,
    batch_size: int = 16,
    max_tokens: int = 48,
) -> Dict[str, np.ndarray]:
    return build_token_text1_cache(
        text_path,
        model_path,
        device,
        cache_dir,
        batch_size=batch_size,
        max_tokens=max_tokens,
        storage_dtype="float16",
    )


def build_text2_cache(
    text_path: str,
    model_path: str,
    device: str,
    cache_dir: str,
    batch_size: int = 64,
) -> Dict[str, np.ndarray]:
    """Build text2 cache.

    Paper runs should use forecast-time available text2, such as NWP-derived
    horizon prompts or issued forecasts.  The timestamp is treated as forecast
    valid time, not as a post-event observation time.  Do not pass post-event
    summaries or residual-oriented prompts built from future observed
    OT/radiation labels.
    """
    ensure_dir(cache_dir)
    fingerprint = _cache_fingerprint(
        text_path, model_path, TEXT2_FIELDS, "main-v12-text2-risk-scalar-availability"
    )
    cache_path = os.path.join(cache_dir, f"text2_{fingerprint}.npz")
    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=False)
        return {key: data[key] for key in data.files}

    df = load_json_or_jsonl(text_path)
    if "timestamp" not in df.columns and "date" in df.columns:
        df["timestamp"] = parse_datetime(df["date"])
    if "low_frequency_trend_prompt" not in df.columns:
        if "trend_prompt" in df.columns:
            df["low_frequency_trend_prompt"] = df["trend_prompt"]
        elif "text" in df.columns:
            df["low_frequency_trend_prompt"] = df["text"]
        else:
            df["low_frequency_trend_prompt"] = ""
    if "high_frequency_component_prompt" not in df.columns:
        if "high_frequency_risk_prompt" in df.columns:
            df["high_frequency_component_prompt"] = df["high_frequency_risk_prompt"]
        elif "text" in df.columns:
            df["high_frequency_component_prompt"] = df["text"]
        elif "trend_prompt" in df.columns:
            df["high_frequency_component_prompt"] = df["trend_prompt"].fillna("").astype(str).map(
                _derive_text2_high_frequency_prompt
            )
        else:
            df["high_frequency_component_prompt"] = ""
    df = (
        df.dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )
    availability_source = None
    for column in ("available_time", "issue_time", "forecast_issue_time"):
        if column in df.columns:
            parsed = parse_datetime(df[column])
            if parsed.notna().any():
                df[column] = parsed
                availability_source = column
                break

    encoder = FrozenBGEEncoder(model_path, device)
    arrays: Dict[str, np.ndarray] = {
        "timestamp_ns": df["timestamp"].astype("int64").to_numpy(dtype=np.int64)
    }
    if availability_source is not None:
        arrays["available_timestamp_ns"] = (
            df[availability_source]
            .fillna(df["timestamp"])
            .astype("int64")
            .to_numpy(dtype=np.int64)
        )
        arrays["availability_source_code"] = np.asarray(
            [{"available_time": 1, "issue_time": 2, "forecast_issue_time": 3}[availability_source]],
            dtype=np.int32,
        )
    for field in TEXT2_FIELDS:
        values = df[field].fillna("").astype(str).tolist()
        arrays[field] = encoder.encode_sentences(values, batch_size=batch_size)
    scalars = [
        _text2_scalar_features(low, high)
        for low, high in zip(
            df["low_frequency_trend_prompt"].fillna("").astype(str).tolist(),
            df["high_frequency_component_prompt"].fillna("").astype(str).tolist(),
        )
    ]
    arrays["text2_scalar"] = np.stack(scalars, axis=0).astype(np.float32)
    np.savez_compressed(cache_path, **arrays)
    return arrays

def _align_text(
    anchors: pd.DatetimeIndex,
    text_timestamp_ns: np.ndarray,
    mode: str,
    tolerance: pd.Timedelta,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return indices, validity mask, and matched timestamp ns."""
    text_ns = np.asarray(text_timestamp_ns, dtype=np.int64)
    anchor_ns = anchors.astype("int64").to_numpy(dtype=np.int64)
    if len(text_ns) == 0:
        return (
            np.full(len(anchor_ns), -1, dtype=np.int64),
            np.zeros(len(anchor_ns), dtype=bool),
            np.zeros(len(anchor_ns), dtype=np.int64),
        )

    if mode == "causal":
        idx = np.searchsorted(text_ns, anchor_ns, side="right") - 1
        safe = np.clip(idx, 0, len(text_ns) - 1)
        age = anchor_ns - text_ns[safe]
        valid = (idx >= 0) & (age >= 0) & (age <= tolerance.value)
    elif mode == "nearest":
        right = np.searchsorted(text_ns, anchor_ns, side="left")
        left = np.clip(right - 1, 0, len(text_ns) - 1)
        right_safe = np.clip(right, 0, len(text_ns) - 1)
        choose_right = np.abs(text_ns[right_safe] - anchor_ns) < np.abs(text_ns[left] - anchor_ns)
        safe = np.where(choose_right, right_safe, left)
        valid = np.abs(text_ns[safe] - anchor_ns) <= tolerance.value
        idx = safe.copy()
    else:
        raise ValueError(f"unknown alignment mode: {mode}")

    idx = np.where(valid, safe, -1).astype(np.int64)
    matched = np.where(valid, text_ns[safe], 0).astype(np.int64)
    return idx, valid.astype(bool), matched


def _gather_embeddings(array: np.ndarray, indices: np.ndarray, valid: np.ndarray) -> np.ndarray:
    output = np.zeros((len(indices), array.shape[1]), dtype=np.float32)
    if valid.any():
        output[valid] = array[indices[valid]]
    return output


def _derive_text2_high_frequency_prompt(text: str) -> str:
    """Build a leakage-safe short-risk prompt from NWP-derived trend text."""
    text = str(text).lower()
    rapid_inc = text.count("rapid increasing") + text.count("rapidly increasing")
    rapid_dec = text.count("rapid decreasing") + text.count("rapidly decreasing")
    slight_inc = text.count("slight increasing")
    slight_dec = text.count("slight decreasing")
    fluctuation = any(
        phrase in text
        for phrase in ["mild fluctuation", "fluctuate slightly", "mixed trend", "transition"]
    )
    upward = "upward risk" in text or rapid_inc >= 2
    downward = "downward risk" in text or rapid_dec >= 2
    mixed_direction = (rapid_inc > 0 and rapid_dec > 0) or (slight_inc > 0 and slight_dec > 0)
    if downward:
        direction = "downward ramp risk"
    elif upward:
        direction = "upward ramp risk"
    elif mixed_direction or fluctuation:
        direction = "ambiguous fluctuation risk"
    else:
        direction = "stable short-term background"

    if downward or upward or rapid_inc + rapid_dec >= 3:
        intensity = "high-confidence"
    elif mixed_direction or fluctuation or rapid_inc + rapid_dec > 0:
        intensity = "medium-confidence"
    else:
        intensity = "low-confidence"

    if direction == "stable short-term background":
        return (
            "Stable future risk view: NWP-derived radiation trend provides no clear "
            f"high-frequency residual event; confidence is {intensity}."
        )
    return (
        f"Fuzzy future risk view: {direction}. "
        "NWP-derived radiation trends indicate cloud-edge timing uncertainty, "
        "possible brief reversal, and a local residual opportunity. "
        f"The semantic confidence is {intensity}."
    )


def _text2_scalar_features(low_text: str, high_text: str) -> np.ndarray:
    text = f"{low_text} {high_text}".lower()
    def has_any(patterns: Sequence[str]) -> float:
        return float(any(p in text for p in patterns))

    downward = has_any([
        "decrease", "decreasing", "drop", "downward", "downward risk",
        "downward ramp", "weaken", "weakening", "decline", "fall",
        "drop risk", "low estimate", "overestimate",
    ])
    upward = has_any([
        "increase", "increasing", "rise", "rising", "upward", "upward risk",
        "upward ramp", "strengthen", "recover", "underestimate", "rise risk",
    ])
    extreme = has_any([
        "sharp", "ramp", "extreme", "cloud-edge", "reversal", "mutation",
        "rapid", "rapidly", "drop risk", "rise risk", "downward risk",
        "upward risk", "local residual opportunity",
    ])
    fuzzy = has_any([
        "fuzzy", "ambiguity", "ambiguous", "uncertainty", "vague",
        "cloud-edge timing", "brief reversal", "mild fluctuation",
        "fluctuate", "mixed trend", "transition",
    ])
    stable_signal = has_any([
        "stable future risk view", "future radiation remains stable",
        "expected to be steady", "minimal correction", "track closely",
    ])
    stable = float(bool(stable_signal) and not bool(extreme or fuzzy or downward or upward))
    day = has_any(["daytime", "radiation", "sun", "cloud", "irradiance"])
    night = has_any(["nighttime", "night"])
    conf_high = has_any(["high-confidence", "high confidence"])
    conf_med = has_any(["medium-confidence", "medium confidence"])
    conf_low = has_any(["low-confidence", "low confidence"])
    bias = downward - upward
    return np.asarray(
        [downward, upward, extreme, fuzzy, stable, day, night, bias, conf_high, conf_med, conf_low],
        dtype=np.float32,
    )



def _gather_token_embeddings(
    token_array: np.ndarray,
    token_mask_array: np.ndarray,
    indices: np.ndarray,
    valid_hours: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    hours, max_tokens, dim = len(indices), token_array.shape[1], token_array.shape[2]
    output = np.zeros((hours, max_tokens, dim), dtype=np.float32)
    output_mask = np.zeros((hours, max_tokens), dtype=bool)
    if valid_hours.any():
        output[valid_hours] = token_array[indices[valid_hours]].astype(np.float32)
        output_mask[valid_hours] = token_mask_array[indices[valid_hours]].astype(bool)
    output_mask &= valid_hours[:, None]
    return output, output_mask


@dataclass(frozen=True)
class DatasetOptions:
    seq_len: int = 96
    pred_len: int = 45
    sample_minutes: int = 15
    strict_frequency: bool = True
    text_tolerance_minutes: int = 50
    prediction_start_hour: Optional[float] = 8.0
    prediction_end_hour: Optional[float] = 19.0
    prediction_include_end: bool = True
    event_sample_quantile: float = 0.70
    event_sample_alpha: float = 1.50
    event_sample_max_weight: float = 4.00


class FusionDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: Sequence[str],
        target_col: str,
        nwp_cols: Sequence[str],
        options: DatasetOptions,
        window_start: int,
        window_stop: int,
        x_scaler: StandardScaler,
        y_scaler: StandardScaler,
        nwp_scaler: Optional[StandardScaler],
        text1_cache: Optional[Dict[str, np.ndarray]],
        text2_cache: Optional[Dict[str, np.ndarray]],
        use_past_nwp: bool,
        use_future_nwp: bool,
        use_text1: bool,
        use_text2: bool,
        dataset_index: int = 0,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.times = pd.DatetimeIndex(self.df["date"])
        self.feature_cols = list(feature_cols)
        self.target_col = target_col
        self.nwp_cols = list(nwp_cols)
        self.options = options
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.nwp_scaler = nwp_scaler
        self.text1_cache = text1_cache
        self.text2_cache = text2_cache
        self.use_past_nwp = use_past_nwp
        self.use_future_nwp = use_future_nwp
        self.use_text1 = use_text1
        self.use_text2 = use_text2
        self.dataset_index = int(dataset_index)

        self.x_all = x_scaler.transform(self.df[self.feature_cols].to_numpy(np.float32))
        target_source = TARGET_VALUE_COL if TARGET_VALUE_COL in self.df.columns else target_col
        y_values = self.df[[target_source]].to_numpy(np.float32)
        if hasattr(y_scaler, "transform_with_index"):
            self.y_all = y_scaler.transform_with_index(y_values, self.dataset_index)
        else:
            self.y_all = y_scaler.transform(y_values)
        if TARGET_AVAILABLE_COL in self.df.columns:
            self.target_available_all = self.df[TARGET_AVAILABLE_COL].to_numpy(bool)
        else:
            self.target_available_all = np.isfinite(y_values[:, 0])
        self.time_features = add_time_features(self.df).to_numpy(np.float32)
        if self.nwp_cols:
            if nwp_scaler is None:
                raise ValueError("nwp_scaler is required when nwp_cols are present")
            self.nwp_all = nwp_scaler.transform(self.df[self.nwp_cols].to_numpy(np.float32))
        else:
            self.nwp_all = None

        max_start = len(self.df) - options.seq_len - options.pred_len + 1
        candidates = np.arange(max(0, window_start), min(window_stop, max_start), dtype=np.int64)
        if options.strict_frequency and len(candidates):
            expected = np.timedelta64(options.sample_minutes, "m")
            time_values = self.times.to_numpy(dtype="datetime64[ns]")
            bad_transition = (np.diff(time_values) != expected).astype(np.int64)
            # prefix[j] counts bad transitions among diff[:j].  A window with
            # total_points points contains transitions [i, i + total_points - 2].
            prefix = np.concatenate([[0], np.cumsum(bad_transition)])
            total_points = options.seq_len + options.pred_len
            bad_count = prefix[candidates + total_points - 1] - prefix[candidates]
            candidates = candidates[bad_count == 0]
        if len(candidates):
            candidates = self._filter_prediction_window(candidates)
        self.indices = candidates

    def event_sampling_weights(
        self,
        quantile: Optional[float] = None,
        alpha: Optional[float] = None,
        max_weight: Optional[float] = None,
    ) -> np.ndarray:
        """Return target-distribution weights for training-only event sampling.

        The score is computed in normalized target space from each supervised
        window's level, ramp, and curvature.  It is dataset-relative and does
        not depend on station names, so the same rule can be used for single-
        and multi-station training.  These weights are only consumed by the
        DataLoader sampler; inference never receives them.
        """
        if len(self.indices) == 0:
            return np.zeros(0, dtype=np.float64)
        q = float(self.options.event_sample_quantile if quantile is None else quantile)
        a = float(self.options.event_sample_alpha if alpha is None else alpha)
        cap = float(self.options.event_sample_max_weight if max_weight is None else max_weight)
        if a <= 0.0 or cap <= 1.0:
            return np.ones(len(self.indices), dtype=np.float64)
        scores: List[float] = []
        for start in self.indices:
            p0 = int(start) + self.options.seq_len
            p1 = p0 + self.options.pred_len
            target = np.asarray(self.y_all[p0:p1, 0], dtype=np.float64)
            if target.size == 0:
                scores.append(0.0)
                continue
            target = np.where(np.isfinite(target), target, 0.0)
            level = float(np.nanpercentile(np.abs(target), 90))
            if target.size > 1:
                ramp = np.diff(target)
                ramp_score = float(np.nanpercentile(np.abs(ramp), 95))
            else:
                ramp_score = 0.0
            if target.size > 2:
                curvature = np.diff(target, n=2)
                curvature_score = float(np.nanpercentile(np.abs(curvature), 95))
            else:
                curvature_score = 0.0
            score = (
                0.44 * np.tanh(level / 2.5)
                + 0.40 * np.tanh(ramp_score / 0.80)
                + 0.16 * np.tanh(curvature_score / 0.55)
            )
            scores.append(float(score))
        score_array = np.asarray(scores, dtype=np.float64)
        if not np.isfinite(score_array).any():
            return np.ones(len(self.indices), dtype=np.float64)
        score_array = np.where(np.isfinite(score_array), score_array, 0.0)
        threshold = float(np.quantile(score_array, np.clip(q, 0.0, 0.98)))
        high = float(np.quantile(score_array, 0.95))
        width = max(high - threshold, 1e-6)
        focus = np.clip((score_array - threshold) / width, 0.0, 1.0)
        weights = 1.0 + a * focus
        weights = np.clip(weights, 1.0, cap)
        mean = float(weights.mean()) if weights.size else 1.0
        if mean > 0.0:
            weights = weights / mean
        return weights.astype(np.float64)

    def __len__(self) -> int:
        return len(self.indices)

    @staticmethod
    def _hour_float(times: pd.DatetimeIndex) -> np.ndarray:
        return (
            times.hour.to_numpy(np.float32)
            + times.minute.to_numpy(np.float32) / 60.0
            + times.second.to_numpy(np.float32) / 3600.0
        )

    def _filter_prediction_window(self, candidates: np.ndarray) -> np.ndarray:
        start_hour = self.options.prediction_start_hour
        end_hour = self.options.prediction_end_hour
        if start_hour is None or end_hour is None:
            return candidates
        kept: List[int] = []
        target_available = np.asarray(self.target_available_all, dtype=bool)
        tolerance = self.options.sample_minutes / 60.0 / 4.0
        for candidate in candidates:
            p0 = int(candidate) + self.options.seq_len
            p1 = p0 + self.options.pred_len
            if p1 > len(self.times):
                continue
            future_times = self.times[p0:p1]
            future_hours = self._hour_float(future_times)
            origin_hour = float(future_hours[0])
            if abs(origin_hour - float(start_hour)) > tolerance:
                continue
            if self.options.prediction_include_end:
                in_window = (future_hours >= float(start_hour) - 1e-6) & (
                    future_hours <= float(end_hour) + 1e-6
                )
            else:
                in_window = (future_hours >= float(start_hour) - 1e-6) & (
                    future_hours < float(end_hour) - 1e-6
                )
            if not bool(np.all(in_window)):
                continue
            if len(pd.DatetimeIndex(future_times).normalize().unique()) != 1:
                continue
            if not bool(np.all(target_available[p0:p1])):
                continue
            kept.append(int(candidate))
        return np.asarray(kept, dtype=np.int64)

    def _text1(self, history_start: int, forecast_origin: int):
        hours = math_ceil_div(self.options.seq_len * self.options.sample_minutes, 60)
        last_anchor = self.times[forecast_origin].floor("h")
        anchors = pd.date_range(end=last_anchor, periods=hours, freq="1h")
        tolerance = pd.Timedelta(minutes=self.options.text_tolerance_minutes)
        if not self.use_text1 or self.text1_cache is None:
            max_tokens, dim = 1, 1
            token_zero = np.zeros((hours, max_tokens, dim), dtype=np.float32)
            mask_zero = np.zeros((hours, max_tokens), dtype=bool)
            sentence_zero = np.zeros((hours, dim), dtype=np.float32)
            return (
                token_zero, token_zero.copy(), token_zero.copy(),
                mask_zero, mask_zero.copy(), mask_zero.copy(),
                sentence_zero, sentence_zero.copy(), sentence_zero.copy(),
                np.zeros(hours, bool), np.zeros(hours, np.float32),
            )

        idx, valid, matched_ns = _align_text(
            anchors, self.text1_cache["timestamp_ns"], "causal", tolerance
        )
        origin_ns = self.times[forecast_origin].value
        relative_hours = (matched_ns - origin_ns) / 3.6e12
        relative_hours = np.where(valid, relative_hours, 0.0).astype(np.float32)
        field_data = []
        for field in TEXT1_FIELDS:
            tokens, token_mask = _gather_token_embeddings(
                self.text1_cache[f"{field}_tokens"],
                self.text1_cache[f"{field}_token_mask"],
                idx,
                valid,
            )
            sentence = _gather_embeddings(
                self.text1_cache[f"{field}_sentence"], idx, valid
            )
            field_data.append((tokens, token_mask, sentence))
        return (
            field_data[0][0], field_data[1][0], field_data[2][0],
            field_data[0][1], field_data[1][1], field_data[2][1],
            field_data[0][2], field_data[1][2], field_data[2][2],
            valid, relative_hours,
        )

    def _text2(self, forecast_origin: int):
        hours = math_ceil_div(self.options.pred_len * self.options.sample_minutes, 60)
        start = self.times[forecast_origin].floor("h")
        anchors = pd.date_range(start=start, periods=hours, freq="1h")
        tolerance = pd.Timedelta(minutes=self.options.text_tolerance_minutes)
        if not self.use_text2 or self.text2_cache is None:
            dim = 1
            zeros = np.zeros((hours, dim), dtype=np.float32)
            scalar = np.zeros((hours, TEXT2_SCALAR_DIM), dtype=np.float32)
            return zeros, zeros.copy(), np.zeros(hours, bool), np.zeros(hours, np.float32), scalar

        idx, valid, matched_ns = _align_text(
            anchors, self.text2_cache["timestamp_ns"], "nearest", tolerance
        )
        origin_ns = self.times[forecast_origin].value
        available_ns = self.text2_cache.get("available_timestamp_ns")
        if available_ns is not None and valid.any():
            availability = np.asarray(available_ns, dtype=np.int64)
            safe_idx = np.clip(idx, 0, max(len(availability) - 1, 0))
            valid = valid & (availability[safe_idx] <= origin_ns)
        relative_hours = (matched_ns - origin_ns) / 3.6e12
        relative_hours = np.where(valid, relative_hours, 0.0).astype(np.float32)
        scalar_cache = self.text2_cache.get("text2_scalar")
        if scalar_cache is None:
            scalar = np.zeros((hours, TEXT2_SCALAR_DIM), dtype=np.float32)
        else:
            scalar = _gather_embeddings(scalar_cache, idx, valid)
        return (
            _gather_embeddings(self.text2_cache[TEXT2_FIELDS[0]], idx, valid),
            _gather_embeddings(self.text2_cache[TEXT2_FIELDS[1]], idx, valid),
            valid,
            relative_hours,
            scalar,
        )

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        i = int(self.indices[item])
        s0, s1 = i, i + self.options.seq_len
        p0, p1 = s1, s1 + self.options.pred_len
        origin = self.times[p0]

        past_offsets = ((self.times[s0:s1] - origin) / pd.Timedelta(hours=1)).to_numpy(np.float32)
        future_offsets = ((self.times[p0:p1] - origin) / pd.Timedelta(hours=1)).to_numpy(np.float32)

        if self.nwp_all is not None and self.use_past_nwp:
            past_nwp = self.nwp_all[s0:s1]
            past_nwp_mask = np.ones(self.options.seq_len, dtype=bool)
        else:
            past_nwp = np.zeros((self.options.seq_len, 1), dtype=np.float32)
            past_nwp_mask = np.zeros(self.options.seq_len, dtype=bool)

        if self.nwp_all is not None and self.use_future_nwp:
            future_nwp = self.nwp_all[p0:p1]
            future_nwp_mask = np.ones(self.options.pred_len, dtype=bool)
        else:
            future_nwp = np.zeros((self.options.pred_len, 1), dtype=np.float32)
            future_nwp_mask = np.zeros(self.options.pred_len, dtype=bool)

        (
            text_state_tokens, text_trend_tokens, text_var_tokens,
            text_state_token_mask, text_trend_token_mask, text_var_token_mask,
            text_state_sentence, text_trend_sentence, text_var_sentence,
            text1_mask, text1_time,
        ) = self._text1(s0, p0)
        text2_low, text2_high, text2_mask, text2_time, text2_scalar = self._text2(p0)

        arrays = {
            "x": self.x_all[s0:s1],
            "y": self.y_all[p0:p1, 0],
            "past_time_features": self.time_features[s0:s1],
            "future_time_features": self.time_features[p0:p1],
            "past_offset_hours": past_offsets,
            "future_offset_hours": future_offsets,
            "past_nwp": past_nwp,
            "future_nwp": future_nwp,
            "past_nwp_mask": past_nwp_mask,
            "future_nwp_mask": future_nwp_mask,
            "text_state_tokens": text_state_tokens,
            "text_trend_tokens": text_trend_tokens,
            "text_var_tokens": text_var_tokens,
            "text_state_token_mask": text_state_token_mask,
            "text_trend_token_mask": text_trend_token_mask,
            "text_var_token_mask": text_var_token_mask,
            "text_state_sentence": text_state_sentence,
            "text_trend_sentence": text_trend_sentence,
            "text_var_sentence": text_var_sentence,
            "text1_mask": text1_mask,
            "text1_offset_hours": text1_time,
            "text2_low": text2_low,
            "text2_high": text2_high,
            "text2_mask": text2_mask,
            "text2_offset_hours": text2_time,
            "text2_scalar": text2_scalar,
            "dataset_index": np.asarray(self.dataset_index, dtype=np.int64),
        }
        output: Dict[str, torch.Tensor] = {}
        for key, value in arrays.items():
            if np.asarray(value).dtype == bool:
                output[key] = torch.as_tensor(value, dtype=torch.bool)
            elif np.asarray(value).dtype.kind in {"i", "u"}:
                output[key] = torch.as_tensor(value, dtype=torch.long)
            else:
                output[key] = torch.as_tensor(value, dtype=torch.float32)
        return output


def math_ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def make_loaders(
    primary_csv: str,
    feature_cols: Sequence[str],
    target_col: str,
    nwp_cols: Sequence[str],
    seq_len: int,
    pred_len: int,
    batch_size: int,
    num_workers: int,
    aux_csv: Optional[str] = None,
    nwp_csv: Optional[str] = None,
    context_csv: Optional[str] = None,
    text1_cache: Optional[Dict[str, np.ndarray]] = None,
    text2_cache: Optional[Dict[str, np.ndarray]] = None,
    use_past_nwp: bool = False,
    use_future_nwp: bool = False,
    use_text1: bool = False,
    use_text2: bool = False,
    strict_frequency: bool = True,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    prediction_start_hour: Optional[float] = 8.0,
    prediction_end_hour: Optional[float] = 19.0,
    prediction_include_end: bool = True,
    event_balanced_sampling: bool = True,
    event_sample_quantile: float = 0.70,
    event_sample_alpha: float = 1.50,
    event_sample_max_weight: float = 4.00,
):
    df = load_context_target_numeric(primary_csv, aux_csv, nwp_csv, context_csv, target_col)
    validate_columns(df, [*feature_cols, *nwp_cols, TARGET_VALUE_COL], primary_csv)
    n = len(df)
    n_train, n_val_end = split_points(n, train_ratio, val_ratio)

    x_scaler = StandardScaler().fit(df.iloc[:n_train][list(feature_cols)].to_numpy(np.float32))
    y_scaler = StandardScaler().fit(
        _target_values_for_scaler(
            df,
            n_train,
            prediction_start_hour,
            prediction_end_hour,
            prediction_include_end,
        ).astype(np.float32)
    )
    nwp_scaler = None
    if nwp_cols:
        nwp_scaler = StandardScaler().fit(df.iloc[:n_train][list(nwp_cols)].to_numpy(np.float32))

    options = DatasetOptions(
        seq_len=seq_len,
        pred_len=pred_len,
        strict_frequency=strict_frequency,
        prediction_start_hour=prediction_start_hour,
        prediction_end_hour=prediction_end_hour,
        prediction_include_end=prediction_include_end,
        event_sample_quantile=event_sample_quantile,
        event_sample_alpha=event_sample_alpha,
        event_sample_max_weight=event_sample_max_weight,
    )
    # Window starts are chosen so the first target is exactly at the split boundary.
    train_range = (0, n_train - seq_len - pred_len + 1)
    val_range = (n_train - seq_len, n_val_end - seq_len - pred_len + 1)
    test_range = (n_val_end - seq_len, n - seq_len - pred_len + 1)

    common = dict(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        nwp_cols=nwp_cols,
        options=options,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        nwp_scaler=nwp_scaler,
        text1_cache=text1_cache,
        text2_cache=text2_cache,
        use_past_nwp=use_past_nwp,
        use_future_nwp=use_future_nwp,
        use_text1=use_text1,
        use_text2=use_text2,
    )
    train_ds = FusionDataset(window_start=train_range[0], window_stop=train_range[1], **common)
    val_ds = FusionDataset(window_start=val_range[0], window_stop=val_range[1], **common)
    test_ds = FusionDataset(window_start=test_range[0], window_stop=test_range[1], **common)

    if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
        raise ValueError(
            "one or more splits contain no valid windows; disable strict_frequency for a gapped dataset "
            "or reduce seq_len/pred_len"
        )

    loader_args = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    train_sampler = None
    train_shuffle = True
    if event_balanced_sampling and event_sample_alpha > 0.0:
        sample_weights = train_ds.event_sampling_weights(
            quantile=event_sample_quantile,
            alpha=event_sample_alpha,
            max_weight=event_sample_max_weight,
        )
        if sample_weights.size:
            train_sampler = WeightedRandomSampler(
                torch.as_tensor(sample_weights, dtype=torch.double),
                num_samples=len(sample_weights),
                replacement=True,
            )
            train_shuffle = False
    train_loader = DataLoader(
        train_ds,
        shuffle=train_shuffle,
        sampler=train_sampler,
        drop_last=False,
        **loader_args,
    )
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_args)
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **loader_args)
    metadata = {
        "forecast_protocol": "direct_multi_horizon",
        "metric_space": "normalized_target",
        "seq_len": int(seq_len),
        "pred_len": int(pred_len),
        "window_stride": 1,
        "sample_minutes": int(options.sample_minutes),
        "feature_cols": list(feature_cols),
        "nwp_cols": list(nwp_cols),
        "split_points": [n_train, n_val_end],
        "split_sizes": [len(train_ds), len(val_ds), len(test_ds)],
        "target_ranges": {
            "train": [str(df.iloc[seq_len]["date"]), str(df.iloc[n_train - 1]["date"])],
            "val": [str(df.iloc[n_train]["date"]), str(df.iloc[n_val_end - 1]["date"])],
            "test": [str(df.iloc[n_val_end]["date"]), str(df.iloc[n - 1]["date"])],
        },
        "context_source": str(context_csv) if context_csv else str(primary_csv),
        "target_source": str(primary_csv),
        "prediction_window": {
            "start_hour": prediction_start_hour,
            "end_hour": prediction_end_hour,
            "include_end": bool(prediction_include_end),
            "context_hours": float(seq_len * options.sample_minutes / 60.0),
        },
        "target_scaler_fit": "train_split_supervised_prediction_window_only",
        "text1_alignment": "causal_24h_hourly_window_with_full_BGE_word_tokens",
        "text2_alignment": (
            "nearest_hourly_forecast_valid_time_tokens_over_horizon;"
            "source_must_be_available_at_forecast_issue_time"
        ),
        "event_balanced_sampling": bool(event_balanced_sampling),
        "event_sample_quantile": float(event_sample_quantile),
        "event_sample_alpha": float(event_sample_alpha),
        "event_sample_max_weight": float(event_sample_max_weight),
        "train_drop_last": False,
    }
    return train_loader, val_loader, test_loader, x_scaler, y_scaler, nwp_scaler, metadata


def make_multi_loaders(
    dataset_specs: Sequence[Dict[str, Optional[str]]],
    feature_cols: Sequence[str],
    target_col: str,
    nwp_cols: Sequence[str],
    seq_len: int,
    pred_len: int,
    batch_size: int,
    num_workers: int,
    text1_caches: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    text2_caches: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    use_past_nwp: bool = False,
    use_future_nwp: bool = False,
    use_text1: bool = False,
    use_text2: bool = False,
    strict_frequency: bool = True,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    balanced_train_sampling: bool = True,
    prediction_start_hour: Optional[float] = 8.0,
    prediction_end_hour: Optional[float] = 19.0,
    prediction_include_end: bool = True,
    event_balanced_sampling: bool = True,
    event_sample_quantile: float = 0.70,
    event_sample_alpha: float = 1.50,
    event_sample_max_weight: float = 4.00,
):
    """Build loaders for multiple independent solar stations.

    Each station keeps its own chronological split and text alignment.  Numeric
    scalers are shared and fitted on the union of all training partitions, which
    avoids timestamp collisions and cross-station text leakage.
    """
    if not dataset_specs:
        raise ValueError("dataset_specs is empty")

    frames: List[pd.DataFrame] = []
    names: List[str] = []
    train_ends: List[int] = []
    val_ends: List[int] = []
    for spec in dataset_specs:
        name = str(spec.get("name") or f"dataset_{len(names)}")
        primary_csv = spec.get("solar_csv")
        if not primary_csv:
            raise ValueError(f"dataset {name} is missing solar_csv")
        df = load_and_merge_numeric(
            str(spec.get("context_csv") or primary_csv),
            spec.get("solar_aux_csv"),
            spec.get("solar_nwp_csv"),
        )
        if spec.get("context_csv"):
            target_df = load_numeric_csv(str(primary_csv))
            validate_columns(target_df, [target_col], str(primary_csv))
            target_df = target_df[["date", target_col]].rename(columns={target_col: TARGET_VALUE_COL})
            df = df.merge(target_df, on="date", how="left", validate="one_to_one")
            df[TARGET_AVAILABLE_COL] = np.isfinite(pd.to_numeric(df[TARGET_VALUE_COL], errors="coerce"))
        else:
            df[TARGET_VALUE_COL] = pd.to_numeric(df[target_col], errors="coerce")
            df[TARGET_AVAILABLE_COL] = np.isfinite(df[TARGET_VALUE_COL])
        validate_columns(df, [*feature_cols, *nwp_cols, TARGET_VALUE_COL], str(primary_csv))
        n_train, n_val_end = split_points(len(df), train_ratio, val_ratio)
        frames.append(df)
        names.append(name)
        train_ends.append(n_train)
        val_ends.append(n_val_end)

    x_scaler = _fit_scaler_from_frames(frames, feature_cols, train_ends)
    y_scaler = IndexedTargetScaler().fit_by_arrays(
        [
            _target_values_for_scaler(
                frame,
                train_end,
                prediction_start_hour,
                prediction_end_hour,
                prediction_include_end,
            )
            for frame, train_end in zip(frames, train_ends)
        ],
        names,
    )
    nwp_scaler = None
    if nwp_cols:
        nwp_scaler = _fit_scaler_from_frames(frames, nwp_cols, train_ends)

    options = DatasetOptions(
        seq_len=seq_len,
        pred_len=pred_len,
        strict_frequency=strict_frequency,
        prediction_start_hour=prediction_start_hour,
        prediction_end_hour=prediction_end_hour,
        prediction_include_end=prediction_include_end,
        event_sample_quantile=event_sample_quantile,
        event_sample_alpha=event_sample_alpha,
        event_sample_max_weight=event_sample_max_weight,
    )
    train_sets: List[FusionDataset] = []
    val_sets: List[FusionDataset] = []
    test_sets: List[FusionDataset] = []
    split_sizes: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
    target_ranges: Dict[str, Dict[str, List[str]]] = {}

    for name, df, n_train, n_val_end in zip(names, frames, train_ends, val_ends):
        train_range = (0, n_train - seq_len - pred_len + 1)
        val_range = (n_train - seq_len, n_val_end - seq_len - pred_len + 1)
        test_range = (n_val_end - seq_len, len(df) - seq_len - pred_len + 1)
        common = dict(
            df=df,
            feature_cols=feature_cols,
            target_col=target_col,
            nwp_cols=nwp_cols,
            options=options,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            nwp_scaler=nwp_scaler,
            text1_cache=(text1_caches or {}).get(name),
            text2_cache=(text2_caches or {}).get(name),
            use_past_nwp=use_past_nwp,
            use_future_nwp=use_future_nwp,
            use_text1=use_text1 and (text1_caches or {}).get(name) is not None,
            use_text2=use_text2 and (text2_caches or {}).get(name) is not None,
            dataset_index=names.index(name),
        )
        train_ds = FusionDataset(window_start=train_range[0], window_stop=train_range[1], **common)
        val_ds = FusionDataset(window_start=val_range[0], window_stop=val_range[1], **common)
        test_ds = FusionDataset(window_start=test_range[0], window_stop=test_range[1], **common)
        if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
            raise ValueError(
                f"dataset {name} has an empty split; disable strict_frequency or reduce seq_len/pred_len"
            )
        train_sets.append(train_ds)
        val_sets.append(val_ds)
        test_sets.append(test_ds)
        split_sizes["train"].append(len(train_ds))
        split_sizes["val"].append(len(val_ds))
        split_sizes["test"].append(len(test_ds))
        target_ranges[name] = {
            "train": [str(df.iloc[seq_len]["date"]), str(df.iloc[n_train - 1]["date"])],
            "val": [str(df.iloc[n_train]["date"]), str(df.iloc[n_val_end - 1]["date"])],
            "test": [str(df.iloc[n_val_end]["date"]), str(df.iloc[len(df) - 1]["date"])],
        }

    train_ds = ConcatDataset(train_sets)
    val_ds = ConcatDataset(val_sets)
    test_ds = ConcatDataset(test_sets)

    loader_args = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    train_sampler = None
    train_shuffle = True
    if balanced_train_sampling or (event_balanced_sampling and event_sample_alpha > 0.0):
        sample_weights: List[float] = []
        for train_set in train_sets:
            if balanced_train_sampling:
                base = np.full(len(train_set), 1.0 / max(len(train_set), 1), dtype=np.float64)
            else:
                base = np.ones(len(train_set), dtype=np.float64)
            if event_balanced_sampling and event_sample_alpha > 0.0:
                event_weights = train_set.event_sampling_weights(
                    quantile=event_sample_quantile,
                    alpha=event_sample_alpha,
                    max_weight=event_sample_max_weight,
                )
                base = base * event_weights
            sample_weights.extend(base.tolist())
        if sample_weights:
            weights = np.asarray(sample_weights, dtype=np.float64)
            total = float(weights.sum())
            if total > 0.0:
                weights = weights * (len(weights) / total)
            train_sampler = WeightedRandomSampler(
                torch.as_tensor(weights, dtype=torch.double),
                num_samples=len(weights),
                replacement=True,
            )
            train_shuffle = False
    train_loader = DataLoader(
        train_ds,
        shuffle=train_shuffle,
        sampler=train_sampler,
        drop_last=False,
        **loader_args,
    )
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_args)
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **loader_args)
    metadata = {
        "forecast_protocol": "direct_multi_horizon_multi_dataset",
        "metric_space": "normalized_target",
        "dataset_names": list(names),
        "dataset_count": len(names),
        "seq_len": int(seq_len),
        "pred_len": int(pred_len),
        "window_stride": 1,
        "sample_minutes": int(options.sample_minutes),
        "feature_cols": list(feature_cols),
        "nwp_cols": list(nwp_cols),
        "split_points": {
            name: [int(n_train), int(n_val_end)]
            for name, n_train, n_val_end in zip(names, train_ends, val_ends)
        },
        "split_sizes_by_dataset": split_sizes,
        "split_sizes": [len(train_ds), len(val_ds), len(test_ds)],
        "target_ranges": target_ranges,
        "context_source": {
            name: str(spec.get("context_csv") or spec.get("solar_csv"))
            for name, spec in zip(names, dataset_specs)
        },
        "target_source": {
            name: str(spec.get("solar_csv"))
            for name, spec in zip(names, dataset_specs)
        },
        "prediction_window": {
            "start_hour": prediction_start_hour,
            "end_hour": prediction_end_hour,
            "include_end": bool(prediction_include_end),
            "context_hours": float(seq_len * options.sample_minutes / 60.0),
        },
        "target_scaler_fit": "per_dataset_train_split_supervised_prediction_window_only",
        "text1_alignment": "causal_24h_hourly_window_with_full_BGE_word_tokens_per_dataset",
        "text2_alignment": (
            "nearest_hourly_forecast_valid_time_tokens_over_horizon_per_dataset;"
            "source_must_be_available_at_forecast_issue_time"
        ),
        "balanced_train_sampling": bool(balanced_train_sampling),
        "event_balanced_sampling": bool(event_balanced_sampling),
        "event_sample_quantile": float(event_sample_quantile),
        "event_sample_alpha": float(event_sample_alpha),
        "event_sample_max_weight": float(event_sample_max_weight),
        "train_drop_last": False,
    }
    return train_loader, val_loader, test_loader, x_scaler, y_scaler, nwp_scaler, metadata


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def regression_metrics(prediction: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    prediction = np.asarray(prediction).reshape(-1)
    target = np.asarray(target).reshape(-1)
    error = prediction - target
    mse = float(np.mean(error ** 2))
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(mse))
    denom = np.maximum(np.abs(target), 1e-3)
    mape = float(np.mean(np.abs(error) / denom) * 100.0)
    smape = float(
        np.mean(2.0 * np.abs(error) / np.maximum(np.abs(prediction) + np.abs(target), 1e-3))
        * 100.0
    )
    wape = float(np.sum(np.abs(error)) / max(np.sum(np.abs(target)), 1e-6) * 100.0)
    ss_res = float(np.sum(error ** 2))
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    r2 = float(1.0 - ss_res / max(ss_tot, 1e-12))
    return {"MSE": mse, "MAE": mae, "RMSE": rmse, "MAPE": mape, "sMAPE": smape, "WAPE": wape, "R2": r2}
