# -*- coding: utf-8 -*-
"""
Paper-grade iTransformer backbone and continuous-time cross-attention blocks.

Design goals
------------
1. Preserve the inverted-token idea: variables are tokens and the historical axis
    is embedded into each variable token.
2. Use a full horizon decoder so future time information participates in the
    numerical forecast rather than only being exposed to auxiliary modules.
3. Keep RevIN stateless inside each forward pass.
4. Support asynchronous 15-minute/hourly fusion through learned continuous-time
    relative attention bias. Hourly text is never copied four times.
5. Stage-1 supports an explicit soft cross-modal gate for real-time text fusion.
6. Stage-2 remains non-gated; this file only provides reusable building blocks.
7. No random-seed manipulation.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RevIN(nn.Module):
    """Reversible instance normalization with per-sample statistics."""

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.affine = bool(affine)
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, num_features))
            self.beta = nn.Parameter(torch.zeros(1, 1, num_features))
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)

    def normalize(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        mean = x.mean(dim=1, keepdim=True).detach()
        std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
        z = (x - mean) / std
        if self.affine:
            z = z * self.gamma + self.beta
        return z, {"mean": mean, "std": std}

    def denormalize_target(
        self,
        y: torch.Tensor,
        stats: Dict[str, torch.Tensor],
        target_index: int,
    ) -> torch.Tensor:
        mean = stats["mean"][:, :, target_index]
        std = stats["std"][:, :, target_index]
        if self.affine:
            gamma = self.gamma[:, :, target_index].clamp_min(1e-6)
            beta = self.beta[:, :, target_index]
            y = (y - beta) / gamma
        return y * std + mean


class MovingAverage(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        kernel_size = int(kernel_size)
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("ma_kernel must be a positive odd integer")
        self.kernel_size = kernel_size
        self.pad = (kernel_size - 1) // 2
        self.pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel_size == 1:
            return x
        left = x[:, :1].expand(-1, self.pad, -1)
        right = x[:, -1:].expand(-1, self.pad, -1)
        padded = torch.cat([left, x, right], dim=1)
        return self.pool(padded.transpose(1, 2)).transpose(1, 2)


class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_average = MovingAverage(kernel_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        trend = self.moving_average(x)
        residual = x - trend
        return residual, trend


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class InvertedEncoderLayer(nn.Module):
    """Pre-norm transformer layer operating on variable tokens."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        z = self.norm1(x)
        attn, _ = self.attn(
            z, z, z, key_padding_mask=key_padding_mask, need_weights=False
        )
        x = x + attn
        x = x + self.ffn(self.norm2(x))
        return x


class InvertedEncoder(nn.Module):
    def __init__(self, layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [InvertedEncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return self.norm(x)


class HorizonDecoderLayer(nn.Module):
    """Full horizon decoder attending to inverted variable memory."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(
        self,
        q: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        z = self.norm1(q)
        h, _ = self.self_attn(z, z, z, need_weights=False)
        q = q + h
        z = self.norm2(q)
        h, _ = self.cross_attn(
            z,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        q = q + h
        q = q + self.ffn(self.norm3(q))
        return q


class HorizonDecoder(nn.Module):
    def __init__(self, layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [HorizonDecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        q: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            q = layer(q, memory, memory_key_padding_mask=memory_key_padding_mask)
        return self.norm(q)


class ContinuousTimeCrossAttention(nn.Module):
    """
    Multi-head cross-attention with a learned bias from continuous time distance.

    q_time and kv_time are measured in hours relative to the forecast origin.
    key_mask uses True for valid tokens and False for padding/missing tokens.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        causal: bool = False,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.causal = causal

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.time_bias = nn.Sequential(
            nn.Linear(4, d_model),
            nn.SiLU(),
            nn.Linear(d_model, n_heads),
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        q_time: torch.Tensor,
        kv_time: torch.Tensor,
        key_mask: Optional[torch.Tensor] = None,
        pair_mask: Optional[torch.Tensor] = None,
        distance_decay_hours: Optional[float] = None,
        return_attention: bool = False,
        context_only: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if query.ndim != 3 or key_value.ndim != 3:
            raise ValueError("query and key_value must be [B,T,D]")
        if q_time.ndim != 2 or kv_time.ndim != 2:
            raise ValueError("q_time and kv_time must be [B,T]")

        qn = self.norm1(query)
        q = self._reshape_heads(self.q_proj(qn))
        k = self._reshape_heads(self.k_proj(key_value))
        v = self._reshape_heads(self.v_proj(key_value))

        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        dt = q_time.unsqueeze(-1) - kv_time.unsqueeze(-2)
        time_features = torch.stack(
            [
                dt,
                dt.abs(),
                torch.sign(dt) * torch.log1p(dt.abs()),
                torch.exp(-dt.abs() / 6.0),
            ],
            dim=-1,
        )
        bias = self.time_bias(time_features).permute(0, 3, 1, 2)
        logits = logits + bias
        if distance_decay_hours is not None:
            if distance_decay_hours <= 0:
                raise ValueError("distance_decay_hours must be positive")
            logits = logits - dt.abs().unsqueeze(1) / float(distance_decay_hours)

        valid = torch.ones_like(logits, dtype=torch.bool)
        if key_mask is not None:
            valid = valid & key_mask[:, None, None, :].bool()
        if pair_mask is not None:
            if pair_mask.ndim == 2:
                pair_mask = pair_mask.unsqueeze(0)
            if pair_mask.ndim != 3:
                raise ValueError("pair_mask must be [Tq,Tk] or [B,Tq,Tk]")
            if pair_mask.shape[0] == 1 and query.shape[0] > 1:
                pair_mask = pair_mask.expand(query.shape[0], -1, -1)
            if pair_mask.shape != (query.shape[0], query.shape[1], key_value.shape[1]):
                raise ValueError(
                    f"pair_mask shape {tuple(pair_mask.shape)} does not match "
                    f"{(query.shape[0], query.shape[1], key_value.shape[1])}"
                )
            valid = valid & pair_mask[:, None, :, :].bool()
        if self.causal:
            valid = valid & (kv_time[:, None, None, :] <= q_time[:, None, :, None])

        # Avoid NaNs when an entire sample has no valid text token.
        any_valid = valid.any(dim=-1, keepdim=True)
        safe_logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        safe_logits = torch.where(any_valid, safe_logits, torch.zeros_like(safe_logits))
        weights = torch.softmax(safe_logits, dim=-1)
        weights = torch.where(any_valid, weights, torch.zeros_like(weights))
        weights = self.attn_dropout(weights)

        context = torch.matmul(weights, v)
        context = context.transpose(1, 2).contiguous().view(query.shape)
        context = self.out_proj(context)

        # A missing modality must be an exact no-op.  In context-only mode the
        # aligned feature is exactly zero; in residual mode the original query
        # is returned unchanged.
        query_has_valid_key = any_valid.any(dim=1).squeeze(-1)  # [B, Tq]
        context = torch.where(
            query_has_valid_key.unsqueeze(-1), context, torch.zeros_like(context)
        )
        attn = weights.mean(dim=1) if return_attention else None
        if context_only:
            return context, attn

        x = query + context
        x = x + self.ffn(self.norm2(x))
        x = torch.where(query_has_valid_key.unsqueeze(-1), x, query)
        return x, attn


class SoftGatedCrossModalFusion(nn.Module):
    """Soft-gated fusion used only by Stage-1 real-time text injection.

    The implementation follows the paper diagram directly:

        Q = H_pv W_q
        K,V = H_text W_k,W_v               (performed by CTCA upstream)
        H_align = Attention(Q,K,V)
        G = sigmoid(MLP(H_pv, H_align, g_text))
        H_fused = LayerNorm(H_pv + G * H_align)

    ``valid_mask`` guarantees an exact no-op when no text is available.
    The aligned branch is normalized before gating so its numerical scale cannot
    explode relative to the numerical backbone.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float,
        gate_bias: float = -2.0,
    ):
        super().__init__()
        self.base_norm = nn.LayerNorm(d_model)
        self.align_norm = nn.LayerNorm(d_model)
        self.global_norm = nn.LayerNorm(d_model)
        self.align_projection = nn.Linear(d_model, d_model, bias=False)
        self.gate_network = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.output_norm = nn.LayerNorm(d_model)

        # Start with a conservative, nearly closed gate.  The residual forecast
        # head is zero-initialized separately, so the predecessor prediction is
        # still exactly preserved at epoch 0.
        last = self.gate_network[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        nn.init.constant_(last.bias, float(gate_bias))

    def forward(
        self,
        base: torch.Tensor,
        aligned: torch.Tensor,
        global_text: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if base.shape != aligned.shape:
            raise ValueError(
                f"base/aligned shape mismatch: {tuple(base.shape)} vs {tuple(aligned.shape)}"
            )
        if global_text.ndim != 2 or global_text.shape != (base.shape[0], base.shape[2]):
            raise ValueError(
                f"global_text must be [B,D], got {tuple(global_text.shape)}"
            )
        if valid_mask.ndim == 1:
            valid_mask = valid_mask[:, None].expand(-1, base.shape[1])
        if valid_mask.shape != base.shape[:2]:
            raise ValueError(
                f"valid_mask must be [B,T], got {tuple(valid_mask.shape)}"
            )

        base_n = self.base_norm(base)
        aligned_n = self.align_projection(self.align_norm(aligned))
        global_n = self.global_norm(global_text).unsqueeze(1).expand(-1, base.shape[1], -1)
        gate_logits = self.gate_network(torch.cat([base_n, aligned_n, global_n], dim=-1))
        gate = torch.sigmoid(gate_logits)
        valid = valid_mask.unsqueeze(-1).to(gate.dtype)
        gate = gate * valid
        injected = gate * aligned_n
        fused = self.output_norm(base + injected)

        # LayerNorm(base) would not be an exact no-op for missing text, so force
        # the original base representation back for invalid queries.
        fused = torch.where(valid_mask.unsqueeze(-1), fused, base)
        injected = torch.where(valid_mask.unsqueeze(-1), injected, torch.zeros_like(injected))
        return fused, gate, injected


class ModalityFusionTransformer(nn.Module):
    """Residual, non-gated fusion with stage-safe modality projections.

    Each modality is normalized and projected independently before summation.
    This avoids the cross-modality coupling caused by a LayerNorm over one large
    concatenated vector.  A newly introduced modality projection can therefore
    be initialized to exactly zero while the inherited fusion path remains
    unchanged.
    """

    def __init__(
        self,
        d_model: int,
        n_modalities: int,
        n_heads: int,
        d_ff: int,
        layers: int,
        dropout: float,
    ):
        super().__init__()
        self.n_modalities = int(n_modalities)
        self.base_norm = nn.LayerNorm(d_model)
        self.context_norms = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(self.n_modalities)]
        )
        self.base_projection = nn.Linear(d_model, d_ff)
        self.context_projections = nn.ModuleList(
            [nn.Linear(d_model, d_ff, bias=False) for _ in range(self.n_modalities)]
        )
        self.merge = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)

    def zero_context_projection(self, index: int) -> None:
        if not 0 <= index < self.n_modalities:
            raise IndexError(index)
        nn.init.zeros_(self.context_projections[index].weight)

    def forward(self, base: torch.Tensor, contexts: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(contexts) != self.n_modalities:
            raise ValueError(f"expected {self.n_modalities} contexts, got {len(contexts)}")
        hidden = self.base_projection(self.base_norm(base))
        for norm, projection, context in zip(
            self.context_norms, self.context_projections, contexts
        ):
            hidden = hidden + projection(norm(context))
        fused = base + self.merge(hidden)
        return self.norm(self.encoder(fused))


class EnhancedITransformer(nn.Module):
    """
    iTransformer-based backbone with seasonal/trend variable-token encoders and
    a full future-horizon decoder.

    Inputs
    ------
    x: [B, seq_len, input_dim]
    future_time: [B, pred_len, time_dim]

    Outputs
    -------
    prediction: [B, pred_len]
    hidden: dictionary used by NWP/text fusion stages.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        input_dim: int,
        time_dim: int = 6,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 4,
        d_layers: int = 3,
        d_ff: int = 2048,
        dropout: float = 0.1,
        ma_kernel: int = 25,
        use_revin: bool = True,
        target_index: int = 0,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.input_dim = int(input_dim)
        self.target_index = int(target_index)
        self.use_revin = bool(use_revin)

        self.revin = RevIN(input_dim) if use_revin else None
        self.decomposition = SeriesDecomposition(ma_kernel)

        self.seasonal_embed = nn.Linear(seq_len, d_model)
        self.trend_embed = nn.Linear(seq_len, d_model)
        self.variable_embedding = nn.Parameter(torch.empty(1, input_dim, d_model))
        self.stream_embedding = nn.Parameter(torch.empty(1, 2, d_model))
        nn.init.trunc_normal_(self.variable_embedding, std=0.02)
        nn.init.trunc_normal_(self.stream_embedding, std=0.02)
        self.register_buffer(
            "active_variable_mask",
            torch.ones(input_dim, dtype=torch.bool),
            persistent=True,
        )

        self.seasonal_encoder = InvertedEncoder(e_layers, d_model, n_heads, d_ff, dropout)
        self.trend_encoder = InvertedEncoder(e_layers, d_model, n_heads, d_ff, dropout)
        self.memory_projection = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.history_projection = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.future_time_projection = nn.Sequential(
            nn.Linear(time_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.horizon_embedding = nn.Parameter(torch.empty(1, pred_len, d_model))
        nn.init.trunc_normal_(self.horizon_embedding, std=0.02)

        self.decoder = HorizonDecoder(d_layers, d_model, n_heads, d_ff, dropout)
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff // 2, 1),
        )

    def set_active_variables(self, active: Optional[torch.Tensor]) -> None:
        if active is None:
            self.active_variable_mask.fill_(True)
            return
        active = active.to(device=self.active_variable_mask.device, dtype=torch.bool)
        if active.numel() != self.input_dim:
            raise ValueError(f"expected {self.input_dim} active flags, got {active.numel()}")
        active = active.flatten()
        active[self.target_index] = True
        self.active_variable_mask.copy_(active)

    def _variable_key_padding_mask(self, batch_size: int) -> Optional[torch.Tensor]:
        active = self.active_variable_mask[: self.input_dim].bool().clone()
        active[self.target_index] = True
        if bool(active.all()):
            return None
        return (~active).unsqueeze(0).expand(batch_size, -1)

    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.shape[1] != self.seq_len:
            raise ValueError(f"expected seq_len={self.seq_len}, got {x.shape[1]}")
        if x.shape[2] != self.input_dim:
            raise ValueError(f"expected input_dim={self.input_dim}, got {x.shape[2]}")

        if self.revin is not None:
            x_norm, stats = self.revin.normalize(x)
        else:
            x_norm, stats = x, {}

        seasonal, trend = self.decomposition(x_norm)
        seasonal_tokens = self.seasonal_embed(seasonal.transpose(1, 2))
        trend_tokens = self.trend_embed(trend.transpose(1, 2))
        variable_embedding = self.variable_embedding[:, : self.input_dim]
        seasonal_tokens = seasonal_tokens + variable_embedding + self.stream_embedding[:, :1]
        trend_tokens = trend_tokens + variable_embedding + self.stream_embedding[:, 1:2]

        variable_key_padding_mask = self._variable_key_padding_mask(x.shape[0])
        seasonal_tokens = self.seasonal_encoder(
            seasonal_tokens, key_padding_mask=variable_key_padding_mask
        )
        trend_tokens = self.trend_encoder(
            trend_tokens, key_padding_mask=variable_key_padding_mask
        )
        memory = self.memory_projection(torch.cat([seasonal_tokens, trend_tokens], dim=-1))

        target_seasonal = seasonal_tokens[:, self.target_index]
        target_trend = trend_tokens[:, self.target_index]
        history_state = self.history_projection(
            torch.cat([target_seasonal, target_trend], dim=-1)
        )
        return {
            "seasonal_tokens": seasonal_tokens,
            "trend_tokens": trend_tokens,
            "variable_memory": memory,
            "variable_key_padding_mask": variable_key_padding_mask,
            "history_state": history_state,
            "revin_stats": stats,
        }

    def decode_horizon(
        self,
        encoded: Dict[str, torch.Tensor],
        future_time: torch.Tensor,
    ) -> torch.Tensor:
        if future_time.shape[1] != self.pred_len:
            raise ValueError(f"expected pred_len={self.pred_len}, got {future_time.shape[1]}")
        history = encoded["history_state"].unsqueeze(1)
        queries = (
            history
            + self.future_time_projection(future_time)
            + self.horizon_embedding[:, : future_time.shape[1]]
        )
        return self.decoder(
            queries,
            encoded["variable_memory"],
            memory_key_padding_mask=encoded.get("variable_key_padding_mask"),
        )

    def forward(
        self,
        x: torch.Tensor,
        future_time: torch.Tensor,
        return_hidden: bool = False,
    ):
        encoded = self.encode(x)
        horizon_state = self.decode_horizon(encoded, future_time)
        prediction_norm = self.output_head(horizon_state).squeeze(-1)
        if self.revin is not None:
            prediction = self.revin.denormalize_target(
                prediction_norm, encoded["revin_stats"], self.target_index
            )
        else:
            prediction = prediction_norm

        hidden = {
            **encoded,
            "future_queries": horizon_state,
            "horizon_state": horizon_state,
            "base_prediction": prediction,
            "base_prediction_norm": prediction_norm,
        }
        if return_hidden:
            return prediction, hidden
        return prediction


class HorizonResidualHead(nn.Module):
    """Deep horizon-wise residual regressor with optional exact-zero output."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float,
        zero_init_output: bool = False,
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_ff // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output = nn.Linear(d_ff // 2, 1)
        if zero_init_output:
            self.zero_output()

    def zero_output(self) -> None:
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.features(x)).squeeze(-1)
