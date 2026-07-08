# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: N803, N806
"""Route Qwen3.5/3.6 Gated DeltaNet prefill to an optimized Metal kernel.

Default route: ``gated_delta_blocked_seq`` — the exact sequential recurrence
restructured for Apple GPUs (threadgroup-staged k/q/v blocks, register-resident
state, Dv/32 split). ~2x faster than mlx_lm's stock sequential kernel at 16k
(14.9ms vs 29.7ms per layer call) with fp32-exact state (rel-err ~5e-8).

Optional route (``OMLX_GDN_IMPL=chunked``): the FLA chunked WY-representation
kernels — accuracy-validated but slower than the stock kernel E2E; kept for
future iteration.

This rebinds ``gated_delta_update`` in ``mlx_vlm.models.qwen3_5.language`` for
scalar-gated prefill with T >= OMLX_GDN_MIN_T. Decode (T==1) and
masked/vectorized paths keep the original kernel.

Toggles:
  OMLX_GDN_KERNEL=0    disable the patch entirely
  OMLX_GDN_IMPL=...    blocked_seq (default) | chunked
  OMLX_GDN_BLOCK_T=N   blocked_seq time block: 16 | 32 | 48 (default 32)
  OMLX_GDN_MIN_T=N     minimum prefill length to engage (default 64)
  OMLX_GDN_FUSED_G_BETA=1
                         use mlx-vlm's Metal g/beta precompute helper
  OMLX_GDN_STUB=1      debug only: skip the GDN op to measure its E2E share
"""

import logging
import os

import mlx.core as mx

logger = logging.getLogger(__name__)

_PATCHED = False


def apply_qwen35_gdn_prefill_patch() -> bool:
    global _PATCHED
    if _PATCHED:
        return True
    if os.environ.get("OMLX_GDN_KERNEL", "1") == "0":
        return False
    if not mx.metal.is_available():
        return False

    try:
        from mlx_vlm.models.qwen3_5 import gated_delta as gd
        from mlx_vlm.models.qwen3_5 import language as lang
    except ImportError:
        logger.debug("mlx_vlm qwen3_5 not importable; GDN prefill patch skipped")
        return False

    min_t = int(os.environ.get("OMLX_GDN_MIN_T", "64"))
    fused_g_beta = os.environ.get("OMLX_GDN_FUSED_G_BETA", "0") == "1"
    stub = os.environ.get("OMLX_GDN_STUB", "0") == "1"
    original = gd.gated_delta_update

    # Resolve g/beta compute helpers across mlx_vlm API revisions.
    #
    # - Older mlx_vlm exposed ``gd._compute_g_beta(A_log, a, b, dt_bias)`` returning
    #   ``(g, beta)`` and optionally ``gd._compute_g_beta_prefill`` for the fused
    #   Metal path. Patching that contract worked up to mlx_vlm ~0.4.
    # - mlx_vlm 0.5+ imports ``compute_g`` from ``mlx_lm.models.gated_delta``
    #   and computes ``beta = mx.sigmoid(b)`` inline at every call site. The
    #   ``_compute_g_beta*`` attributes are gone, so the previous patch path
    #   raised AttributeError mid-prefill and surfaced as empty SSE responses.
    #
    # Fast path preference order (matches the production contract of
    # ``gated_delta_blocked_seq`` / ``gated_delta_chunked_metal`` which both
    # expect ``g`` and ``beta`` of shape ``(B, T, Hv)`` fp32):
    #   1. ``gd._compute_g_beta_prefill`` (legacy fused Metal helper)
    #   2. ``gd._compute_g_beta`` (legacy 4-arg tuple helper)
    #   3. ``mlx_lm.models.gated_delta.compute_g`` + ``mx.sigmoid(b)`` (new API)
    #   4. ``mlx_vlm.models.qwen3_5.gated_delta.compute_g`` (re-exported)
    #
    # If none of the above are resolvable we log a warning and skip the patch
    # entirely so the stock mlx_vlm kernel stays in effect.
    compute_g_beta_prefill = getattr(gd, "_compute_g_beta_prefill", None)
    compute_g_beta = getattr(gd, "_compute_g_beta", None)
    new_api_compute_g = getattr(gd, "compute_g", None)
    if new_api_compute_g is None:
        try:
            from mlx_lm.models.gated_delta import compute_g as new_api_compute_g
        except ImportError:
            new_api_compute_g = None

    if compute_g_beta_prefill is None and compute_g_beta is None and new_api_compute_g is None:
        logger.warning(
            "Qwen3.5/3.6 GDN prefill patch disabled: no _compute_g_beta[_prefill] "
            "and no mlx_lm.gated_delta.compute_g available on this mlx_vlm/mlx_lm "
            "build; falling back to stock kernel"
        )
        return False

    def _compute_g_beta(A_log, a, b, dt_bias):
        if compute_g_beta_prefill is not None:
            return compute_g_beta_prefill(A_log, a, b, dt_bias)
        if compute_g_beta is not None:
            return compute_g_beta(A_log, a, b, dt_bias)
        # New mlx_vlm/mlx_lm API: g = compute_g(A_log, a, dt_bias), beta = sigmoid(b)
        g = new_api_compute_g(A_log, a, dt_bias)
        beta = mx.sigmoid(b)
        return g, beta

    from omlx.custom_kernels.qwen35_prefill import (
        gated_delta_blocked_seq,
        gated_delta_chunked_metal,
    )

    impl = os.environ.get("OMLX_GDN_IMPL", "blocked_seq")
    fast_prefill = (
        gated_delta_chunked_metal if impl == "chunked" else gated_delta_blocked_seq
    )

    def gated_delta_update_metal(
        q, k, v, a, b, A_log, dt_bias, state=None, mask=None, use_kernel=True
    ):
        # Debug-only: skip the GDN op entirely to measure its E2E share.
        # Output is garbage; never enable outside profiling.
        if stub and q.shape[1] > 1:
            if state is None:
                B_, Hv_, Dv_ = v.shape[0], v.shape[-2], v.shape[-1]
                state = mx.zeros((B_, Hv_, Dv_, q.shape[-1]), dtype=mx.float32)
            return v, state
        if (
            use_kernel
            and mask is None
            and q.shape[1] >= min_t
            and q.shape[-1] % 16 == 0
            and v.shape[-1] % 32 == 0
            and a.ndim == 3  # scalar per-head gating
        ):
            if fused_g_beta and compute_g_beta_prefill is not None:
                g, beta = compute_g_beta_prefill(A_log, a, b, dt_bias)
            else:
                g, beta = _compute_g_beta(A_log, a, b, dt_bias)
            return fast_prefill(q, k, v, g, beta, state)
        return original(
            q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel=use_kernel
        )

    lang.gated_delta_update = gated_delta_update_metal
    gd.gated_delta_update = gated_delta_update_metal
    _PATCHED = True
    logger.info(
        "Qwen3.5/3.6 GDN prefill kernel patch applied (Metal, impl=%s, min_t=%d)",
        impl, min_t,
    )
    return True


def apply_qwen35_gdn_chunked_patch() -> bool:
    """Backward-compatible name for older callers/configuration."""
    return apply_qwen35_gdn_prefill_patch()
