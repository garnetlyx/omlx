# SPDX-License-Identifier: Apache-2.0
"""Monkey-patch for ml-explore/mlx-lm PR #1485 — Tencent Hy3 native MTP.

PR #1485 is stacked on PR #1211 (kernelpool, base architecture). It keeps
and uses the Multi-Token-Prediction (MTP) layer for self-speculative
decoding rather than stripping it at load time.

oMLX already injects the Hy3 base model via ``omlx/patches/hy3/`` which
lands the model class into ``sys.modules['mlx_lm.models.hy_v3']``; this
patch sits on top and adds the MTP head + ``mtp_forward`` /
``make_mtp_cache``. Apply order: caller (``patches/mlx_lm_mtp/__init__.py``)
runs ``apply()`` after the base Hy3 patch has registered the module.

Hy3's MTP is structurally simpler than DeepSeek-V4's:
- Hidden states are **3D** (``B, S, H``), no Hyper-head broadcasting,
  no 4D adapter needed.
- Backbone attention uses standard grouped-query attention with QK-norm,
  no PoolingCache / SparseCompressedAttention; only ``RotatingKVCache``.
- One MTP layer total (``num_nextn_predict_layers=1``).
- The MTP block is one ``DecoderLayer`` wrapped with
  ``enorm`` / ``hnorm`` / ``eh_proj`` / ``final_layernorm`` (already
  defined as ``MTPBlock`` in the vendored ``hy_v3_model.py``).
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _is_our_method(cls: Any, attr: str, marker: str) -> bool:
    """True iff ``cls.<attr>`` carries our marker. Mirrors qwen35_model
    / deepseek_v4_model — used for self-healing idempotency so another
    patch clobbering ``__call__`` between two Hy3 loads doesn't leave
    the class stuck on the wrong signature (issue #1388)."""
    existing = cls.__dict__.get(attr)
    return getattr(existing, marker, False)


def apply() -> bool:
    """Apply PR #1485 model-side patches when the Hy3 base patch is active.

    Self-healing: re-runs sub-patches when class state has drifted
    (mirrors qwen35_model.apply / deepseek_v4_model.apply).
    """
    hy3 = sys.modules.get("mlx_lm.models.hy_v3")
    if hy3 is None or not hasattr(hy3, "Model"):
        logger.debug(
            "Hy3 module not registered; skipping MTP patch (this is "
            "expected for non-Hy3 models)"
        )
        return False

    _patch_model_args(hy3)
    _ensure_mtp_block(hy3)
    _patch_hy_v3_model_call(hy3)
    _patch_model(hy3)

    if not hasattr(hy3.Model, "_omlx_mtp_patched"):
        hy3.Model._omlx_mtp_patched = "patch"
        logger.info("Hy3 MTP model patch applied (PR #1485)")
    return True


# ---------------------------------------------------------------------------
# ModelArgs — surface num_nextn_predict_layers (already in the dataclass).
# ---------------------------------------------------------------------------

def _patch_model_args(hy3: Any) -> None:
    """No-op for Hy3 — ``ModelArgs`` already declares
    ``num_nextn_predict_layers`` as a real dataclass field (default 0).

    Mirrors the structural slot used by the qwen35 patch for symmetry;
    kept as a placeholder in case a future Hy3 revision needs the same
    attribute-on-instance trick Qwen3.5 uses.
    """
    args_cls = hy3.ModelArgs
    if hasattr(args_cls, "_omlx_mtp_args_patched"):
        return
    args_cls._omlx_mtp_args_patched = True


# ---------------------------------------------------------------------------
# MTPBlock — the vendored hy_v3_model.py already defines it. Re-use verbatim.
# ---------------------------------------------------------------------------

def _ensure_mtp_block(hy3: Any) -> None:
    """Confirm ``MTPBlock`` is exposed on the module. The vendored
    ``hy_v3_model.py`` (PR #1485) already defines it; we just verify
    it is accessible and leave a debug breadcrumb if not."""
    if hasattr(hy3, "MTPBlock"):
        return
    raise RuntimeError(
        "Hy3 base patch registered mlx_lm.models.hy_v3 but MTPBlock is "
        "missing — the vendored hy_v3_model.py is likely the PR #1211 "
        "(MTP-stripping) variant rather than the PR #1485 (MTP-keeping) "
        "variant. Re-vendor from eauchs/mlx-lm@hy_v3-mtp."
    )


# ---------------------------------------------------------------------------
# HYV3Model — return_hidden support (backbone __call__).
# ---------------------------------------------------------------------------

def _patch_hy_v3_model_call(hy3: Any) -> None:
    """Wrap ``HYV3Model.__call__`` to optionally return pre-norm hidden.

    The vendored PR #1485 source already supports ``return_hidden_states``
    on ``HYV3Model.__call__`` and ``Model.__call__``. We re-expose the
    same path under the oMLX-conventional kwarg name ``return_hidden``
    so the BatchGenerator MTP dispatch (which calls
    ``model(inputs, cache=..., return_hidden=True, n_confirmed=...)``)
    finds a uniform interface across Qwen3.5 / DeepSeek-V4 / Hy3.
    """
    cls = hy3.HYV3Model
    if _is_our_method(cls, "__call__", "_omlx_mtp_call_marker"):
        return

    original_call = cls.__call__

    def __call__(
        self,
        x,
        cache: Optional[Any] = None,
        return_hidden: bool = False,
        return_hidden_states: bool = False,
        **kwargs,
    ):
        out, h = original_call(
            self, x, cache=cache, return_hidden_states=True, **kwargs
        )
        if return_hidden or return_hidden_states:
            return out, h
        return out

    __call__._omlx_mtp_call_marker = True
    cls.__call__ = __call__


# ---------------------------------------------------------------------------
# Model — wrap __init__, replace __call__, add mtp_forward / make_mtp_cache,
# extend sanitize to keep mtp.* keys when an MTP head is attached.
# ---------------------------------------------------------------------------

def _patch_model(hy3: Any) -> None:
    cls = hy3.Model
    init_wrapped = getattr(cls, "_omlx_mtp_init_wrapped", False)
    call_owned = _is_our_method(cls, "__call__", "_omlx_mtp_call_marker")
    if init_wrapped and call_owned:
        return

    import mlx.core as mx
    from mlx_lm.models.cache import KVCache
    from mlx_lm.models.base import create_attention_mask

    MTPBlock = hy3.MTPBlock
    original_init = cls.__init__
    original_sanitize = cls.sanitize

    def __init__(self, args):
        original_init(self, args)
        n_mtp = int(getattr(args, "num_nextn_predict_layers", 0) or 0)
        # Gated on the MTP active-flag so mtp_enabled=False produces a
        # model indistinguishable from stock. See qwen35_model._patch_model
        # / deepseek_v4_model._patch_model for the same pattern.
        from . import is_mtp_active

        mtp_decode_enabled = bool(n_mtp > 0 and is_mtp_active())
        self._omlx_mtp_decode_enabled = mtp_decode_enabled
        # PR #1485's original_init attaches ``self.mtp`` (a single
        # MTPBlock) whenever ``num_nextn_predict_layers > 0``. We keep it
        # attached even when ``mtp_decode_enabled=False`` so sanitize
        # preserves ``mtp.*`` weights (decode eligibility is gated by
        # ``_omlx_mtp_decode_enabled`` at decode time, not by ``self.mtp``
        # presence — see batch_generator._model_has_mtp_module).
        #
        # Hy3's checkpoint stores mtp weights as ``mtp.<field>`` (no list
        # index), so ``self.mtp`` must stay a single MTPBlock, not a list.

    def __call__(
        self,
        inputs,
        cache: Optional[Any] = None,
        return_hidden: bool = False,
        n_confirmed: int = 0,
    ):
        # ``n_confirmed`` is part of the patched-backbone interface that
        # ``batch_generator._call_backbone`` passes during MTP verify
        # cycles. It only matters for models with module-level recurrent
        # state (Qwen3.5's GatedDeltaNet snapshots conv/ssm state after
        # the confirmed prefix). Hy3 keeps all decode state in cache
        # objects (RotatingKVCache) and draft rejection rolls back via
        # ``cache.trim`` in ``_restore_or_trim_caches``, so the argument
        # is accepted and unused.
        out, h = self.model(
            inputs, cache=cache, return_hidden_states=True
        )
        logits = self._logits(out)
        if return_hidden:
            return logits, h
        return logits

    def make_mtp_cache(self) -> Optional[List[Any]]:
        """Build MTP block caches. Hy3's MTP head is a single
        ``DecoderLayer`` with standard grouped-query attention. The backbone
        uses unbounded ``KVCache`` (no sliding window — Hy3 has full
        attention), so the MTP head matches that layout. ``KVCache`` carries
        the ``trim()`` method that ``cache_rollback.py`` relies on for
        draft-rejection rollback. Returns a 1-element list so the
        BatchGenerator's iteration contract is satisfied."""
        if not hasattr(self, "mtp"):
            return None
        return [KVCache()]

    def mtp_forward(
        self,
        h: Any,
        input_ids: Any,
        cache: Optional[List[Any]] = None,
    ) -> Any:
        """Run the MTP head to draft next-token logits from a hidden state.

        Mirrors the ``MTPBlock.__call__`` contract from the vendored
        ``hy_v3_model.py``: fuses ``enorm(embed(input_ids))`` with
        ``hnorm(h)``, projects through ``eh_proj``, runs one
        ``DecoderLayer``, applies ``final_layernorm``, then the shared
        ``lm_head`` / embed-as-linear to produce logits.
        """
        if not hasattr(self, "mtp"):
            raise ValueError("MTP is not enabled or its weights are not loaded.")
        if cache is None:
            cache = [None]

        # Embed the proposed next token. input_ids shape (B, S) with S=1
        # during draft.
        e = self.model.embed_tokens(input_ids)

        # Build attention mask from the embedded input + first cache slot.
        first_cache = cache[0]
        mask = create_attention_mask(e, first_cache)

        # Hy3's self.mtp is a single MTPBlock (not a list); call directly.
        # cache[0] is the KVCache for the MTP block's internal DecoderLayer.
        h_mtp = self.mtp(h, e, mask=mask, cache=first_cache)

        return self._logits(h_mtp)

    def sanitize(self, weights: Dict[str, Any]) -> Dict[str, Any]:
        """Combined sanitize: base PR #1211/#1485 remap + mtp.* retention.

        The vendored ``Model.sanitize`` (from PR #1485) already:
        - remaps ``model.layers.{n_layers}.{enorm|hnorm|eh_proj|...}.*``
          to ``mtp.*`` or ``mtp.layer.*``
        - moves ``mlp.expert_bias`` → ``mlp.router.expert_bias``
        - stacks per-expert tensors into ``mlp.switch_mlp.*``
        - drops ``lm_head.weight`` if tied

        We delegate to that, then add a graceful-fallback: if the model
        has ``self.mtp`` but the checkpoint has no ``mtp.*`` weights
        (e.g. a quantized variant that stripped them), delete the MTP
        head and ``_omlx_mtp_decode_enabled`` so BatchGenerator falls
        back to standard decoding.
        """
        has_mtp = hasattr(self, "mtp")
        has_mtp_weights = any(k.startswith("mtp.") for k in weights)

        if has_mtp and not has_mtp_weights:
            try:
                del self.mtp
            except AttributeError:
                pass
            self._omlx_mtp_decode_enabled = False
            logger.warning(
                "Hy3 model had MTP head attached but checkpoint ships no "
                "mtp.* weights — disabling MTP decode (will use standard "
                "decoding)."
            )

        return original_sanitize(self, weights)

    if not init_wrapped:
        cls.__init__ = __init__
        cls._omlx_mtp_init_wrapped = True
    __call__._omlx_mtp_call_marker = True
    cls.__call__ = __call__
    cls.mtp_forward = mtp_forward
    cls.make_mtp_cache = make_mtp_cache
    cls.sanitize = sanitize
