# SPDX-License-Identifier: Apache-2.0.
"""Unit tests for the Hy3 (hy_v3) MTP model-side patch.

Covers hook attachment (``mtp_forward`` / ``make_mtp_cache`` /
``_omlx_mtp_decode_enabled``), idempotency, self-healing, and the
no-op behavior when the base Hy3 patch has not registered the module.
"""

from __future__ import annotations

import importlib
import sys

import pytest


def _ensure_base_patch_applied():
    """Apply the base Hy3 patch if not already applied. Each test calls this
    first because pytest test ordering is not guaranteed and previous tests
    may have popped mlx_lm.models.hy_v3 from sys.modules."""
    if "mlx_lm.models.hy_v3" not in sys.modules:
        from omlx.patches.hy3 import apply_hy3_patch
        apply_hy3_patch()


def test_apply_returns_false_when_base_patch_missing():
    """If ``mlx_lm.models.hy_v3`` is not in sys.modules, the MTP patch must
    skip cleanly (mirrors deepseek_v4_model behavior for non-DeepSeek models)."""
    sys.modules.pop("mlx_lm.models.hy_v3", None)
    mod = importlib.reload(importlib.import_module("omlx.patches.mlx_lm_mtp.hy_v3_model"))
    assert mod.apply() is False
    _ensure_base_patch_applied()  # restore for subsequent tests


def test_apply_installs_hooks_when_base_patch_present():
    _ensure_base_patch_applied()
    from omlx.patches.mlx_lm_mtp import hy_v3_model

    hy3 = sys.modules["mlx_lm.models.hy_v3"]

    # Pre-state: clear any hooks installed by previous tests so we exercise
    # the install path; apply() must restore them.
    for attr in (
        "mtp_forward",
        "make_mtp_cache",
        "_omlx_mtp_patched",
        "_omlx_mtp_init_wrapped",
    ):
        try:
            delattr(hy3.Model, attr)
        except AttributeError:
            pass

    assert hy_v3_model.apply() is True
    assert hasattr(hy3.Model, "mtp_forward")
    assert hasattr(hy3.Model, "make_mtp_cache")
    assert getattr(hy3.Model, "_omlx_mtp_patched", None) == "patch"
    assert getattr(hy3.Model, "_omlx_mtp_init_wrapped", False) is True
    # The patched Model.__call__ carries the marker attribute.
    assert getattr(hy3.Model.__call__, "_omlx_mtp_call_marker", False) is True
    # HYV3Model.__call__ must also be marker-stamped.
    assert getattr(hy3.HYV3Model.__call__, "_omlx_mtp_call_marker", False) is True


def test_apply_is_idempotent_and_self_healing():
    _ensure_base_patch_applied()
    from omlx.patches.mlx_lm_mtp import hy_v3_model

    hy3 = sys.modules["mlx_lm.models.hy_v3"]

    hy_v3_model.apply()
    call_first = hy3.Model.__call__
    assert getattr(call_first, "_omlx_mtp_call_marker", False) is True

    # Second apply: no-op (already installed).
    hy_v3_model.apply()
    assert hy3.Model.__call__ is call_first

    # Simulate clobbering by another patch (e.g. dflash overwrite).
    def impostor(self, *args, **kwargs):
        raise RuntimeError("impostor")

    hy3.Model.__call__ = impostor
    assert not getattr(hy3.Model.__call__, "_omlx_mtp_call_marker", False)

    # Self-healing: re-apply restores our marker.
    hy_v3_model.apply()
    assert getattr(hy3.Model.__call__, "_omlx_mtp_call_marker", False) is True


def test_make_mtp_cache_returns_none_when_no_mtp_head():
    _ensure_base_patch_applied()
    from omlx.patches.mlx_lm_mtp import hy_v3_model, set_mtp_active

    hy_v3_model.apply()
    set_mtp_active(False)

    hy3 = sys.modules["mlx_lm.models.hy_v3"]
    args = hy3.ModelArgs.from_dict({
        "model_type": "hy_v3",
        "vocab_size": 32,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "num_experts": 2,
        "num_experts_per_tok": 1,
        "num_shared_experts": 1,
        "expert_hidden_dim": 4,
        "first_k_dense_replace": 1,
        "rms_norm_eps": 1e-5,
        "rope_parameters": {"rope_theta": 10000.0},
    })
    model = hy3.Model(args)
    assert hasattr(model, "_omlx_mtp_decode_enabled")
    assert model._omlx_mtp_decode_enabled is False
    assert not hasattr(model, "mtp")
    assert model.make_mtp_cache() is None


def test_mtp_head_attached_when_active_and_config_declares_layers():
    _ensure_base_patch_applied()
    from omlx.patches.mlx_lm_mtp import hy_v3_model, set_mtp_active

    hy_v3_model.apply()
    set_mtp_active(True)

    hy3 = sys.modules["mlx_lm.models.hy_v3"]
    args = hy3.ModelArgs.from_dict({
        "model_type": "hy_v3",
        "vocab_size": 32,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "num_experts": 2,
        "num_experts_per_tok": 1,
        "num_shared_experts": 1,
        "expert_hidden_dim": 4,
        "first_k_dense_replace": 1,
        "rms_norm_eps": 1e-5,
        "rope_parameters": {"rope_theta": 10000.0},
        "num_nextn_predict_layers": 1,
    })
    model = hy3.Model(args)
    assert model._omlx_mtp_decode_enabled is True
    assert hasattr(model, "mtp")
    # Hy3's checkpoint stores mtp weights without a list index, so self.mtp
    # is a single MTPBlock (not a list). make_mtp_cache wraps it as a
    # 1-element list for BatchGenerator uniformity.
    assert not isinstance(model.mtp, list)
    cache = model.make_mtp_cache()
    assert cache is not None
    assert len(cache) == 1
    set_mtp_active(False)

    hy3 = sys.modules["mlx_lm.models.hy_v3"]
    # Construct a tiny args instance so __init__ can run without a real config.
    # (The patched __init__ reads num_nextn_predict_layers; default 0 from the
    # dataclass means no MTP head attached.)
    args = hy3.ModelArgs.from_dict({
        "model_type": "hy_v3",
        "vocab_size": 32,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "num_experts": 2,
        "num_experts_per_tok": 1,
        "num_shared_experts": 1,
        "expert_hidden_dim": 4,
        "first_k_dense_replace": 1,
        "rms_norm_eps": 1e-5,
        "rope_parameters": {"rope_theta": 10000.0},
    })
    model = hy3.Model(args)
    assert hasattr(model, "_omlx_mtp_decode_enabled")
    assert model._omlx_mtp_decode_enabled is False
    assert not hasattr(model, "mtp")
    assert model.make_mtp_cache() is None


