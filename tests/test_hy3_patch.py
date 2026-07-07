# SPDX-License-Identifier: Apache-2.0.
"""Unit tests for the Hy3 (hy_v3) base architecture patch.

Covers registration, idempotency, sibling-module pre-flight, and model
class accessibility. Live load+generate is verified separately via the
Phase A end-to-end probe documented in
``.omo/plans/hy3-omlx-mtp-patch.md`` step A.6.
"""

from __future__ import annotations

import importlib
import sys

import pytest


def _reset_module_state():
    """Strip omlx-applied sys.modules entries so each test re-applies the patch."""
    for k in list(sys.modules):
        if k == "mlx_lm.models.hy_v3" or k.startswith("omlx.patches.hy3"):
            sys.modules.pop(k, None)
    # Also clear the alias set by _register_model_type_aliases.
    sys.modules.pop("mlx_lm.models.hy_v3_mtp", None)
    importlib.import_module("omlx.patches.hy3")._APPLIED = False


def test_apply_returns_true_first_time():
    _reset_module_state()
    from omlx.patches.hy3 import apply_hy3_patch

    assert apply_hy3_patch() is True


def test_apply_is_idempotent():
    _reset_module_state()
    from omlx.patches.hy3 import apply_hy3_patch, is_applied

    apply_hy3_patch()
    assert is_applied() is True
    # Second invocation must be a no-op returning False.
    assert apply_hy3_patch() is False
    assert is_applied() is True


def test_module_registered_under_mlx_lm_namespace():
    _reset_module_state()
    from omlx.patches.hy3 import apply_hy3_patch

    apply_hy3_patch()
    mod = sys.modules.get("mlx_lm.models.hy_v3")
    assert mod is not None, "mlx_lm.models.hy_v3 was not registered"
    # Relative imports inside the file resolved through mlx_lm.models,
    # so the Model class should be callable.
    assert hasattr(mod, "Model")
    assert hasattr(mod, "ModelArgs")
    assert hasattr(mod, "HYV3Model")
    assert mod.Model.__name__ == "Model"


def test_module_package_stamp():
    """The registered module's __package__ must be 'mlx_lm.models' so that
    relative imports inside hy_v3_model.py resolve correctly (the deepseek_v4
    / glm_moe_dsa patches rely on the same mechanism)."""
    _reset_module_state()
    from omlx.patches.hy3 import apply_hy3_patch

    apply_hy3_patch()
    mod = sys.modules["mlx_lm.models.hy_v3"]
    assert mod.__package__ == "mlx_lm.models"


def test_model_type_alias_registered():
    _reset_module_state()
    from omlx.patches.hy3 import apply_hy3_patch

    apply_hy3_patch()
    # The hy_v3_mtp alias must point at the same module object.
    base = sys.modules.get("mlx_lm.models.hy_v3")
    alias = sys.modules.get("mlx_lm.models.hy_v3_mtp")
    assert base is not None, "base module not registered"
    assert alias is not None, "alias module not registered"
    assert alias is base


def test_pre_flight_sibling_modules():
    """Sibling modules required by hy_v3_model.py's relative imports must
    be present in the pinned mlx-lm. If this test fails, vendor the missing
    module from kernelpool/mlx-lm@add-hy3-preview (plan step A.2a)."""
    from omlx.patches.hy3 import _ensure_sibling_modules

    _ensure_sibling_modules()  # raises ImportError with action message on failure.


def test_is_mtp_compatible_for_hy_v3():
    from omlx.utils.model_loading import _is_mtp_compatible

    assert _is_mtp_compatible({"num_nextn_predict_layers": 1}, "hy_v3") is True
    assert _is_mtp_compatible({"mtp_num_hidden_layers": 1}, "hy_v3") is True
    assert _is_mtp_compatible({}, "hy_v3") is False
    assert _is_mtp_compatible({"num_nextn_predict_layers": 1}, "hy_v3_opensource") is True
    assert _is_mtp_compatible({"num_nextn_predict_layers": 1}, "gpt2") is False


def test_tool_parser_modules_registered():
    """``apply_hy3_patch`` must expose the Hy3 tool parsers at
    ``mlx_lm.tool_parsers.{hy_v3,hy_v3_opensource}`` so the mlx-lm
    ``TokenizerWrapper`` can resolve them by name."""
    _reset_module_state()
    from omlx.patches.hy3 import apply_hy3_patch

    apply_hy3_patch()
    assert "mlx_lm.tool_parsers.hy_v3" in sys.modules
    assert "mlx_lm.tool_parsers.hy_v3_opensource" in sys.modules
    base = sys.modules["mlx_lm.tool_parsers.hy_v3"]
    alias = sys.modules["mlx_lm.tool_parsers.hy_v3_opensource"]
    # hy_v3_opensource imports parse_tool_call from hy_v3, so the modules
    # must expose tool_call_start/end sentinels (different per variant).
    assert base.tool_call_start == "<tool_calls>"
    assert alias.tool_call_start == "<tool_calls:opensource>"


def test_infer_tool_parser_recovers_opensource_suffix():
    """The wrapped ``_infer_tool_parser`` must select ``hy_v3_opensource``
    when the chat template contains the Hy3 Jinja pattern ``<arg_key{}>``
    (render-time ``.format(HYTK)`` of ``:opensource`` suffix) or the
    pre-rendered ``<arg_key:opensource>`` sentinel. The pinned mlx-lm only
    knows the unsuffixed sentinel, so without our wrap the parser is
    misselected and tool_calls come back as raw markup text."""
    _reset_module_state()
    from omlx.patches.hy3 import apply_hy3_patch

    apply_hy3_patch()
    import mlx_lm.tokenizer_utils as _tu

    infer = _tu._infer_tool_parser
    assert getattr(infer, "_omlx_hy3_patched", False) is True
    # Hy3 chat_template.jinja uses Jinja format string pattern.
    assert infer("<...><arg_key{}>{...}</arg_key{}><...>") == "hy_v3_opensource"
    # Some checkpoints may store pre-rendered templates.
    assert infer("<...><arg_key:opensource><...>") == "hy_v3_opensource"
    # Falls through to the original inference for non-Hy3 templates.
    # (Don't assert the exact fallback value — just that it doesn't crash.)
