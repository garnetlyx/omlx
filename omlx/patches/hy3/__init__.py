# SPDX-License-Identifier: Apache-2.0
"""Tencent Hy3 (``hy_v3``) monkey-patch for the pinned mlx-lm.

Vendors the Hy3 architecture model from mlx-lm PR #1485
(https://github.com/ml-explore/mlx-lm/pull/1485) — kept verbatim from
eauchs/mlx-lm branch ``hy_v3-mtp`` HEAD ``a7cc3054`` — so that oMLX can
load Tencent Hy3 / Hunyuan 3 (295B MoE, 21B active, 192 routed experts
top-8, GQA + QK-norm, 256K context) with the Multi-Token-Prediction (MTP)
layer enabled for self-speculative decoding.

PR #1485 is stacked on PR #1211 (kernelpool, base architecture + tool
parsers + tokenizer fixes). PR #1211 strips the MTP layer in
``sanitize()``; PR #1485 keeps and uses it. The vendored
``hy_v3_model.py`` is the #1485 head which contains both.

The MTP runtime integration lives in ``omlx.patches.mlx_lm_mtp.hy_v3_model``
which depends on this base patch registering ``mlx_lm.models.hy_v3`` in
``sys.modules`` first.

Public module name stays ``mlx_lm.models.hy_v3`` so mlx-lm's normal
model loader (``utils._get_classes`` -> ``importlib.import_module``) can
find it. The relative imports inside ``hy_v3_model.py``
(``from .activations import swiglu``, ``from .base import ...``,
``from .pipeline import PipelineMixin``, ``from .rope_utils import
initialize_rope``, ``from .switch_layers import SwitchGLU``) resolve
through ``mlx_lm.models.*`` because the module's ``__package__`` is
stamped to ``"mlx_lm.models"`` at load time (see ``_register_module``).
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Source-of-truth references for the vendored code.
PR_BASE_URL = "https://github.com/ml-explore/mlx-lm/pull/1211"
PR_MTP_URL = "https://github.com/ml-explore/mlx-lm/pull/1485"
# eauchs/mlx-lm branch head SHA at the time of vendoring.
PR_HEAD_SHA = "a7cc3054b1ff48c19950513b422a66cfed7baa60"

_APPLIED = False


def _register_module(qualname: str, file_name: str) -> None:
    """Load a local file as if it were ``qualname``.

    Mirrors ``omlx.patches.deepseek_v4._register_module``: stamps the
    module's ``__package__`` to ``"mlx_lm.models"`` so relative imports
    inside the file (``from .activations import swiglu``, etc.) resolve
    through the real mlx_lm package — *not* through omlx.

    Idempotent.
    """
    if qualname in sys.modules:
        return

    here = Path(__file__).parent
    file_path = here / file_name
    spec = importlib.util.spec_from_file_location(qualname, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create spec for {qualname} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "mlx_lm.models"
    sys.modules[qualname] = module
    spec.loader.exec_module(module)
    logger.info("Registered %s from %s", qualname, file_path.name)


def _register_model_type_aliases() -> None:
    """Map Hy3 variant model_types to the injected base module.

    Some converted checkpoints encode ``model_type=hy_v3_mtp`` or use a
    ``:opensource`` variant suffix. Point those aliases at the same module
    so mlx-lm's ``_get_classes`` resolves them to a single implementation.
    """
    try:
        import mlx_lm.utils as _utils

        remapping = getattr(_utils, "MODEL_REMAPPING", None)
        if isinstance(remapping, dict):
            remapping.setdefault("hy_v3_mtp", "hy_v3")
    except Exception as e:
        logger.debug("Hy3 model_type remapping skipped: %s", e)

    base_module = sys.modules.get("mlx_lm.models.hy_v3")
    if base_module is not None:
        sys.modules.setdefault("mlx_lm.models.hy_v3_mtp", base_module)


def _register_tool_parser_modules() -> None:
    """Expose Hy3 tool parsers to mlx-lm and teach ``_infer_tool_parser`` to
    recognize Hy3's ``:opensource`` chat templates.

    Mirrors ``deepseek_v4/tokenizer_patch.py::apply_load_patch`` step 1-2:
    (1) register vendored parser modules at ``mlx_lm.tool_parsers.*`` so
    mlx-lm's ``TokenizerWrapper`` can import them by name; (2) wrap
    ``_infer_tool_parser`` to detect the Hy3 chat template and select
    ``hy_v3_opensource`` for it.

    Sentinel detection: Hy3's chat template (chat_template.jinja) defines
    tokens via Jinja format strings like ``<arg_key{}>`` where ``{}`` is
    filled by the ``HYTK = ':opensource'`` variable at render time. So the
    raw template contains the literal substring ``<arg_key{}>``, not the
    rendered ``<arg_key:opensource>``. We match the literal Jinja pattern
    so detection works before the template is rendered.

    Idempotent via the ``_omlx_hy3_patched`` marker on the wrapper.
    """
    try:
        import mlx_lm.tokenizer_utils as _tu

        from .tools import hy_v3 as _hy3_tools
        from .tools import hy_v3_opensource as _hy3_os_tools

        sys.modules.setdefault("mlx_lm.tool_parsers.hy_v3", _hy3_tools)
        sys.modules.setdefault(
            "mlx_lm.tool_parsers.hy_v3_opensource", _hy3_os_tools
        )

        orig_infer = getattr(_tu, "_infer_tool_parser", None)
        if orig_infer is not None and not getattr(
            orig_infer, "_omlx_hy3_patched", False
        ):

            def _infer_tool_parser(chat_template):
                # Hy3's chat_template.jinja uses ``HYTK = ':opensource'``
                # variable with ``<arg_key{}>``.format(HYTK) pattern —
                # match the literal Jinja source.
                if (
                    isinstance(chat_template, str)
                    and "<arg_key{}>" in chat_template
                ):
                    return "hy_v3_opensource"
                # Some checkpoints may store the pre-rendered template.
                if (
                    isinstance(chat_template, str)
                    and "<arg_key:opensource>" in chat_template
                ):
                    return "hy_v3_opensource"
                return orig_infer(chat_template)

            _infer_tool_parser._omlx_hy3_patched = True
            _tu._infer_tool_parser = _infer_tool_parser
        logger.info("Hy3 tool parsers registered (hy_v3, hy_v3_opensource)")
    except Exception as e:
        logger.debug("Hy3 tool parser registration skipped: %s", e)


def apply_hy3_patch() -> bool:
    """Apply the Hy3 base architecture patch to mlx-lm. Idempotent.

    Must run *before* ``mlx_lm.load()`` encounters a Hy3 model so that
    ``_get_classes`` can resolve ``mlx_lm.models.hy_v3``.

    Returns ``True`` if the patch was freshly applied, ``False`` if already
    applied (and the module is still registered) or mlx-lm is not importable.
    """
    global _APPLIED
    # Re-apply if our module entry was popped out from under us (e.g. by a
    # unit test that exercises the "missing base patch" path).
    if _APPLIED and "mlx_lm.models.hy_v3" in sys.modules:
        return False

    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        logger.debug("mlx_lm not importable - hy_v3 patch skipped")
        return False

    # Order matters: register any sibling modules first if the pinned
    # mlx-lm lacks them, then register hy_v3 itself. The reference
    # template is omlx.patches.deepseek_v4 (registers hyper_connection
    # BEFORE deepseek_v4). For Hy3, the upstream mlx-lm tree already
    # ships activations/base/pipeline/rope_utils/switch_layers — verify
    # at runtime, fail loudly with a clear message if missing.
    _ensure_sibling_modules()

    _register_module("mlx_lm.models.hy_v3", "hy_v3_model.py")
    _register_model_type_aliases()
    _register_tool_parser_modules()

    _APPLIED = True
    logger.info(
        "Hy3 base architecture patch applied (PR #1211 + #1485 head %s)",
        PR_HEAD_SHA[:8],
    )
    return True


def _ensure_sibling_modules() -> None:
    """Pre-flight check that the relative-import siblings exist in mlx_lm.

    ``hy_v3_model.py`` imports ``from .activations / .base / .pipeline /
    .rope_utils / .switch_layers`` — these resolve through
    ``module.__package__ = "mlx_lm.models"``. If the pinned mlx-lm is
    missing any of them, surface the issue with a clear actionable error
    rather than letting exec_module raise a bare ImportError.

    If a sibling is missing, vendor it from
    ``kernelpool/mlx-lm@add-hy3-preview`` and register it here BEFORE
    hy_v3 (mirror deepseek_v4's hyper_connection step).
    """
    required = {
        "activations": ("swiglu",),
        "base": (
            "BaseModelArgs",
            "create_attention_mask",
            "scaled_dot_product_attention",
        ),
        "pipeline": ("PipelineMixin",),
        "rope_utils": ("initialize_rope",),
        "switch_layers": ("SwitchGLU",),
    }
    missing: list[str] = []
    for mod_name, symbols in required.items():
        try:
            mod = importlib.import_module(f"mlx_lm.models.{mod_name}")
        except ImportError as e:
            missing.append(f"mlx_lm.models.{mod_name} (import failed: {e})")
            continue
        for sym in symbols:
            if not hasattr(mod, sym):
                missing.append(f"mlx_lm.models.{mod_name}.{sym}")

    if missing:
        raise ImportError(
            "Hy3 patch requires the following sibling modules/symbols to be "
            "present in mlx_lm.models.*, but they are missing from the pinned "
            "mlx-lm: " + ", ".join(missing) + ". Vendor them from "
            "kernelpool/mlx-lm@add-hy3-preview and register before hy_v3."
        )


def is_applied() -> bool:
    return _APPLIED


__all__ = [
    "apply_hy3_patch",
    "is_applied",
    "PR_BASE_URL",
    "PR_MTP_URL",
    "PR_HEAD_SHA",
]
