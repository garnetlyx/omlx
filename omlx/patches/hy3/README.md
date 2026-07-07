# SPDX-License-Identifier: Apache-2.0.
"""Tencent Hy3 (``hy_v3``) base architecture patch.

Vendors the Hy3 model class from
[ml-explore/mlx-lm#1211](https://github.com/ml-explore/mlx-lm/pull/1211)
(kernelpool, base architecture + tool parsers + tokenizer fixes) **plus**
[ml-explore/mlx-lm#1485](https://github.com/ml-explore/mlx-lm/pull/1485)
(eauchs, kept-and-used Multi-Token-Prediction layer for self-speculative
decoding — stacked on #1211).

The vendored ``hy_v3_model.py`` is the PR #1485 head SHA ``a7cc3054``
(literally verbatim from the source PR). PR #1211 strips the MTP layer at
load; PR #1485 keeps and uses it. We vendor #1485 directly so a single
file handles both mtp-enabled and mtp-disabled paths.

## Activation

Gate: ``config.json``'s ``model_type`` is exactly ``"hy_v3"``. The patch is
triggered from ``omlx/utils/model_loading.py::maybe_apply_pre_load_patches``
which dispatches via the inline if-chain there.

## What this package does

1. Registers ``mlx_lm.models.hy_v3`` in ``sys.modules`` by loading
   ``hy_v3_model.py`` through ``importlib.util.spec_from_file_location``
   with ``module.__package__ = "mlx_lm.models"``. This stamp makes the
   file's relative imports (``from .activations import swiglu`` etc.)
   resolve through the real mlx_lm package.
2. Pre-flights the required sibling modules/symbols at apply time so
   missing siblings surface as actionable errors instead of bare
   ``ImportError`` during ``exec_module``.
3. Aliases ``hy_v3_mtp`` model_type variant to the same module.

## MTP integration

The MTP runtime hooks (``mtp_forward``, ``make_mtp_cache``,
``return_hidden`` on ``__call__``, ``_omlx_mtp_decode_enabled`` flag) live
in a sibling module:
``omlx/patches/mlx_lm_mtp/hy_v3_model.py``. That patch runs **after** the
base patch (so it can read ``sys.modules['mlx_lm.models.hy_v3']`` and
monkey-patch the class).

## Removal recipe

Once mlx-lm merges PR #1485 upstream:

1. ``rm -rf omlx/patches/hy3/``
2. ``rm omlx/patches/mlx_lm_mtp/hy_v3_model.py``
3. Revert the ``hy_v3`` branch in ``omlx/utils/model_loading.py``
   (the ``if model_type == "hy_v3":`` block).
4. Drop the ``hy_v3_model`` import in ``omlx/patches/mlx_lm_mtp/__init__.py``.
5. Revert the ``hy_v3*`` entries in ``model_settings.py``,
   ``admin/routes.py``, ``oq.py``, and ``_is_mtp_compatible``.
6. Repin mlx-lm in ``pyproject.toml`` to the merged upstream commit.

## File map

| Local file | Source at vendoring time |
|---|---|
| ``hy_v3_model.py`` | [eauchs/mlx-lm@hy_v3-mtp HEAD](https://github.com/eauchs/mlx-lm/tree/hy_v3-mtp) — verbatim ``mlx_lm/models/hy_v3.py`` |
| ``tools/hy_v3.py`` | [kernelpool/mlx-lm@add-hy3-preview HEAD](https://github.com/kernelpool/mlx-lm/tree/add-hy3-preview) — verbatim ``mlx_lm/tool_parsers/hy_v3.py`` |
| ``tools/hy_v3_opensource.py`` | Same branch — verbatim ``mlx_lm/tool_parsers/hy_v3_opensource.py`` |

## Test checkpoint

End-to-end verification used
``ox-ox/Hy3-295B-Instruct-w2q3exp-AProjQ8-SExpQ8-OutQ8-MTP-mlx`` (112.6 GB,
stable ``tencent/Hy3`` base, mixed 2/3/Q8 affine group quant, MTP layer
enabled). See ``.omo/plans/hy3-omlx-mtp-patch.md`` for the live-test plan.
