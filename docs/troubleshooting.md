# Troubleshooting

## `last_backend` is the stock library, not `shira`

That call is not covered yet, so it ran on the stock library with the same result as without SHIRA.
Turn on logging to read the reason:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

The line comes from the logger `shira.xgboost`, `shira.lightgbm`, `shira.catboost` or `shira.sklearn`.
[What is covered](coverage.md) lists the calls SHIRA runs.

## "SHIRA's engines reproduce version X only"

SHIRA follows each stock library version by version. Install the pinned versions listed in
[Install](install.md); pip does this for you in a fresh environment.

## `import shira` fails after installing

SHIRA must be installed with pip into a virtual environment. `pip install --target` and copying the
files by hand are not supported in this version.

## "Illegal instruction" on Linux

The Linux build needs a chip with AVX2: any AMD Zen chip, or Intel from Haswell on.

## The library does not load on an older macOS

The Mac build needs macOS 15 or later.

## My results differ from the stock library

That is the default for tree models: SHIRA takes its fastest route. Use the settings in
[Bit-identical results](bit-identical.md) to get the stock library's digits.

## I want SHIRA off for one run

Do not call `shira.patch()`, or import the stock library directly. With the per-library imports, the
stock modules are untouched: `import xgboost` is still stock XGBoost.
