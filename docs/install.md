# Install

## What you need

- Python 3.13
- One of:
  - a Mac with Apple Silicon, on macOS 15 or later
  - Linux on x86-64 with an AMD chip from 2017 on (any chip with AVX2 works), and glibc 2.27 or later
- pip, and a virtual environment

## Install the wheel

Download the wheel for your platform from the [Releases page](https://github.com/Binomial-Capital-Management/binomial-shira/releases), then:

```
python3.13 -m venv .venv
source .venv/bin/activate
pip install shira-0.1.0-py313-none-macosx_15_0_arm64.whl        # Mac
pip install shira-0.1.0-py313-none-manylinux_2_28_x86_64.whl    # Linux
```

pip also installs the versions of the stock libraries that SHIRA is built against:

| Library | Version |
|---|---|
| scikit-learn | 1.8.0 |
| XGBoost | 3.2.0 |
| LightGBM | 4.6.0 |
| CatBoost | 1.2.10 |
| NumPy | 2.4.2 |
| SciPy | 1.15.3 |
| joblib | 1.5.3 |

SHIRA follows each library's behaviour version by version, so these versions are fixed. Use a fresh
virtual environment if your project needs other versions elsewhere.

## Check it works

```
python -c "import shira.xgboost as xgb, numpy as np; X = np.random.rand(20000, 8); xgb.XGBRegressor(n_estimators=20).fit(X, X[:, 0]); print(xgb.last_backend)"
```

It prints `shira`.

## Not supported in this version

- Installs that are not a pip install into an environment, such as `pip install --target`.
- Intel Macs, Windows, and Linux on ARM.
