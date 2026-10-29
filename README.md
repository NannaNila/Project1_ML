# Project 1: fresh baseline

All model training uses the six functions in `implementations.py` directly.
Data loading and CSV export use `helpers.py`. Those two files are unchanged.
The only other application code is `run.py`, which prepares data, calls the
functions, measures performance, and optionally exports predictions.

## Run

Place `x_train.csv`, `x_test.csv`, and `y_train.csv` in `data/dataset/`.
Computation requires only Python and NumPy.

```bash
python3 run.py
```

This compares all six methods and writes `output/results.json`. A quick check
using the helper's every-50th-row training subset is:

```bash
python3 run.py --sub-sample --max-iters 100 --sgd-iters 1000
```

To compare the methods and then refit each on all labelled rows for six **new**
submission files, run:

```bash
python3 run.py --export
```

Files are saved as `output/submissions/<function_name>/submission.csv`.
Nothing is submitted online. Subset runs cannot export leaderboard predictions.

## What this baseline does

- Uses one stratified 60% training / 20% threshold-calibration / 20% evaluation
  split, seed 42. Every method receives the same rows and features.
- Fits median imputation and standardization on training rows only. Removes
  constant and linearly dependent columns so the supplied normal-equation
  least-squares function can solve the system. Adds an intercept.
- Keeps the raw feature representation: no one-hot encoding, nonlinear features,
  class weights, resampling, kernels, trees, or ensembles.
- Uses -1/+1 targets for squared-error methods and 0/1 for logistic methods.
- Selects a separate F1-maximizing score threshold on calibration rows for each
  method, without an accuracy constraint. Evaluation labels do not select it.
- Reports F1, accuracy, precision, recall, training time, and an objective-gradient
  check. A fixed iteration budget does **not** guarantee convergence, particularly
  for SGD; the report explicitly records this. Learning rates use training data.
- Starts with lambda=0.001 for both regularized methods, 1,000 full-batch steps
  and 100,000 SGD steps. These are starting settings, not tuned best parameters.
  Override them with `--lambda`, `--max-iters`, and `--sgd-iters`.
- Preserves the supplied functions' treatment of the intercept and returned loss.
  Both regularized functions penalize all weights, including the intercept.

`mean_squared_error_gd`, `mean_squared_error_sgd`, and `least_squares` fit the same
unregularized linear model with different solvers. Their comparison checks
optimization as well as predictive performance. This is a fresh baseline, with
no reused models, hyperparameters selected from old runs, or claimed new scores.

Assignment documents, teacher tests, report template, dataset and original team
Git history are retained. Old experiment code and predictions are removed from
the working tree; a full backup was made outside this repository before cleanup.
