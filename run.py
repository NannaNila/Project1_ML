"""Compare the six supplied implementations; standard library and NumPy only."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

import helpers
import implementations


ROOT = Path(__file__).resolve().parent
METHODS = (
    "mean_squared_error_gd",
    "mean_squared_error_sgd",
    "least_squares",
    "ridge_regression",
    "logistic_regression",
    "reg_logistic_regression",
)


def split_indices(y, seed=42):
    """Stratify identical 60/20/20 splits for every method."""
    rng = np.random.default_rng(seed)
    parts = [[], [], []]
    for label in (-1, 1):
        ids = rng.permutation(np.flatnonzero(y == label))
        if len(ids) < 5:
            raise ValueError("Need at least five examples of each class.")
        a, b = int(0.6 * len(ids)), int(0.8 * len(ids))
        for destination, chunk in zip(parts, (ids[:a], ids[a:b], ids[b:])):
            destination.extend(chunk.tolist())
    return tuple(rng.permutation(np.array(p, dtype=int)) for p in parts)


def fit_preprocessing(x):
    """Training-only median imputation, scaling and redundant-column removal."""
    observed = np.flatnonzero(np.isfinite(x).any(axis=0))
    clean = np.where(np.isfinite(x[:, observed]), x[:, observed], np.nan)
    medians = np.nanmedian(clean, axis=0)
    filled = np.where(np.isfinite(clean), clean, medians)
    means, scales = filled.mean(axis=0), filled.std(axis=0)
    keep = scales > 1e-10
    state = dict(
        raw_columns=observed[keep],
        medians=medians[keep],
        means=means[keep],
        scales=scales[keep],
    )
    tx = transform(x, state)
    # Select original columns with independent residual variance. This only
    # removes redundant features; learned model weights come from implementations.
    residual = tx.T @ tx / len(tx)
    independent = []
    for _ in range(tx.shape[1]):
        j = 0 if not independent else int(np.argmax(np.diag(residual)))
        if residual[j, j] <= 1e-8:
            break
        independent.append(j)
        direction = residual[:, j].copy() / np.sqrt(residual[j, j])
        residual -= np.outer(direction, direction)
        residual[j, :] = residual[:, j] = 0
    state["independent_columns"] = np.array(independent, dtype=int)
    return state


def transform(x, state):
    selected = x[:, state["raw_columns"]]
    filled = np.where(np.isfinite(selected), selected, state["medians"])
    tx = np.column_stack((np.ones(len(x)), (filled - state["means"]) / state["scales"]))
    return tx[:, state["independent_columns"]] if "independent_columns" in state else tx


def training_constants(tx):
    """Bounds for fixed learning rates, computed on training features only."""
    largest = float(np.linalg.eigvalsh(tx.T @ tx / len(tx))[-1])
    largest_row = float(np.einsum("ij,ij->i", tx, tx).max())
    return largest, largest_row


def fit_method(name, tx, y, args, constants):
    """Every weight update and solve below is delegated to the supplied file."""
    largest, largest_row = constants
    initial = np.zeros(tx.shape[1])
    y01 = (y == 1).astype(float)
    gamma, iterations = None, None
    started = time.monotonic()
    if name == "mean_squared_error_gd":
        gamma, iterations = 1 / largest, args.max_iters
        w, loss = implementations.mean_squared_error_gd(
            y, tx, initial, iterations, gamma
        )
    elif name == "mean_squared_error_sgd":
        np.random.seed(args.seed)
        gamma, iterations = 1 / largest_row, args.sgd_iters
        w, loss = implementations.mean_squared_error_sgd(
            y, tx, initial, iterations, gamma
        )
    elif name == "least_squares":
        w, loss = implementations.least_squares(y, tx)
    elif name == "ridge_regression":
        w, loss = implementations.ridge_regression(y, tx, args.lambda_)
    elif name == "logistic_regression":
        gamma, iterations = 4 / largest, args.max_iters
        w, loss = implementations.logistic_regression(
            y01, tx, initial, iterations, gamma
        )
    elif name == "reg_logistic_regression":
        gamma, iterations = 1 / (largest / 4 + 2 * args.lambda_), args.max_iters
        w, loss = implementations.reg_logistic_regression(
            y01, tx, args.lambda_, initial, iterations, gamma
        )
    else:
        raise ValueError(name)
    seconds = time.monotonic() - started
    if not np.isfinite(w).all() or not np.isfinite(loss):
        raise ValueError(f"{name} returned nonfinite weights or loss.")
    # Diagnostics only: never use this gradient to update or replace weights.
    residual = (
        implementations.sigmoid(tx @ w) - y01 if "logistic" in name else tx @ w - y
    )
    gradient = tx.T @ residual / len(y)
    if name in {"ridge_regression", "reg_logistic_regression"}:
        gradient += 2 * args.lambda_ * w
    norm = float(np.max(np.abs(gradient)))
    return w, dict(
        loss=float(loss),
        fit_seconds=seconds,
        gamma=gamma,
        iterations=iterations,
        lambda_=(
            args.lambda_
            if name in {"ridge_regression", "reg_logistic_regression"}
            else 0
        ),
        gradient_max_abs=norm,
        converged=norm <= 1e-5,
        gradient_tolerance=1e-5,
    )


def f1_threshold(y, scores):
    """Optimize calibration F1, respecting tied scores; no accuracy constraint."""
    order = np.argsort(-scores, kind="stable")
    s = scores[order]
    ends = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]
    tp = np.cumsum(y[order] == 1)[ends]
    f1 = 2 * tp / (ends + 1 + np.sum(y == 1))
    return float(s[ends[np.argmax(f1)]])


def metrics(y, scores, threshold):
    positive = scores >= threshold
    tp = int(np.sum(positive & (y == 1)))
    fp = int(np.sum(positive & (y == -1)))
    fn = int(np.sum(~positive & (y == 1)))
    tn = len(y) - tp - fp - fn
    return dict(
        f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        accuracy=(tp + tn) / len(y),
        precision=tp / (tp + fp) if tp + fp else 0.0,
        recall=tp / (tp + fn) if tp + fn else 0.0,
        confusion_matrix=[[tn, fp], [fn, tp]],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/dataset")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    parser.add_argument("--max-iters", type=int, default=1000)
    parser.add_argument("--sgd-iters", type=int, default=100000)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sub-sample", action="store_true")
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()
    if args.max_iters < 1 or args.sgd_iters < 1 or args.lambda_ < 0:
        parser.error("Iteration budgets must be positive and lambda nonnegative.")
    if args.export and args.sub_sample:
        parser.error("Use the complete training data when exporting submissions.")
    x, xt, y, ids, test_ids = helpers.load_csv_data(
        str(args.data_dir), sub_sample=args.sub_sample
    )
    if set(np.unique(y)) != {-1, 1}:
        raise ValueError("Expected -1/+1 training labels.")
    train, calibration, evaluation = split_indices(y, args.seed)
    state = fit_preprocessing(x[train])
    tx, tc, te = (transform(x[idx], state) for idx in (train, calibration, evaluation))
    constants = training_constants(tx)
    print(
        f"Rows: train={len(train)}, calibration={len(calibration)}, evaluation={len(evaluation)}; columns={tx.shape[1]}",
        flush=True,
    )
    results = {}
    for name in METHODS:
        print(f"Training implementations.{name}...", flush=True)
        w, diagnostics = fit_method(name, tx, y[train], args, constants)
        threshold = f1_threshold(y[calibration], tc @ w)
        result = dict(
            training=diagnostics,
            score_threshold=threshold,
            calibration=metrics(y[calibration], tc @ w, threshold),
            evaluation=metrics(y[evaluation], te @ w, threshold),
        )
        results[name] = result
        m = result["evaluation"]
        print(
            f"  F1={m['f1']:.5f}, accuracy={m['accuracy']:.5f}, gradient check passed={diagnostics['converged']}",
            flush=True,
        )
    report = dict(
        seed=args.seed,
        sub_sample=args.sub_sample,
        rows=len(y),
        split_rows=dict(
            training=len(train),
            calibration=len(calibration),
            evaluation=len(evaluation),
        ),
        columns=tx.shape[1],
        accuracy_constraint=False,
        results=results,
        training_source="Only direct calls to the six implementations.py functions",
        source_sha256={
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
            for p in ["implementations.py", "helpers.py"]
        },
        note="Fixed starting settings; no hyperparameter search. Local evaluation; subset scores are smoke-test results only.",
    )
    del tx, tc, te
    if args.export:
        full_state = fit_preprocessing(x)
        full = transform(x, full_state)
        test = transform(xt, full_state)
        constants = training_constants(full)
        for name, result in results.items():
            print(
                f"Refitting implementations.{name} on all {len(y)} labels...",
                flush=True,
            )
            w, diagnostics = fit_method(name, full, y, args, constants)
            folder = args.output_dir / "submissions" / name
            folder.mkdir(parents=True, exist_ok=True)
            labels = np.where(test @ w >= result["score_threshold"], 1, -1)
            helpers.create_csv_submission(
                test_ids, labels, str(folder / "submission.csv")
            )
            np.savez_compressed(folder / "model.npz", w=w, **full_state)
            result["all_data_training"] = diagnostics
            result["submission"] = str(folder / "submission.csv")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print("Results:", args.output_dir / "results.json")


if __name__ == "__main__":
    main()
