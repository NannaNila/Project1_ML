import argparse
import csv
import json
import importlib.util
import random
import time
from pathlib import Path
import numpy as np
import helpers
import implementations
import svm_model

ROOT = Path(__file__).resolve().parent
MODELS = (
    "mean_squared_error_gd",
    "mean_squared_error_sgd",
    "least_squares",
    "ridge_regression",
    "logistic_regression",
    "reg_logistic_regression",
    "svm",
)
DEFAULT_LAMBDA = {"ridge_regression": 0.0001, "reg_logistic_regression": 0.001, "svm": 0.001}
# A dash is not valid in a normal Python import, so load the requested filename.
_spec = importlib.util.spec_from_file_location("data_preprocessing", ROOT / "data-preprocessing.py")
preprocessing = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(preprocessing)


def split_indices(y, seed=42):
    """Stratified outer 80% development / 20% test split."""
    rng = np.random.default_rng(seed)
    development, test = [], []
    for label in (-1, 1):
        ids = rng.permutation(np.flatnonzero(y == label))
        n = int(0.8 * len(ids))
        development.extend(ids[:n])
        test.extend(ids[n:])
    return rng.permutation(development), rng.permutation(test)


def inner_folds(y, seed=104):
    """Two stratified folds entirely inside the development 80%."""
    rng = np.random.default_rng(seed)
    parts = [[], []]
    for label in (-1, 1):
        for destination, chunk in zip(
            parts, np.array_split(rng.permutation(np.flatnonzero(y == label)), 2)
        ):
            destination.extend(chunk)
    return [rng.permutation(p).astype(int) for p in parts]


def call_model(name, y, tx, w, iterations, gamma, l2):
    """These are the only calls that learn model weights."""
    if name == "mean_squared_error_gd":
        return implementations.mean_squared_error_gd(y, tx, w, iterations, gamma)
    if name == "mean_squared_error_sgd":
        return implementations.mean_squared_error_sgd(y, tx, w, iterations, gamma)
    if name == "least_squares":
        return implementations.least_squares(y, tx)
    if name == "ridge_regression":
        return implementations.ridge_regression(y, tx, l2)
    if name == "reg_logistic_regression":
        return implementations.reg_logistic_regression(y, tx, l2, w, iterations, gamma)
    if name == "logistic_regression":
        return implementations.logistic_regression(y, tx, w, iterations, gamma)
    raise ValueError(f"Unknown model: {name}")


def fit_svm(
    X,
    y,
    lambda_=0.001,
    max_iters=1000000,
    optimizer="coordinate",
    gamma=1e-4,
    seed=42,
    tol=1e-5,
):
    """Fit a linear SVM and report convergence at objective checkpoints."""
    if lambda_ <= 0 or not np.isfinite(lambda_):
        raise ValueError("lambda must be finite and positive")
    if optimizer not in {"coordinate", "sgd"}:
        raise ValueError("Unknown SVM optimizer")
    if max_iters < 1 or gamma <= 0 or tol <= 0:
        raise ValueError("Iteration cap, gamma and tolerance must be positive")
    if X.ndim != 2 or y.shape != (len(X),) or len(y) == 0:
        raise ValueError("Expected a nonempty matrix and one label per row")
    if not np.isfinite(X).all() or not np.isin(y, [-1, 1]).all():
        raise ValueError("SVM requires finite features and -1/+1 labels")
    w = np.zeros(X.shape[1])
    alpha = np.zeros(len(y)) if optimizer == "coordinate" else None
    rng = random.Random(seed)
    started = time.monotonic()
    check_every = min(100000, max(1000, len(y)))
    gap, dual = None, None
    history = []
    for it in range(max_iters):
        n = rng.randint(0, len(y) - 1)
        if optimizer == "coordinate":
            w, alpha = svm_model.calculate_coordinate_update(y, X, lambda_, alpha, w, n)
        else:
            grad = svm_model.calculate_stochastic_gradient(y, X, w, lambda_, n)
            w -= gamma / (it + 1) * grad
        if (it + 1) % check_every == 0 or it + 1 == max_iters:
            primal = svm_model.calculate_primal_objective(y, X, w, lambda_)
            if alpha is not None:
                dual = svm_model.calculate_dual_objective(w, alpha, lambda_)
                gap = primal - dual
            history.append(dict(iterations=it + 1, primal=primal, dual=dual, gap=gap))
            if not np.isfinite(primal):
                raise ValueError("Nonfinite SVM objective")
            if alpha is not None and -1e-10 <= gap <= tol:
                break
    info = dict(
        iterations=it + 1,
        objective=primal,
        dual_objective=dual,
        primal_dual_gap=gap,
        converged=gap is not None and -1e-10 <= gap <= tol,
        convergence_criterion=(
            f"primal-dual gap <= {tol}"
            if alpha is not None
            else "iteration cap (no dual certificate for SGD)"
        ),
        optimizer=(
            "Exercise 06 dual coordinate ascent"
            if alpha is not None
            else "Exercise 06 SGD"
        ),
        gamma=gamma if optimizer == "sgd" else None,
        learning_rate_schedule="gamma/(iteration+1)" if optimizer == "sgd" else None,
        fit_seconds=time.monotonic() - started,
        history=history,
    )
    return w, alpha, info


def train(name, tx, y, l2, args):
    """Train from zero, calling the supplied function in chunks until stopped."""
    if name == "svm":
        w, _, info = fit_svm(
            tx, y, lambda_=l2, max_iters=args.svm_iters,
            optimizer=args.svm_optimizer, gamma=args.gamma or 1e-4,
            seed=args.seed, tol=args.svm_tol,
        )
        return w, info
    started = time.monotonic()
    gram, rhs = tx.T @ tx / len(y), tx.T @ y / len(y)
    logistic = "logistic" in name
    target = (y == 1).astype(float) if logistic else y
    sgd = name == "mean_squared_error_sgd"
    matrix = None
    if name in {"mean_squared_error_sgd", "logistic_regression"}:
        values, vectors = np.linalg.eigh(gram)
        if values.min() <= 0:
            raise ValueError("Whitening requires independent feature columns")
        matrix = vectors / np.sqrt(values)
    inputs = tx @ matrix if matrix is not None else tx
    w = np.zeros(inputs.shape[1])

    def check(weights):
        original = matrix @ weights if matrix is not None else weights
        scores = tx @ original
        if logistic:
            loss = float(np.mean(np.logaddexp(0, scores) - target * scores))
            gradient = tx.T @ (implementations.sigmoid(scores) - target) / len(y)
        else:
            loss = float(0.5 * np.mean((scores - y) ** 2))
            gradient = gram @ original - rhs
        gradient += 2 * l2 * original
        return loss, float(np.max(np.abs(gradient)))

    np.random.seed(args.seed)
    loss, gradient = check(w)
    iterations, close_count = 0, 0
    reference = implementations.least_squares(y, tx)[1] if sgd else None
    cap = args.sgd_iters if sgd else args.max_iters
    if name in {"least_squares", "ridge_regression"}:
        w, loss = call_model(name, target, inputs, w, 0, None, l2)
    else:
        largest = float(np.linalg.eigvalsh(gram)[-1])
        row_bound = float(np.einsum("ij,ij->i", inputs, inputs).max()) if sgd else None
        while iterations < cap and (sgd or gradient > args.tol):
            count = min(
                1000000 if sgd else 500 if logistic else 10000, cap - iterations
            )
            if args.gamma is not None:
                gamma = args.gamma
            elif sgd:
                gamma = 1.8 / row_bound * 0.5 ** (iterations // 5000000)
            elif logistic:
                p = implementations.sigmoid(inputs @ w)
                hessian = inputs.T @ ((p * (1 - p))[:, None] * inputs) / len(y)
                gamma = 1.5 / (float(np.linalg.eigvalsh(hessian)[-1]) + 2 * l2)
            else:
                gamma = 1.8 / largest
            # For automatic logistic rates, retry a chunk if training loss rises.
            before = loss + l2 * float(w @ w)
            for attempt in range(20):
                with np.errstate(over="ignore", invalid="ignore"):
                    candidate, candidate_loss = call_model(
                        name, target, inputs, w, count, gamma, l2
                    )
                objective = float(candidate_loss) + l2 * float(candidate @ candidate)
                if (
                    np.isfinite(candidate).all()
                    and np.isfinite(objective)
                    and (not logistic or objective <= before + 1e-12)
                ):
                    w = candidate
                    break
                if args.gamma is not None or not logistic:
                    raise ValueError("Unstable learning rate: reduce --gamma")
                gamma *= 0.5
            else:
                raise RuntimeError("Could not find a stable learning rate")
            iterations += count
            loss, gradient = check(w)
            if sgd:
                close_count = close_count + 1 if loss - reference <= getattr(args, "sgd_loss_tol", 1e-4) else 0
                if close_count == 2:
                    break
    loss, gradient = check(w)
    weights = matrix @ w if matrix is not None else w
    return weights, dict(
        loss=loss,
        iterations=iterations,
        fit_seconds=time.monotonic() - started,
        gradient_max_abs=gradient,
        gradient_check_passed=gradient <= args.tol,
        converged=close_count >= 2 if sgd else gradient <= args.tol,
        convergence_criterion=(
            f"loss gap <= {getattr(args, 'sgd_loss_tol', 1e-4)} twice"
            if sgd else f"gradient <= {args.tol}"
        ),
        loss_gap=loss - reference if sgd else None,
        final_gamma=gamma if iterations else None,
        whitening=matrix is not None,
    )


def best_threshold(y, scores):
    """Choose the F1 threshold on out-of-fold training predictions only."""
    order = np.argsort(-scores, kind="stable")
    values = scores[order]
    ends = np.r_[np.flatnonzero(np.diff(values) != 0), len(values) - 1]
    tp = np.cumsum(y[order] == 1)[ends]
    f1 = 2 * tp / (ends + 1 + np.sum(y == 1))
    return float(values[ends[np.argmax(f1)]])


def metrics(y, scores, threshold):
    positive = scores >= threshold
    tp = int(np.sum(positive & (y == 1)))
    fp = int(np.sum(positive & (y == -1)))
    fn = int(np.sum(~positive & (y == 1)))
    return dict(
        f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        accuracy=float(np.mean(positive == (y == 1))),
        precision=tp / (tp + fp) if tp + fp else 0.0,
        recall=tp / (tp + fn) if tp + fn else 0.0,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["all", *MODELS], default="all")
    parser.add_argument("--svm-iters", type=int, default=5000000,
                        help="SVM single-example update cap")
    parser.add_argument("--svm-optimizer", choices=["coordinate", "sgd"], default="coordinate")
    parser.add_argument("--svm-tol", type=float, default=1e-4,
                        help="SVM primal-dual gap tolerance")
    parser.add_argument(
        "--lambda", dest="l2", type=float, help="Fixed regularization strength"
    )
    parser.add_argument(
        "--tune", action="store_true", help="Compare lambda=0.0001, 0.001, 0.01 for regularized models"
    )
    parser.add_argument(
        "--gamma", type=float, help="Fixed learning rate (default: automatic)"
    )
    parser.add_argument(
        "--max-iters", type=int, default=1000000, help="Batch-GD iteration cap"
    )
    parser.add_argument(
        "--sgd-iters", type=int, default=60000000, help="Single-example SGD update cap"
    )
    parser.add_argument(
        "--tol", type=float, help="Batch-gradient tolerance (default: 1e-4)"
    )
    parser.add_argument(
        "--sgd-loss-tol", type=float, default=1e-4,
        help="SGD training-loss gap tolerance (two consecutive checkpoints)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/dataset")
    parser.add_argument("--output-dir", type=Path, help="Default: output/corrected_sergi")
    parser.add_argument(
        "--export",
        action="store_true",
        help="Write one submission.csv per selected model",
    )
    parser.add_argument(
        "--sub-sample",
        action="store_true",
        help="Quick check on every 50th training row",
    )
    args = parser.parse_args()
    if args.tol is None:
        args.tol = 1e-4
    if args.output_dir is None:
        args.output_dir = ROOT / "output/corrected_sergi"
    if not np.isfinite(args.sgd_loss_tol) or args.sgd_loss_tol <= 0:
        parser.error("--sgd-loss-tol must be finite and positive")
    if args.svm_iters < 1 or not np.isfinite(args.svm_tol) or args.svm_tol <= 0:
        parser.error("SVM iteration cap and tolerance must be positive")
    if args.model in {"svm", "all"}:
        if args.l2 is not None and args.l2 <= 0:
            parser.error("SVM lambda must be positive")
        if args.gamma is not None and args.svm_optimizer != "sgd":
            parser.error("For SVM, --gamma requires --svm-optimizer sgd")
    if (
        args.max_iters < 1
        or args.sgd_iters < 1
        or not np.isfinite(args.tol)
        or args.tol <= 0
    ):
        parser.error("Iteration caps and tolerance must be positive")
    if args.l2 is not None and (not np.isfinite(args.l2) or args.l2 < 0):
        parser.error("lambda must be finite and nonnegative")
    if args.gamma is not None and (not np.isfinite(args.gamma) or args.gamma <= 0):
        parser.error("gamma must be finite and positive")
    if args.tune and args.l2 is not None:
        parser.error("Choose --tune or a fixed --lambda")
    if (args.tune or args.l2 is not None) and args.model not in (
        "all",
        "svm",
        *DEFAULT_LAMBDA,
    ):
        parser.error(
            "--lambda and --tune apply to ridge, regularized logistic, or SVM"
        )
    if args.sub_sample and args.export:
        parser.error("Use the complete dataset for leaderboard predictions")
    names = list(MODELS) if args.model == "all" else [args.model]

    # 1. Load data and reserve the outer test 20%.
    x, x_test, y, ids, test_ids = helpers.load_csv_data(
        str(args.data_dir), args.sub_sample
    )
    if set(np.unique(y)) != {-1, 1}:
        raise ValueError("Expected -1/+1 training labels")
    with (args.data_dir / "x_train.csv").open() as source:
        feature_names = np.array(next(csv.reader(source))[1:])
    fit = lambda rows: preprocessing.fit_preprocessing(rows, feature_names)
    encode = preprocessing.transform
    development, test = split_indices(y, args.seed)
    x_train, y_train = x[development], y[development]
    candidates = {}
    for name in names:
        lambdas = [0.0]
        if name in DEFAULT_LAMBDA:
            lambdas = (
                [0.0001, 0.001, 0.01]
                if args.tune
                else [args.l2 if args.l2 is not None else DEFAULT_LAMBDA[name]]
            )
        candidates[name] = [
            dict(l2=l2, scores=np.empty(len(y_train)), fits=[]) for l2 in lambdas
        ]

    # 2. Two-fold predictions inside the 80% choose thresholds and optional lambda.
    for fold, validation in enumerate(inner_folds(y_train)):
        fitting = np.setdiff1d(np.arange(len(y_train)), validation)
        state = fit(x_train[fitting])
        tx, tv = encode(x_train[fitting], state), encode(
            x_train[validation], state
        )
        for name in names:
            for candidate in candidates[name]:
                print(
                    f"Fold {fold + 1}/2: {name}, lambda={candidate['l2']:g}", flush=True
                )
                w, info = train(name, tx, y_train[fitting], candidate["l2"], args)
                candidate["scores"][validation] = tv @ w
                candidate["fits"].append(info)
    selected = {}
    for name in names:
        for candidate in candidates[name]:
            scores = candidate.pop("scores")
            candidate["threshold"] = best_threshold(y_train, scores)
            candidate["oof"] = metrics(y_train, scores, candidate["threshold"])
        eligible = (
            [c for c in candidates[name] if all(f["converged"] for f in c["fits"])]
            if args.tune
            else candidates[name]
        )
        if not eligible:
            raise RuntimeError(f"{name}: increase iteration caps before tuning")
        selected[name] = max(eligible, key=lambda c: c["oof"]["f1"])

    # 3. Train the chosen models on the full 80%, then score the reserved 20%.
    state = fit(x_train)
    tx, te = encode(x_train, state), encode(x[test], state)
    tt = encode(x_test, state) if args.export else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "preprocessing.npz", **state)
    results = {}
    for name in names:
        choice = selected[name]
        print(f"Final training: {name}, lambda={choice['l2']:g}", flush=True)
        w, info = train(name, tx, y_train, choice["l2"], args)
        result = dict(
            lambda_=choice["l2"],
            threshold=choice["threshold"],
            oof=choice["oof"],
            training=info,
            test=metrics(y[test], te @ w, choice["threshold"]),
            candidates=candidates[name],
        )
        folder = args.output_dir / name
        folder.mkdir(exist_ok=True)
        np.savez_compressed(folder / "model.npz", w=w, **state)
        if args.export:
            helpers.create_csv_submission(
                test_ids,
                np.where(tt @ w >= choice["threshold"], 1, -1),
                str(folder / "submission.csv"),
            )
        (folder / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        results[name] = result
        print(
            f"  F1={result['test']['f1']:.5f}, accuracy={result['test']['accuracy']:.5f}, converged={info['converged']}",
            flush=True,
        )
    report = dict(
        training_rows=len(development),
        test_rows=len(test),
        sub_sample=args.sub_sample,
        seed=args.seed,
        max_iters=args.max_iters,
        sgd_iters=args.sgd_iters,
        tolerance=args.tol,
        sgd_loss_tolerance=args.sgd_loss_tol,
        preprocessing="corrected_sergi",
        svm_iters=args.svm_iters,
        svm_tolerance=args.svm_tol,
        feature_columns=tx.shape[1],
        gamma_override=args.gamma,
        tune=args.tune,
        note="80/20 outer split; two-fold threshold/optional lambda tuning inside the 80%; SGD convergence uses near-optimal training loss; SVM uses a primal-dual gap; other iterative models use a gradient check.",
        results=results,
    )
    (args.output_dir / "results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print("Results:", args.output_dir / "results.json")


if __name__ == "__main__":
    main()
