from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
    cross_val_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from eeg_pipeline import (
    MODE_OPTIONS,
    get_model_paths,
    get_report_path,
    prepare_dataset,
    resolve_data_path,
    save_feature_cache,
    write_json,
)


def build_model(random_state: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )


def _class_complete_split(y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> bool:
    classes = set(np.unique(y))
    return set(np.unique(y[train_idx])) == classes and set(np.unique(y[test_idx])) == classes


def make_split(
    features: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    unique_groups = np.unique(groups)
    if len(unique_groups) >= 2:
        for offset in range(50):
            splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state + offset)
            train_idx, test_idx = next(splitter.split(features, y, groups))
            if _class_complete_split(y, train_idx, test_idx):
                return train_idx, test_idx, "GroupShuffleSplit(patient)"

    class_counts = Counter(y)
    if min(class_counts.values()) >= 2:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(features, y))
        return train_idx, test_idx, "StratifiedShuffleSplit(label)"

    indices = np.arange(len(y))
    rng = np.random.default_rng(random_state)
    rng.shuffle(indices)
    split_at = max(1, int(len(indices) * (1 - test_size)))
    return indices[:split_at], indices[split_at:], "RandomShuffle(fallback)"


def cross_validate_model(
    features: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    random_state: int,
) -> tuple[list[float], str]:
    pipeline = make_pipeline(StandardScaler(), build_model(random_state))
    unique_groups = np.unique(groups)

    if len(unique_groups) >= 2:
        n_splits = min(5, len(unique_groups))
        if n_splits >= 2:
            cv = GroupKFold(n_splits=n_splits)
            scores = cross_val_score(pipeline, features, y, groups=groups, cv=cv, scoring="accuracy")
            return scores.tolist(), f"GroupKFold(n_splits={n_splits})"

    min_class_count = min(Counter(y).values())
    n_splits = min(5, min_class_count)
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        scores = cross_val_score(pipeline, features, y, cv=cv, scoring="accuracy")
        return scores.tolist(), f"StratifiedKFold(n_splits={n_splits})"

    return [], "skipped(insufficient samples)"


def train(args: argparse.Namespace) -> dict:
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1.")

    data_path = resolve_data_path(args.data)
    mode_defaults = MODE_OPTIONS[args.mode]
    max_epochs = args.max_epochs if args.max_epochs is not None else mode_defaults["max_epochs"]
    max_rows = args.max_rows if args.max_rows is not None else mode_defaults["max_rows"]

    dataset = prepare_dataset(
        data_path,
        mode=args.mode,
        max_epochs=max_epochs,
        max_rows=max_rows,
        apply_ica=not args.skip_ica,
    )
    save_feature_cache(args.mode, dataset)

    features = dataset["features"]
    labels = dataset["labels"]
    groups = dataset["patients"]
    if len(features) < 4:
        raise ValueError("At least 4 complete EEG epochs are required for training.")
    if len(np.unique(labels)) < 2:
        raise ValueError("At least two intent labels are required for supervised training.")

    le = LabelEncoder()
    y = le.fit_transform(labels)
    train_idx, test_idx, split_method = make_split(
        features,
        y,
        groups,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(features[train_idx])
    X_test = scaler.transform(features[test_idx])
    y_train = y[train_idx]
    y_test = y[test_idx]

    model = build_model(args.random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    cv_scores, cv_method = cross_validate_model(features, y, groups, random_state=args.random_state)
    report = classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(le.classes_)),
        target_names=le.classes_,
        zero_division=0,
    )

    metrics = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": args.mode,
        "data_path": str(data_path),
        "samples": int(len(features)),
        "feature_dim": int(features.shape[1]),
        "label_counts": {str(label): int(count) for label, count in Counter(labels).items()},
        "groups": int(len(np.unique(groups))),
        "split_method": split_method,
        "train_samples": int(len(train_idx)),
        "test_samples": int(len(test_idx)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "cv_method": cv_method,
        "cv_scores": cv_scores,
        "cv_accuracy_mean": float(np.mean(cv_scores)) if cv_scores else None,
        "cv_accuracy_std": float(np.std(cv_scores)) if cv_scores else None,
        "classification_report": report,
        "preprocessing": dataset["metadata"].get("preprocessing", {}),
        "alignment": dataset["metadata"].get("alignment"),
    }

    paths = get_model_paths(args.mode)
    paths["model"].parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, paths["model"])
    joblib.dump(scaler, paths["scaler"])
    joblib.dump(le, paths["label_encoder"])
    write_json(paths["metadata"], metrics)
    write_json(get_report_path(args.mode, "json"), metrics)

    md = [
        f"# EEG model evaluation ({args.mode})",
        "",
        f"- Generated at: {metrics['generated_at']}",
        f"- Data path: {metrics['data_path']}",
        f"- Alignment: {metrics['alignment']}",
        f"- Split: {metrics['split_method']}",
        f"- Samples: {metrics['samples']}",
        f"- Feature dimension: {metrics['feature_dim']}",
        f"- Accuracy: {metrics['accuracy']:.2%}",
        f"- Weighted F1: {metrics['f1_weighted']:.2%}",
        f"- CV: {metrics['cv_method']}",
        "",
        "## Classification Report",
        "",
        "```text",
        report,
        "```",
    ]
    get_report_path(args.mode, "md").parent.mkdir(parents=True, exist_ok=True)
    get_report_path(args.mode, "md").write_text("\n".join(md), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EEG intent model and generate runtime artifacts.")
    parser.add_argument("--mode", choices=sorted(MODE_OPTIONS), default="demo")
    parser.add_argument("--data", default=None, help="CSV path. Defaults to project data candidates.")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override mode epoch limit.")
    parser.add_argument("--max-rows", type=int, default=None, help="Override mode CSV row limit.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--skip-ica", action="store_true", help="Skip IIR+ICA preprocessing for fast smoke tests.")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        result = train(parse_args())
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        raise SystemExit(
            f"Missing dependency: {missing}. Install dependencies with: "
            "python -m pip install -r requirements.txt"
        ) from exc
    except Exception as exc:
        raise SystemExit(f"Training failed: {exc}") from exc
    print(f"trained mode={result['mode']} samples={result['samples']} accuracy={result['accuracy']:.2%}")
