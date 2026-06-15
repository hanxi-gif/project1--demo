from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    BASE_DIR / "BCICIV_2a_all_patients.csv",
    BASE_DIR / "data" / "demo_patients.csv",
]
DATA_PATH = next((path for path in DATA_CANDIDATES if path.exists()), DATA_CANDIDATES[-1])

MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
REPORTS_DIR = BASE_DIR / "reports"

SFREQ = 250
EPOCH_SECONDS = 3
EPOCH_SAMPLES = SFREQ * EPOCH_SECONDS
EEG_CHANNEL_COUNT = 22
MODE_OPTIONS = {
    "demo": {"max_epochs": 64, "max_rows": 64 * EPOCH_SAMPLES * 4},
    "full": {"max_epochs": None, "max_rows": None},
}


def normalize_mode(mode: str) -> str:
    mode = (mode or "demo").lower()
    if mode not in MODE_OPTIONS:
        raise ValueError(f"Unsupported mode: {mode}. Use one of {', '.join(MODE_OPTIONS)}.")
    return mode


def resolve_data_path(data_path: str | Path | None = None) -> Path:
    if data_path:
        path = Path(data_path)
        if not path.is_absolute():
            path = BASE_DIR / path
        return path
    return DATA_PATH


def get_model_paths(mode: str = "demo") -> dict[str, Path]:
    mode = normalize_mode(mode)
    root = MODELS_DIR / mode
    return {
        "model": root / "eeg_intent_model.pkl",
        "scaler": root / "eeg_scaler.pkl",
        "label_encoder": root / "eeg_label_encoder.pkl",
        "metadata": root / "training_metadata.json",
    }


def get_feature_cache_path(mode: str = "demo") -> Path:
    mode = normalize_mode(mode)
    return ARTIFACTS_DIR / f"features_{mode}.joblib"


def get_report_path(mode: str = "demo", suffix: str = "json") -> Path:
    mode = normalize_mode(mode)
    return REPORTS_DIR / f"model_evaluation_{mode}.{suffix}"


def _infer_columns(df) -> dict[str, Any]:
    columns = list(df.columns)
    lower = {str(col).strip().lower(): col for col in columns}
    label_col = lower.get("label", columns[2] if len(columns) > 2 else None)
    patient_col = lower.get("patient")
    epoch_col = lower.get("epoch")
    time_col = lower.get("time")
    eeg_cols = [col for col in columns if str(col).strip().lower().startswith("eeg")]
    if len(eeg_cols) < EEG_CHANNEL_COUNT and len(columns) >= 4 + EEG_CHANNEL_COUNT:
        eeg_cols = columns[4:4 + EEG_CHANNEL_COUNT]
    eeg_cols = eeg_cols[:EEG_CHANNEL_COUNT]

    if label_col is None or len(eeg_cols) != EEG_CHANNEL_COUNT:
        raise ValueError(
            "CSV must contain a label column and 22 EEG columns, or match the layout "
            "patient,time,label,epoch,<22 EEG columns>."
        )
    return {
        "label": label_col,
        "patient": patient_col,
        "epoch": epoch_col,
        "time": time_col,
        "eeg": eeg_cols,
    }


def _label_for_group(group, label_col: str) -> str:
    values = group[label_col].dropna().astype(str).to_numpy()
    if len(values) == 0:
        return "unknown"
    return Counter(values).most_common(1)[0][0]


def _balanced_limit(records: list[dict[str, Any]], max_epochs: int | None) -> list[dict[str, Any]]:
    if max_epochs is None or len(records) <= max_epochs:
        return records

    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_label[str(record["label"])].append(record)

    selected: list[dict[str, Any]] = []
    labels = sorted(by_label)
    index = 0
    while len(selected) < max_epochs:
        added = False
        for label in labels:
            bucket = by_label[label]
            if index < len(bucket):
                selected.append(bucket[index])
                added = True
                if len(selected) >= max_epochs:
                    break
        if not added:
            break
        index += 1
    return selected


def load_epoch_aligned_segments(
    data_path: str | Path | None = None,
    *,
    mode: str = "demo",
    max_epochs: int | None = None,
    max_rows: int | None = None,
) -> dict[str, Any]:
    import pandas as pd

    mode = normalize_mode(mode)
    mode_defaults = MODE_OPTIONS[mode]
    if max_epochs is None:
        max_epochs = mode_defaults["max_epochs"]
    if max_rows is None:
        max_rows = mode_defaults["max_rows"]

    path = resolve_data_path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path, encoding_errors="ignore", nrows=max_rows)
    cols = _infer_columns(df)
    records: list[dict[str, Any]] = []
    epoch_sample_count = EPOCH_SAMPLES

    if cols["epoch"] is not None:
        group_cols = [cols["epoch"]]
        if cols["patient"] is not None:
            group_cols.insert(0, cols["patient"])
        grouped = df.groupby(group_cols, sort=False)
        group_items = []
        for key, group in grouped:
            if cols["time"] is not None:
                group = group.sort_values(cols["time"])
            if len(group) < 2:
                continue
            group_items.append((key, group))

        if not group_items:
            raise ValueError("No non-empty EEG epoch groups were found in the CSV.")

        size_counts = Counter(len(group) for _, group in group_items)
        epoch_sample_count = size_counts.most_common(1)[0][0]
        if epoch_sample_count < 2:
            raise ValueError("EEG epoch groups are too short for feature extraction.")

        for key, group in group_items:
            if len(group) < epoch_sample_count:
                continue
            segment = group.loc[:, cols["eeg"]].iloc[:epoch_sample_count].to_numpy(dtype=np.float32).T
            patient = str(group[cols["patient"]].iloc[0]) if cols["patient"] is not None else "single_patient"
            epoch_key = "|".join(str(part) for part in (key if isinstance(key, tuple) else (key,)))
            records.append({
                "segment": segment,
                "label": _label_for_group(group, cols["label"]),
                "patient": patient,
                "epoch_key": epoch_key,
            })
    else:
        eeg_values = df.loc[:, cols["eeg"]].to_numpy(dtype=np.float32)
        label_values = df.loc[:, cols["label"]].astype(str).to_numpy()
        patient_values = (
            df.loc[:, cols["patient"]].astype(str).to_numpy()
            if cols["patient"] is not None
            else np.array(["single_patient"] * len(df))
        )
        n_epochs = len(df) // EPOCH_SAMPLES
        for epoch_idx in range(n_epochs):
            start = epoch_idx * EPOCH_SAMPLES
            end = start + EPOCH_SAMPLES
            records.append({
                "segment": eeg_values[start:end].T,
                "label": Counter(label_values[start:end]).most_common(1)[0][0],
                "patient": Counter(patient_values[start:end]).most_common(1)[0][0],
                "epoch_key": str(epoch_idx),
            })

    records = _balanced_limit(records, max_epochs)
    if not records:
        raise ValueError("No complete EEG epochs were found. Check the epoch column and sample count.")

    segments = np.stack([record["segment"] for record in records]).astype(np.float32)
    labels = np.array([str(record["label"]) for record in records])
    patients = np.array([str(record["patient"]) for record in records])
    epoch_keys = np.array([str(record["epoch_key"]) for record in records])
    metadata = {
        "data_path": str(path),
        "mode": mode,
        "rows_loaded": int(len(df)),
        "epochs": int(len(records)),
        "epoch_samples": int(epoch_sample_count),
        "sfreq": SFREQ,
        "alignment": "csv_epoch_column" if cols["epoch"] is not None else "fixed_window_fallback",
        "label_column": str(cols["label"]),
        "patient_column": str(cols["patient"]) if cols["patient"] is not None else None,
        "epoch_column": str(cols["epoch"]) if cols["epoch"] is not None else None,
        "eeg_columns": [str(col) for col in cols["eeg"]],
        "label_counts": {str(label): int(count) for label, count in Counter(labels).items()},
        "epoch_size_distribution": {
            str(size): int(count)
            for size, count in Counter(record["segment"].shape[1] for record in records).items()
        },
    }
    return {
        "eeg_segments": segments,
        "labels": labels,
        "patients": patients,
        "epoch_keys": epoch_keys,
        "sfreq": SFREQ,
        "metadata": metadata,
    }


def clean_segments_with_ica(
    eeg_segments: np.ndarray,
    *,
    sfreq: int = SFREQ,
    apply_ica: bool = True,
    n_components: int = 4,
    exclude: tuple[int, ...] = (0, 1),
) -> tuple[np.ndarray, dict[str, Any]]:
    import mne

    if not apply_ica:
        return eeg_segments.astype(np.float32), {"filter": "skipped", "ica": "skipped", "ica_exclude": []}

    n_epochs, n_channels, n_times = eeg_segments.shape
    raw_data = np.concatenate([epoch for epoch in eeg_segments], axis=1)
    ch_names = [f"EEG{i + 1}" for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(raw_data, info, verbose="ERROR")
    raw.filter(2, 40, method="iir", verbose="ERROR")

    component_count = min(n_components, n_channels)
    ica = mne.preprocessing.ICA(n_components=component_count, random_state=42, max_iter="auto")
    ica.fit(raw, verbose="ERROR")
    ica.exclude = [idx for idx in exclude if idx < component_count]
    cleaned = ica.apply(raw.copy(), verbose="ERROR").get_data()
    cleaned_segments = cleaned.reshape(n_channels, n_epochs, n_times).transpose(1, 0, 2)
    return cleaned_segments.astype(np.float32), {
        "filter": "iir_2_40hz",
        "ica": "mne.preprocessing.ICA",
        "ica_components": component_count,
        "ica_exclude": list(ica.exclude),
        "ica_note": "Fixed exclusion of components [0, 1] follows the competition document assumption for frontal artifacts.",
    }


def extract_features_from_segments(eeg_segments: np.ndarray, *, sfreq: int = SFREQ) -> dict[str, np.ndarray]:
    from scipy import signal

    features: list[list[float]] = []
    psd_for_plot: list[np.ndarray] = []
    freqs = None

    for epoch in eeg_segments:
        epoch_features: list[float] = []
        channel_psd: list[np.ndarray] = []
        for sig in epoch:
            time_feats = [
                np.mean(sig),
                np.std(sig),
                np.var(sig),
                np.max(sig) - np.min(sig),
                np.median(sig),
                np.mean(np.abs(sig)),
            ]
            diff_feats = [np.mean(np.diff(sig)), np.std(np.diff(sig))]
            nperseg = min(64, len(sig) // 2)
            freqs, pxx = signal.welch(sig, fs=sfreq, nperseg=nperseg)
            alpha = np.mean(pxx[(freqs >= 8) & (freqs <= 13)]) if np.any((freqs >= 8) & (freqs <= 13)) else 0.0
            beta = np.mean(pxx[(freqs >= 13) & (freqs <= 30)]) if np.any((freqs >= 13) & (freqs <= 30)) else 0.0
            epoch_features.extend(time_feats + diff_feats + [alpha, beta])
            channel_psd.append(pxx)
        features.append(epoch_features)
        psd_for_plot.append(np.mean(channel_psd, axis=0))

    return {
        "features": np.asarray(features, dtype=np.float32),
        "freqs": np.asarray(freqs, dtype=np.float32),
        "psd": np.asarray(psd_for_plot, dtype=np.float32),
    }


def prepare_dataset(
    data_path: str | Path | None = None,
    *,
    mode: str = "demo",
    max_epochs: int | None = None,
    max_rows: int | None = None,
    apply_ica: bool = True,
) -> dict[str, Any]:
    dataset = load_epoch_aligned_segments(
        data_path,
        mode=mode,
        max_epochs=max_epochs,
        max_rows=max_rows,
    )
    cleaned, preprocessing = clean_segments_with_ica(
        dataset["eeg_segments"],
        sfreq=dataset["sfreq"],
        apply_ica=apply_ica,
    )
    extracted = extract_features_from_segments(cleaned, sfreq=dataset["sfreq"])
    dataset["eeg_segments"] = cleaned
    dataset.update(extracted)
    dataset["metadata"]["preprocessing"] = preprocessing
    dataset["metadata"]["feature_dim"] = int(dataset["features"].shape[1])
    return dataset


def save_feature_cache(mode: str, dataset: dict[str, Any]) -> Path:
    import joblib

    path = get_feature_cache_path(mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dataset, path)
    return path


def load_feature_cache(mode: str) -> dict[str, Any]:
    import joblib

    return joblib.load(get_feature_cache_path(mode))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
