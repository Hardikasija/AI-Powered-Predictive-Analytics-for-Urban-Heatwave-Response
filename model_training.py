"""Training and evaluation for urban heatwave forecasting models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from data_preprocessing import prepare_dataset, time_based_split

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

try:
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
except Exception:  # pragma: no cover
    tf = None
    Sequential = None
    EarlyStopping = None
    Input = None
    LSTM = None
    Dense = None
    Dropout = None


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    forecast_horizon: int = 3
    test_size: float = 0.2
    random_state: int = 42
    cv_splits: int = 5
    alert_threshold: float = 0.55
    lstm_sequence_length: int = 14
    epochs: int = 15
    batch_size: int = 32
    data_source: str = "real"


def _build_preprocessor(feature_columns: List[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(transformers=[("num", numeric_pipeline, feature_columns)])


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


def _cross_validate_model(
    estimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: int,
) -> Dict[str, float]:
    cv = TimeSeriesSplit(n_splits=cv_splits)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    scores = cross_validate(estimator, X_train, y_train, cv=cv, scoring=scoring, n_jobs=None)
    return {f"cv_{metric}": float(np.nanmean(scores[f'test_{metric}'])) for metric in scoring}


def _save_json(payload: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _plot_temperature_trend(df: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_df = df.groupby("date", as_index=False)["temperature_avg"].mean()

    plt.figure(figsize=(12, 5))
    sns.lineplot(data=plot_df, x="date", y="temperature_avg", color="#c0392b")
    plt.title("Average Urban Temperature Trend")
    plt.xlabel("Date")
    plt.ylabel("Temperature Avg (C)")
    plt.tight_layout()

    path = output_dir / "temperature_trends.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def _plot_predictions_vs_actual(
    results_df: pd.DataFrame,
    output_dir: Path,
    forecast_horizon: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_slice = results_df.sort_values("date").tail(180).copy()

    plt.figure(figsize=(12, 5))
    plt.plot(latest_slice["date"], latest_slice["actual"], label="Actual", linewidth=2, color="#2c3e50")
    plt.plot(
        latest_slice["date"],
        latest_slice["predicted_probability"],
        label="Predicted Probability",
        linewidth=2,
        color="#e67e22",
    )
    plt.title(f"Predicted Heatwave Probability vs Actual Events ({forecast_horizon}-Day Horizon)")
    plt.xlabel("Date")
    plt.ylabel("Probability / Event")
    plt.legend()
    plt.tight_layout()

    path = output_dir / f"predictions_vs_actual_horizon_{forecast_horizon}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def _plot_roc_curve(
    curves: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    output_dir: Path,
    forecast_horizon: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Baseline")

    for model_name, (fpr, tpr, auc_score) in curves.items():
        plt.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC={auc_score:.3f})")

    plt.title(f"ROC Curve for Heatwave Forecasting ({forecast_horizon}-Day Horizon)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()

    path = output_dir / f"roc_curve_horizon_{forecast_horizon}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def _make_lstm_sequences(
    frame: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
    sequence_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    sequences_x: List[np.ndarray] = []
    sequences_y: List[int] = []

    for _, group in frame.groupby("zone"):
        group = group.sort_values("date").reset_index(drop=True)
        values = group[feature_columns].to_numpy()
        targets = group[target_column].to_numpy()

        if len(group) <= sequence_length:
            continue

        for index in range(sequence_length, len(group)):
            sequences_x.append(values[index - sequence_length : index])
            sequences_y.append(int(targets[index]))

    return np.asarray(sequences_x, dtype=np.float32), np.asarray(sequences_y, dtype=np.int32)


def _build_lstm(input_shape: Tuple[int, int]) -> Sequential:
    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(64, return_sequences=False),
            Dropout(0.25),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_and_evaluate(config: TrainingConfig | None = None) -> Dict[str, object]:
    """Train Random Forest, XGBoost, and LSTM models and save all artifacts."""

    config = config or TrainingConfig()
    forecast_horizon = config.forecast_horizon

    prepared_df, feature_columns, target_column = prepare_dataset(
        forecast_horizon=forecast_horizon,
        data_source=config.data_source,
    )
    train_df, test_df = time_based_split(prepared_df, target_column=target_column, test_size=config.test_size)

    X_train = train_df[feature_columns]
    y_train = train_df[target_column]
    X_test = test_df[feature_columns]
    y_test = test_df[target_column]

    artifact_root = Path("artifacts")
    model_dir = artifact_root / "models"
    report_dir = artifact_root / "reports"
    output_dir = Path("outputs")
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_map = {int(label): float(weight) for label, weight in zip(classes, class_weights)}
    positive_weight = class_weight_map.get(1, 1.0)

    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", _build_preprocessor(feature_columns)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=12,
                    min_samples_leaf=3,
                    random_state=config.random_state,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    models: Dict[str, object] = {"random_forest": rf_pipeline}
    all_metrics: Dict[str, Dict[str, float | str]] = {}
    if XGBClassifier is not None:
        xgb_pipeline = Pipeline(
            steps=[
                ("preprocessor", _build_preprocessor(feature_columns)),
                (
                    "classifier",
                    XGBClassifier(
                        n_estimators=350,
                        max_depth=5,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        eval_metric="logloss",
                        random_state=config.random_state,
                        scale_pos_weight=max(positive_weight, 1.0),
                    ),
                ),
            ]
        )
        models["xgboost"] = xgb_pipeline
    else:
        all_metrics["xgboost"] = {"status": "xgboost not installed"}

    comparison_rows: List[Dict[str, float | str]] = []
    roc_curves: Dict[str, Tuple[np.ndarray, np.ndarray, float]] = {}
    best_model_name = None
    best_auc = -np.inf
    best_result_df = None

    for model_name, model in models.items():
        cv_metrics = _cross_validate_model(model, X_train, y_train, config.cv_splits)
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= config.alert_threshold).astype(int)
        test_metrics = _classification_metrics(y_test.to_numpy(), y_pred, y_prob)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = test_metrics["roc_auc"]
        roc_curves[model_name] = (fpr, tpr, auc_score)

        metrics = {
            "model": model_name,
            "threshold": config.alert_threshold,
            **cv_metrics,
            **test_metrics,
        }
        all_metrics[model_name] = metrics
        comparison_rows.append(metrics)

        joblib.dump(model, model_dir / f"{model_name}_horizon_{forecast_horizon}.joblib")

        result_df = test_df[["date", "zone", target_column]].copy()
        result_df["predicted_probability"] = y_prob
        result_df["predicted_label"] = y_pred
        result_df = result_df.rename(columns={target_column: "actual"})
        result_df.to_csv(report_dir / f"{model_name}_predictions_horizon_{forecast_horizon}.csv", index=False)

        if np.isfinite(auc_score) and auc_score > best_auc:
            best_auc = auc_score
            best_model_name = model_name
            best_result_df = result_df

    if tf is not None:
        lstm_preprocessor = _build_preprocessor(feature_columns)
        scaled_train = lstm_preprocessor.fit_transform(X_train)
        scaled_test = lstm_preprocessor.transform(X_test)

        scaled_train_df = train_df[["date", "zone", target_column]].copy()
        scaled_test_df = test_df[["date", "zone", target_column]].copy()
        scaled_train_df[feature_columns] = scaled_train
        scaled_test_df[feature_columns] = scaled_test

        X_train_seq, y_train_seq = _make_lstm_sequences(
            scaled_train_df, feature_columns, target_column, config.lstm_sequence_length
        )
        X_test_seq, y_test_seq = _make_lstm_sequences(
            scaled_test_df, feature_columns, target_column, config.lstm_sequence_length
        )

        if len(X_train_seq) > 0 and len(X_test_seq) > 0 and len(np.unique(y_train_seq)) > 1:
            tf.random.set_seed(config.random_state)
            lstm_model = _build_lstm((X_train_seq.shape[1], X_train_seq.shape[2]))
            early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

            sequence_classes = np.unique(y_train_seq)
            sequence_weights = compute_class_weight(
                class_weight="balanced",
                classes=sequence_classes,
                y=y_train_seq,
            )
            sequence_weight_map = {
                int(label): float(weight) for label, weight in zip(sequence_classes, sequence_weights)
            }

            lstm_model.fit(
                X_train_seq,
                y_train_seq,
                validation_split=0.15,
                epochs=config.epochs,
                batch_size=config.batch_size,
                verbose=0,
                callbacks=[early_stopping],
                class_weight=sequence_weight_map,
            )

            y_prob = lstm_model.predict(X_test_seq, verbose=0).reshape(-1)
            y_pred = (y_prob >= config.alert_threshold).astype(int)
            test_metrics = _classification_metrics(y_test_seq, y_pred, y_prob)

            placeholder_cv = {f"cv_{metric}": float("nan") for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]}
            metrics = {"model": "lstm", "threshold": config.alert_threshold, **placeholder_cv, **test_metrics}
            all_metrics["lstm"] = metrics
            comparison_rows.append(metrics)

            fpr, tpr, _ = roc_curve(y_test_seq, y_prob)
            roc_curves["lstm"] = (fpr, tpr, test_metrics["roc_auc"])

            lstm_model.save(model_dir / f"lstm_horizon_{forecast_horizon}.keras")
            joblib.dump(lstm_preprocessor, model_dir / f"lstm_preprocessor_horizon_{forecast_horizon}.joblib")

            aligned_blocks = []
            for _, block in scaled_test_df.groupby("zone"):
                trimmed = block.iloc[config.lstm_sequence_length:].copy()
                if not trimmed.empty:
                    aligned_blocks.append(trimmed)

            aligned_test = pd.concat(aligned_blocks, ignore_index=True) if aligned_blocks else pd.DataFrame()
            aligned_test = aligned_test[["date", "zone", target_column]].copy()
            aligned_test["predicted_probability"] = y_prob
            aligned_test["predicted_label"] = y_pred
            aligned_test = aligned_test.rename(columns={target_column: "actual"})
            aligned_test.to_csv(report_dir / f"lstm_predictions_horizon_{forecast_horizon}.csv", index=False)

            auc_score = test_metrics["roc_auc"]
            if np.isfinite(auc_score) and auc_score > best_auc:
                best_auc = auc_score
                best_model_name = "lstm"
                best_result_df = aligned_test
        else:
            all_metrics["lstm"] = {"status": "insufficient sequence data for LSTM training"}
    else:
        all_metrics["lstm"] = {"status": "tensorflow not installed"}

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(report_dir / f"comparison_horizon_{forecast_horizon}.csv", index=False)
    _save_json(all_metrics, report_dir / f"metrics_horizon_{forecast_horizon}.json")

    if best_result_df is not None:
        _plot_predictions_vs_actual(best_result_df, output_dir, forecast_horizon)
    if roc_curves:
        _plot_roc_curve(roc_curves, output_dir, forecast_horizon)
    _plot_temperature_trend(prepared_df, output_dir)

    metadata = {
        "forecast_horizon": forecast_horizon,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "best_model": best_model_name or "random_forest",
        "alert_threshold": config.alert_threshold,
        "lstm_sequence_length": config.lstm_sequence_length,
        "data_source": config.data_source,
    }
    _save_json(metadata, artifact_root / "metadata.json")

    if best_model_name is not None:
        best_metrics = all_metrics.get(best_model_name, {})
        report_text = classification_report(
            y_test,
            (models[best_model_name].predict_proba(X_test)[:, 1] >= config.alert_threshold).astype(int),
            zero_division=0,
        ) if best_model_name in models else "Sequence-based classification report is stored in CSV outputs."
        confusion = (
            confusion_matrix(
                y_test,
                (models[best_model_name].predict_proba(X_test)[:, 1] >= config.alert_threshold).astype(int),
            ).tolist()
            if best_model_name in models
            else []
        )
        diagnostics = {
            "best_model": best_model_name,
            "best_metrics": best_metrics,
            "confusion_matrix": confusion,
            "classification_report": report_text,
        }
        _save_json(diagnostics, report_dir / f"best_model_diagnostics_horizon_{forecast_horizon}.json")

    return {
        "prepared_data": prepared_df,
        "train_data": train_df,
        "test_data": test_df,
        "metrics": all_metrics,
        "metadata": metadata,
    }
