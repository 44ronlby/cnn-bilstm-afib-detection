
from typing import Tuple, Optional
from pathlib import Path

import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def combine_datasets(
    X_mit: np.ndarray,
    y_mit: np.ndarray,
    X_ptb: np.ndarray,
    y_ptb: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Concatenate processed MIT-BIH and PTB-XL ECG signals and labels.
    """
    if len(X_mit) != len(y_mit) or len(X_ptb) != len(y_ptb):
        raise ValueError("Each dataset must have matching signals and labels.")

    if X_mit.shape[1:] != X_ptb.shape[1:]:
        raise ValueError("MIT-BIH and PTB-XL signals must have matching shapes.")

    X_all = np.concatenate([X_ptb, X_mit], axis=0)
    y_all = np.concatenate([y_ptb, y_mit], axis=0)

    print(f"Combined dataset shape: {X_all.shape}")
    print(
        f"Combined class distribution: "
        f"Normal={np.sum(y_all == 0)}, AFib={np.sum(y_all == 1)}"
    )

    return X_all, y_all


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    deterministic: bool = True,
    test_ratio: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the dataset into training and test sets.

    If deterministic is True, every fifth sample is assigned to the
    test set. Otherwise, use a stratified random split.
    """
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of samples.")

    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1.")

    if deterministic:
        step = round(1 / test_ratio)
        indices = np.arange(len(X))

        test_idx = indices[::step]
        train_idx = np.delete(indices, test_idx)

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_ratio,
            stratify=y,
            random_state=random_state,
        )

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test


def scale_signals(
    X_train: np.ndarray,
    X_test: np.ndarray,
    scaler_save_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Flatten ECG signals, fit StandardScaler on training data only,
    transform both sets, and restore the original 3D shapes.
    """
    if X_train.ndim != 3 or X_test.ndim != 3:
        raise ValueError("Expected 3D ECG arrays: (samples, time_steps, channels).")

    if X_train.shape[1:] != X_test.shape[1:]:
        raise ValueError("Training and test signals must have matching shapes.")

    n_train, steps, channels = X_train.shape
    n_test = X_test.shape[0]

    X_train_flat = X_train.reshape(n_train, -1)
    X_test_flat = X_test.reshape(n_test, -1)

    # Fit only on training data to prevent data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)

    # Restore 3D shapes
    X_train_scaled = X_train_scaled.reshape(n_train, steps, channels)
    X_test_scaled = X_test_scaled.reshape(n_test, steps, channels)

    if scaler_save_path:
        save_path = Path(scaler_save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, save_path)
        print(f"Fitted scaler saved to '{save_path}'")

    return X_train_scaled, X_test_scaled, scaler


def apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE to flattened training signals, then restore 3D shape.
    """
    if X_train.ndim != 3:
        raise ValueError("Expected a 3D training array.")

    if len(X_train) != len(y_train):
        raise ValueError("Training signals and labels must have equal lengths.")

    n_samples, steps, channels = X_train.shape
    X_train_flat = X_train.reshape(n_samples, -1)

    smote = SMOTE(random_state=random_state)
    X_train_bal_flat, y_train_bal = smote.fit_resample(
        X_train_flat,
        y_train,
    )

    X_train_bal = X_train_bal_flat.reshape(-1, steps, channels)

    print(f"Balanced training shape: {X_train_bal.shape}")
    print(
        f"Class counts after SMOTE: "
        f"Normal={np.sum(y_train_bal == 0)}, "
        f"AFib={np.sum(y_train_bal == 1)}"
    )

    return X_train_bal, y_train_bal


def run_preprocessing_pipeline(
    X_mit: np.ndarray,
    y_mit: np.ndarray,
    X_ptb: np.ndarray,
    y_ptb: np.ndarray,
    scaler_save_path: Optional[str] = "saved_models/scaler.pkl",
    deterministic_split: bool = True,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    StandardScaler,
]:
    """
    Run the complete preprocessing pipeline:
    Combine -> Split -> Scale -> SMOTE.
    """

    # 1. Combine datasets
    X_all, y_all = combine_datasets(X_mit, y_mit, X_ptb, y_ptb)

    # 2. Split into training and test sets
    X_train, X_test, y_train, y_test = split_data(
        X_all,
        y_all,
        deterministic=deterministic_split,
    )

    # 3. Scale signals using training data statistics
    X_train_scaled, X_test_scaled, scaler = scale_signals(
        X_train,
        X_test,
        scaler_save_path=scaler_save_path,
    )

    # 4. Balance training data only
    X_train_bal, y_train_bal = apply_smote(
        X_train_scaled,
        y_train,
    )

    return (
        X_train_bal,
        y_train_bal,
        X_test_scaled,
        y_test,
        scaler,
    )
