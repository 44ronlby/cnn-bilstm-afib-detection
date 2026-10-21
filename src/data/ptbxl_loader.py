
import ast
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from scipy.signal import resample
from tqdm import tqdm


def parse_afib_label(scp_str: str) -> Optional[int]:
    """
    Parse SCP codes into a binary label:
    AFIB = 1, NORM = 0, other = None.
    """
    try:
        codes = ast.literal_eval(scp_str)

        if "AFIB" in codes:
            return 1
        elif "NORM" in codes:
            return 0

    except (ValueError, SyntaxError, TypeError):
        pass

    return None


def load_ptbxl_lead2(
    meta_csv_path: str,
    signals_npy_path: str,
    lead_index: int = 1,
    orig_fs: int = 500,
    target_fs: int = 360,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load PTB-XL metadata and ECG signals, extract Lead II,
    filter AFIB/NORM labels, and resample signals.

    Parameters
    ----------
    meta_csv_path : str
        Path to 'ptbxl_database.csv'.
    signals_npy_path : str
        Path to the NumPy file containing 12-lead ECG signals.
    lead_index : int
        Index of the ECG lead to extract (default: 1 for Lead II).
    orig_fs : int
        Original sampling frequency in Hz (default: 500 Hz).
    target_fs : int
        Target sampling frequency in Hz (default: 360 Hz).

    Returns
    -------
    X_ptb_resampled : np.ndarray
        Resampled ECG signals of shape (N, target_samples, 1).
    y_ptb : np.ndarray
        Binary labels of shape (N,).
    """

    # 1. Read metadata
    ptb_meta = pd.read_csv(
        meta_csv_path,
        sep=";",
        header=0,
        index_col=0,
    )

    # 2. Parse AFIB/NORM labels
    ptb_meta["afib_label"] = ptb_meta["scp_codes"].apply(
        parse_afib_label
    )

    # Keep records with valid binary labels only
    valid_mask = ptb_meta["afib_label"].notna()

    y_ptb = ptb_meta.loc[
        valid_mask, "afib_label"
    ].to_numpy(dtype=int)

    # 3. Load ECG signals
    raw_signals = np.load(
        signals_npy_path,
        allow_pickle=False,
    )

    if raw_signals.ndim != 3:
        raise ValueError(
            "Expected ECG array with shape (records, samples, leads)."
        )

    if raw_signals.shape[0] != len(ptb_meta):
        raise ValueError(
            "The number of ECG recordings does not match "
            "the number of metadata rows."
        )

    if not 0 <= lead_index < raw_signals.shape[2]:
        raise ValueError(
            f"Invalid lead_index {lead_index}; "
            f"the dataset has {raw_signals.shape[2]} leads."
        )

    # Apply the same metadata mask to the ECG recordings
    lead_signals = raw_signals[valid_mask.to_numpy(), :, lead_index]

    # 4. Resample ECG signals
    target_len = round(
        lead_signals.shape[1] * target_fs / orig_fs
    )

    resampled_signals = []

    for sig in tqdm(
        lead_signals,
        desc="Resampling PTB-XL Signals",
    ):
        sig_resampled = resample(sig, target_len)
        resampled_signals.append(sig_resampled)

    # 5. Convert to (N, time_steps, 1)
    X_ptb_resampled = np.asarray(
        resampled_signals,
        dtype=np.float32,
    )[..., np.newaxis]

    return X_ptb_resampled, y_ptb
