
import os
from typing import Tuple, List, Optional

import numpy as np
import wfdb
from scipy.signal import resample
from sklearn.utils import resample as sk_resample
from tqdm import tqdm


DEFAULT_MITBIH_RECORDS = [
    "04015", "04043", "04048", "04126", "04746",
    "04908", "04936", "05091", "05121", "05261",
    "06426", "06453", "06995", "07162", "07859",
    "07879", "07910", "08215", "08219", "08378",
]


def load_mitbih_afib(
    mit_path: str,
    records: Optional[List[str]] = None,
    target_fs: int = 360,
    window_sec: int = 10,
    normal_downsample: Optional[int] = 300,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load, resample, segment, and label MIT-BIH AFib records.

    Parameters
    ----------
    mit_path : str
        Directory containing WFDB record files (.dat, .hea, .atr).
    records : List[str], optional
        Record names to process. Defaults to 20 core records.
    target_fs : int
        Target sampling frequency in Hz (default: 360 Hz).
    window_sec : int
        Segment duration in seconds (default: 10 seconds).
    normal_downsample : int, optional
        Maximum number of normal segments to retain.
        Set to None to disable downsampling.
    random_state : int
        Random seed for reproducible downsampling.

    Returns
    -------
    X_mit : np.ndarray
        Signal array of shape (N, target_fs * window_sec, 1).
    y_mit : np.ndarray
        Binary labels of shape (N,).
        0 = Normal, 1 = AFib.
    """
    if records is None:
        records = DEFAULT_MITBIH_RECORDS

    window_samples = target_fs * window_sec
    segments = []
    labels = []

    for rec in tqdm(records, desc="Processing MIT-BIH AFib Records"):
        rec_file = os.path.join(mit_path, rec)

        # Load signal channel 0 and annotations
        record = wfdb.rdrecord(rec_file, channels=[0])
        annotation = wfdb.rdann(rec_file, "atr")

        sig = record.p_signal[:, 0]
        fs = int(record.fs)

        # Resample signal to the target frequency if necessary
        if fs != target_fs:
            sig = resample(
                sig,
                int(len(sig) * target_fs / fs),
            )

        # Extract non-overlapping windows
        for start in range(
            0,
            len(sig) - window_samples + 1,
            window_samples,
        ):
            end = start + window_samples

            # Identify AFib annotations within this window
            anns = [
                note
                for sample, note in zip(
                    annotation.sample,
                    annotation.aux_note,
                )
                if start <= sample < end
            ]

            label = 1 if any("AFIB" in note.upper() for note in anns) else 0

            segments.append(sig[start:end])
            labels.append(label)

    if not segments:
        raise ValueError("No ECG segments were extracted. Check the records and data path.")

    segments = np.asarray(segments, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)

    # Reshape to (N, time_steps, 1)
    X = segments[..., np.newaxis]
    y = labels

    # Optionally downsample the majority normal class
    normal_idx = np.where(y == 0)[0]
    afib_idx = np.where(y == 1)[0]

    if (
        normal_downsample is not None
        and len(normal_idx) > normal_downsample
    ):
        normal_down = sk_resample(
            normal_idx,
            replace=False,
            n_samples=normal_downsample,
            random_state=random_state,
        )

        balanced_idx = np.concatenate([normal_down, afib_idx])

        rng = np.random.default_rng(random_state)
        rng.shuffle(balanced_idx)

        X = X[balanced_idx]
        y = y[balanced_idx]

    return X, y
