
import os
from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential

from src.data.mitbih_loader import load_mitbih_afib
from src.data.ptbxl_loader import load_ptbxl_lead2
from src.data.preprocessing import run_preprocessing_pipeline
from src.models.cnn_bilstm import build_cnn_bilstm


def get_callbacks(
    checkpoint_path: str = "saved_models/cnn_bilstm_ckpt.keras",
    patience: int = 5
) -> list:
    """
    Creates EarlyStopping and ModelCheckpoint callbacks.

    Parameters:
    -----------
    checkpoint_path : str
        Path where the best model weights will be saved.
    patience : int
        Number of epochs with no improvement after which training will be stopped.
    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]

    return callbacks


def train_model(
    mit_path: str = "data/raw/mitbih_afib/files/",
    ptb_csv_path: str = "data/raw/ptbxl_database.csv",
    ptb_signals_path: str = "data/raw/ecgeq-500hzsrfava.npy",
    model_save_path: str = "saved_models/MabuhAI.keras",
    batch_size: int = 64,
    epochs: int = 100,
    learning_rate: float = 0.001,
    patience: int = 5,
    val_split: float = 0.2
) -> Tuple[Sequential, tf.keras.callbacks.History]:
    """
    Full end-to-end training pipeline:
    1. Loads raw MIT-BIH & PTB-XL ECG records.
    2. Runs preprocessing (Scale, Split, SMOTE).
    3. Builds CNN-BiLSTM architecture.
    4. Trains model with EarlyStopping and ModelCheckpoint.
    5. Saves final model artifact.
    """
    print("=" * 60)
    print("STEP 1: Loading Datasets...")
    print("=" * 60)

    X_mit, y_mit = load_mitbih_afib(mit_path)
    X_ptb, y_ptb = load_ptbxl_lead2(
        ptb_csv_path,
        ptb_signals_path
    )

    print("\n" + "=" * 60)
    print("STEP 2: Preprocessing & Class Balancing (SMOTE)...")
    print("=" * 60)

    X_train_bal, y_train_bal, X_test, y_test, scaler = (
        run_preprocessing_pipeline(
            X_mit,
            y_mit,
            X_ptb,
            y_ptb
        )
    )

    print("\n" + "=" * 60)
    print("STEP 3: Building CNN-BiLSTM Model...")
    print("=" * 60)

    input_shape = (
        X_train_bal.shape[1],
        X_train_bal.shape[2]
    )  # (3600, 1)

    model = build_cnn_bilstm(
        input_shape=input_shape,
        learning_rate=learning_rate
    )
    model.summary()

    print("\n" + "=" * 60)
    print("STEP 4: Training Model...")
    print("=" * 60)

    callbacks = get_callbacks(
        checkpoint_path=model_save_path,
        patience=patience
    )

    history = model.fit(
        X_train_bal,
        y_train_bal,
        validation_split=val_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Save final model state
    os.makedirs(
        os.path.dirname(model_save_path),
        exist_ok=True
    )
    model.save(model_save_path)

    print(f"\nModel training complete! Saved to '{model_save_path}'")

    return model, history


if __name__ == "__main__":
    train_model()
