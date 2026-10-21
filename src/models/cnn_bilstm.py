
from typing import Tuple, Optional
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
    BatchNormalization,
    Input
)
from tensorflow.keras.optimizers import Adam


def build_cnn_bilstm(
    input_shape: Tuple[Optional[int], int] = (None, 1),
    learning_rate: float = 0.001
) -> Sequential:
    """
    Builds and compiles the 1D CNN + Bidirectional LSTM model for AFib detection.

    Architecture:
    -------------
    - Conv1D (64 filters, kernel=5, ReLU) + MaxPooling1D (pool=2)
    - Conv1D (128 filters, kernel=5, ReLU) + MaxPooling1D (pool=2)
    - BatchNormalization
    - Bidirectional LSTM (64 units, return_sequences=True)
    - Bidirectional LSTM (32 units)
    - Dense (16 units, ReLU) + Dropout (0.5)
    - Output Dense (1 unit, Sigmoid)

    Parameters:
    -----------
    input_shape : Tuple[Optional[int], int]
        Shape of input time series (time_steps, channels).
        Default: (None, 1).
    learning_rate : float
        Adam optimizer learning rate (default: 0.001).

    Returns:
    --------
    model : tf.keras.models.Sequential
        Compiled Keras model ready for training or evaluation.
    """

    model = Sequential([
        # --- CNN Feature Extractor ---
        Conv1D(64, kernel_size=5, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(128, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        BatchNormalization(),

        # --- Bidirectional LSTM Layers ---
        Bidirectional(LSTM(64, return_sequences=True)),
        Bidirectional(LSTM(32)),

        # --- Classification Head ---
        Dense(16, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ], name="CNN_BiLSTM_AFib")

    # Compile with Adam optimizer, binary crossentropy, and clinical metrics
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall', 'AUC']
    )

    return model


if __name__ == "__main__":
    # Test instantiating model and printing summary
    model = build_cnn_bilstm(input_shape=(3600, 1))
    model.summary()
