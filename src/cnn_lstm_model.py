import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dropout, Dense

def build_cnn_lstm_model(input_shape: tuple) -> Sequential:
    """
    Build a hybrid CNN-LSTM model for time-series forecasting.
    
    Args:
        input_shape: Tuple representing (time_steps, features). Default is usually (60, 1).
        
    Returns:
        model: Compiled Keras Sequential model.
    """
    model = Sequential()
    
    # CNN portion for feature extraction
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    
    # LSTM portion for sequence learning
    model.add(LSTM(50, return_sequences=False))
    
    # Regularization
    model.add(Dropout(0.2))
    
    # Output layer
    model.add(Dense(1))
    
    # Compile
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

def train_model(model: Sequential, X_train, y_train, epochs: int = 20, batch_size: int = 32) -> Sequential:
    """
    Train the CNN-LSTM model.
    
    Args:
        model: Compiled Keras model.
        X_train: Training data features.
        y_train: Training data targets.
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        
    Returns:
        model: Trained Keras model.
    """
    print(f"Training CNN-LSTM for {epochs} epochs with batch size {batch_size}...")
    model.fit(
        X_train, 
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,  # Hold out 10% for validation during training
        verbose=1
    )
    print("Training successfully completed.")
    
    return model
