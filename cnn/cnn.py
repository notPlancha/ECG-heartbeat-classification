import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization

def build_cnn(input_shape, num_classes):
    """
    Builds a 1D CNN model for classification.

    Parameters:
    - input_shape: tuple, shape of the input data (e.g., (window_size, num_channels)).
    - num_classes: int, number of output classes.

    Returns:
    - model: a compiled Keras model.
    """
    model = Sequential()

    # Convolutional Layer 1
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    # Convolutional Layer 2
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    # Convolutional Layer 3
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    # Flatten and Fully Connected Layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))  # Dropout for regularization
    model.add(Dense(num_classes, activation='softmax'))  # Output layer

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',  # Use sparse_categorical_crossentropy if labels are integers
        metrics=['accuracy']
    )

    return model

# Example usage
if __name__ == "__main__":
    # Example input shape: (window_size, num_channels)
    input_shape = (200, 1)  # 200 samples per window, 1 channel (e.g., ECG signal)
    num_classes = 4  # Number of heartbeat classes (N, S, V, F)

    # Build the model
    model = build_cnn(input_shape, num_classes)

    # Print the model summary
    model.summary()