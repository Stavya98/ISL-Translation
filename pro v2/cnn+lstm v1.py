





import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split


padded_data = np.load('D:/code/Mini/pro v2/padded_data.npy')
labels = np.load('D:/code/Mini/pro v2/labels.npy')
# Assuming padded_data and labels are prepared with max_len=70
# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    padded_data, labels, test_size=0.2, random_state=42, stratify=labels
)

# Define the model
model = Sequential([
    Input(shape=(70, 258)),  # Updated to max_len=70

    # Conv1D layers with BatchNormalization
    Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    Dropout(0.3),

    # LSTM layers
    LSTM(128, return_sequences=True),
    Dropout(0.4),
    LSTM(64, return_sequences=False),
    Dropout(0.4),

    # Dense layers
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(8, activation='softmax')
])

# Compile with the same learning rate
optimizer = Adam(learning_rate=0.0003)
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Print model summary
model.summary()

# Train with early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=16,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_accuracy:.4f}")

# Save the model
model.save('isl_model_v5.keras')
print("Model saved as 'isl_model_v5.keras'.")