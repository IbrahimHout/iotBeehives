import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import librosa

# --- Constants ---
DATA_DIR = 'data/Dataset/Bee Hive Audios/'
MODEL_PATH = 'models/beehive_model.h5'
SAMPLE_RATE = 22050
DURATION = 5
N_MELS = 128

# --- Data Loading and Preprocessing ---
def load_and_preprocess_data():
    """
    Loads the audio data, extracts features, and prepares it for training.
    """
    labels = []
    features = []

    for label_type in ['QueenBee Present', 'QueenBee Absent']:
        path = os.path.join(DATA_DIR, label_type)
        if not os.path.exists(path):
            print(f"Directory not found: {path}")
            print("Please make sure you have downloaded and extracted the dataset correctly.")
            return None, None
        
        for filename in os.listdir(path):
            if filename.endswith('.wav'):
                filepath = os.path.join(path, filename)
                try:
                    # Load audio file
                    audio, _ = librosa.load(filepath, sr=SAMPLE_RATE, duration=DURATION)
                    
                    # Extract Mel spectrogram
                    mel_spec = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
                    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                    
                    # Append features and labels (keep as variable-length for now)
                    features.append(mel_spec_db)
                    labels.append(1 if label_type == 'QueenBee Present' else 0)
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")

    # At this point, "features" is a list of 2D arrays with potentially different
    # time dimensions (number of columns). We need to make them the same shape
    # so that np.array(features) produces a proper 3D tensor instead of an
    # inhomogeneous object array.
    if not features:
        return None, None

    # Determine the maximum time length among all spectrograms
    max_time_length = max(f.shape[1] for f in features)

    # Initialize a zero-padded array: (num_samples, n_mels, max_time_length)
    padded_features = np.zeros((len(features), N_MELS, max_time_length), dtype=np.float32)

    for i, f in enumerate(features):
        # If some spectrograms are shorter, they will be zero-padded at the end.
        # If any are longer (shouldn't really happen with fixed DURATION), truncate.
        time_len = min(f.shape[1], max_time_length)
        padded_features[i, :, :time_len] = f[:, :time_len]

    return padded_features, np.array(labels)

# --- Model Definition ---
def build_model(input_shape):
    """
    Builds the CNN model for audio classification.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

# --- Main Training Logic ---
if __name__ == '__main__':
    # Load and preprocess the data
    features, labels = load_and_preprocess_data()
    
    if features is not None and labels is not None:
        # Split the data into training and validation sets
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        
        # Reshape the data for the CNN
        X_train = X_train[..., np.newaxis]
        X_test = X_test[..., np.newaxis]
        
        # Build the model
        input_shape = X_train.shape[1:]
        model = build_model(input_shape)
        
        # Print model summary
        model.summary()
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=10,
            batch_size=32
        )
        
        # Save the trained model
        model.save(MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

