import os
import librosa
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow_hub as hub
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import json

# Preprocess audio files
def preprocess_audio(file_path, target_sr=16000):
    try:
        audio, sr = librosa.load(file_path, sr=None, mono=True)
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        return audio, target_sr
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None

# Label mapping
RAVDESS_LABELS = {
    1: 'neutral', 
    2: 'calm', 
    3: 'happy', 
    4: 'sad', 
    5: 'angry', 
    6: 'fearful', 
    7: 'disgust', 
    8: 'surprised'
}

TARGET_LABELS = {
    'neutral': 'Tender',
    'calm': 'Tender',
    'happy': 'Happy',
    'sad': 'Sad',
    'angry': 'Angry',
    'fearful': 'Scary',
    'disgust': 'disgust',
    'surprised': 'surprised',
}

MOOD_INDEX = {
    'Happy': 0,
    'Sad': 1,
    'Tender': 2,
    'Exciting': 3,
    'Angry': 4,
    'Scary': 5
}

def map_labels(label):
    return TARGET_LABELS.get(RAVDESS_LABELS[label], None)

# Load YAMNet model
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

def extract_features(audio, yamnet_model):
    try:
        audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)
        _, embeddings, _ = yamnet_model(audio_tensor)
        return embeddings.numpy()
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

# Prepare dataset 
def prepare_dataset(dataset_path):
    features, labels = [], []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                try:
                    label = int(file.split('-')[2])  # Extract label from filename
                    mapped_label = map_labels(label)  # Map the label to the target
                    if mapped_label:
                        # Preprocess the audio
                        audio, sr = preprocess_audio(file_path)
                        if audio is None or len(audio) == 0:
                            print(f"Skipping invalid audio: {file_path}")
                            continue
                        
                        # Extract features from the original audio
                        embeddings = extract_features(audio, yamnet_model)
                        if embeddings is not None:
                            features.append(np.mean(embeddings, axis=0))
                            labels.append(mapped_label)
                except Exception as e:
                    print(f"Error processing file {file}: {e}")
    return np.array(features), pd.get_dummies(labels).values

# Plot training history
def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Training and Evaluation
def train_and_evaluate_model(features, labels, initial_epochs=50, model_save_path=None):
    # Split the dataset
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Build the model
    input_shape = features.shape[1]
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(2048, activation='relu'),  # Increased neurons for better learning capacity
        BatchNormalization(),  # Added batch normalization for stability
        Dropout(0.5),  # Increased dropout to prevent overfitting
        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(labels.shape[1], activation='softmax')  # Output layer
    ])

    # Compile model
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    # Train the model
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    history = model.fit(X_train, y_train, 
                        validation_data=(X_val, y_val), 
                        epochs=initial_epochs, 
                        batch_size=32,
                        callbacks=[early_stopping, reduce_lr])

    # Save the model if a save path is provided
    if model_save_path:
        model.save(model_save_path)
        print(f"Model saved to {model_save_path}")

    # Evaluate the model
    val_loss, val_acc = model.evaluate(X_val, y_val)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

    # Plot training history
    plot_training_history(history)

    return model

# Predict emotions for every chunk
def predict_emotions_by_chunks(file_path, yamnet_model, model, chunk_duration=5, output_json="predictions.json"):
    audio, sr = preprocess_audio(file_path)
    if audio is None:
        print(f"Invalid audio file: {file_path}")
        return
    chunk_samples = int(chunk_duration * sr)
    num_chunks = len(audio) // chunk_samples
    mood_sequence = []

    print(f"Processing {num_chunks} chunks of {chunk_duration} seconds each...")
    
    for i in range(num_chunks):
        start = i * chunk_samples
        end = start + chunk_samples
        chunk = audio[start:end]
        
        if len(chunk) < chunk_samples:
            continue
        
        embeddings = extract_features(chunk, yamnet_model)
        if embeddings is None:
            continue
        
        mean_embedding = np.mean(embeddings, axis=0).reshape(1, -1)
        prediction = model.predict(mean_embedding)
        predicted_index = np.argmax(prediction)
        
        if predicted_index >= 0 and predicted_index < len(MOOD_INDEX):
            mood_sequence.append(predicted_index)
            print(f"Chunk {i + 1}: Index {predicted_index}")
        else:
            print(f"Chunk {i + 1}: Invalid Index {predicted_index} (Skipping)")

    # Save predictions to JSON
    save_predictions_to_json(mood_sequence, output_json)

    return mood_sequence

# Save predictions to JSON
def save_predictions_to_json(mood_sequence, output_file):
    predictions_dict = {"mood": "".join(map(str, mood_sequence))}
    with open(output_file, "w") as f:
        json.dump(predictions_dict, f, indent=4)
    print(f"Predictions saved to {output_file}")

# Main script
dataset_path = "/Users/elmira/Desktop/AML/Audio_Song_Actors_01-24"
song_path = "/Users/elmira/Desktop/YAMNet/smallville-music_radiohead-creep.mp3"
output_json_path = "/Users/elmira/Desktop/YAMNet/emotion_predictions.json"

# Prepare dataset
features, labels = prepare_dataset(dataset_path)

# Train the model on the original dataset and save it
initial_model_path = "/Users/elmira/Desktop/YAMNet/initial_model.h5"
model = train_and_evaluate_model(features, labels, initial_epochs=50, model_save_path=initial_model_path)

# Predict emotions for a song and save to JSON
chunk_predictions = predict_emotions_by_chunks(
    file_path=song_path, 
    yamnet_model=yamnet_model, 
    model=model, 
    chunk_duration=5, 
    output_json=output_json_path
)

# Display all predictions
print("\nMood sequence:")
print("".join(map(str, chunk_predictions)))
