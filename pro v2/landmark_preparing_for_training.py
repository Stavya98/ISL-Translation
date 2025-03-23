import numpy as np
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Paths
landmarks_dir = 'D:/code/Mini/pro v2/landmarks_output'

# Adjective labels (0 to 7)
adjectives = [
    '1. loud',
    '2. quiet',
    '3. happy',
    '4. sad',
    '5. Beautiful',
    '6. Ugly',
    '7. Deaf',
    '8. Blind'
]

# Lists to store data and labels
all_data = []
labels = []

# Load landmarks for each adjective
for label_idx, adjective in enumerate(adjectives):
    adjective_path = os.path.join(landmarks_dir, adjective)
    if not os.path.exists(adjective_path):
        print(f"Skipping {adjective_path} - directory does not exist.")
        continue
    
    # Load each .npy file in the adjective folder
    for landmark_file in os.listdir(adjective_path):
        if not landmark_file.endswith('.npy'):
            continue
        
        file_path = os.path.join(adjective_path, landmark_file)
        data = np.load(file_path)
        
        # Add to lists
        all_data.append(data)
        labels.append(label_idx)
        print(f"Loaded {file_path}: {data.shape}")

# Convert labels to numpy array
labels = np.array(labels)

# Pad or truncate sequences to a fixed length
max_len = 75  # Choose based on typical video length (adjust if needed)
padded_data = pad_sequences(all_data, maxlen=max_len, padding='post', truncating='post', dtype='float32')

# Check final shapes
print(f"Final data shape: {padded_data.shape}")  # Should be (num_videos, max_len, 258)
print(f"Labels shape: {labels.shape}")  # Should be (num_videos,)
print(f"Labels: {labels[:10]}")  # First 10 labels

np.save('D:/code/Mini/pro v2/padded_data.npy', padded_data)
np.save('D:/code/Mini/pro v2/labels.npy', labels)

frame_lengths = [data.shape[0] for data in all_data]
print(f"Frame lengths: min={min(frame_lengths)}, max={max(frame_lengths)}, avg={np.mean(frame_lengths):.1f}")

unique, counts = np.unique(labels, return_counts=True)
print("Class distribution:", dict(zip(unique, counts)))