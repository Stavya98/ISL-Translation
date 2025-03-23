import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Dataset and output paths
dataset_dir = 'D:/code/Mini/Adjectives'  # Adjust if your path is different
output_dir = 'D:/code/Mini/pro v2/landmarks_output'
os.makedirs(output_dir, exist_ok=True)

# List of adjective folders
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

# Process each adjective folder
for adjective in adjectives:
    adjective_path = os.path.join(dataset_dir, adjective)
    if not os.path.exists(adjective_path):
        print(f"Skipping {adjective_path} - directory does not exist.")
        continue
    
    # Create output directory for this adjective
    output_adjective_dir = os.path.join(output_dir, adjective)
    os.makedirs(output_adjective_dir, exist_ok=True)
    
    # Process each video in the adjective folder
    for video_file in os.listdir(adjective_path):
        if not video_file.endswith('.MOV'):  # Check for .MOV files
            continue
        
        video_path = os.path.join(adjective_path, video_file)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Skipping {video_path} - could not open video.")
            continue
        
        # List to store landmarks for this video
        landmark_data = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            # Convert BGR to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            
            # Convert back to BGR for display
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            frame_landmarks = []
            
            # Pose landmarks (33 points: x, y, z, visibility)
            if results.pose_landmarks:
                pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]).flatten()  # 132 values
            else:
                pose = np.zeros(33 * 4)
            
            # Left hand landmarks (21 points: x, y, z)
            if results.left_hand_landmarks:
                left_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()  # 63 values
            else:
                left_hand = np.zeros(21 * 3)
            
            # Right hand landmarks (21 points: x, y, z)
            if results.right_hand_landmarks:
                right_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()  # 63 values
            else:
                right_hand = np.zeros(21 * 3)
            
            # Combine landmarks (258 values per frame)
            frame_landmarks = np.concatenate([pose, left_hand, right_hand])
            landmark_data.append(frame_landmarks)
        
        # Save landmarks to a .npy file
        output_file = os.path.join(output_adjective_dir, f"{video_file[:-4]}_landmarks.npy")
        np.save(output_file, np.array(landmark_data))
        print(f"Processed {video_path}: {frame_count} frames -> {output_file}")
        
        cap.release()

holistic.close()
print("Finished processing all videos.")