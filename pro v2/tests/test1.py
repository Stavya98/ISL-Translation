import numpy as np
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model
model = load_model('D:/code/Mini/pro v2/isl_model_v8.keras')
print("Model loaded successfully.")

# Initialize MediaPipe Holistic and Drawing utilities
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Adjective labels
adjectives = ['1. loud', '2. quiet', '3. happy', '4. sad', '5. Beautiful', '6. Ugly', '7. Deaf', '8. Blind']

# Function to extract landmarks from a video
def extract_landmarks_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    landmarks = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)
        
        # Extract landmarks
        frame_landmarks = []
        # Pose landmarks
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                frame_landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
        else:
            frame_landmarks.extend([0.0] * 132)  # 33 landmarks × 4
        
        # Left hand landmarks
        if results.left_hand_landmarks:
            for landmark in results.left_hand_landmarks.landmark:
                frame_landmarks.extend([landmark.x, landmark.y, landmark.z])
        else:
            frame_landmarks.extend([0.0] * 63)  # 21 landmarks × 3
        
        # Right hand landmarks
        if results.right_hand_landmarks:
            for landmark in results.right_hand_landmarks.landmark:
                frame_landmarks.extend([landmark.x, landmark.y, landmark.z])
        else:
            frame_landmarks.extend([0.0] * 63)  # 21 landmarks × 3
        
        landmarks.append(frame_landmarks)
    
    cap.release()
    return np.array(landmarks)

# Function to preprocess landmarks (same as training)
def preprocess_landmarks(landmarks, max_len=80):
    # Pad or truncate to max_len
    padded_data = pad_sequences([landmarks], maxlen=max_len, padding='post', truncating='post', dtype='float32')[0]
    
    # Smooth the data (moving average)
    def moving_average(data, window_size=3):
        smoothed_data = np.copy(data)
        for j in range(data.shape[1]):  # Iterate over features
            smoothed_data[:, j] = np.convolve(data[:, j], np.ones(window_size)/window_size, mode='same')
        return smoothed_data
    
    padded_data = moving_average(padded_data, window_size=3)
    
    # Normalize (using approximate mean and std from training data)
    # Ideally, you should load the mean and std from training
    padded_data_reshaped = padded_data.reshape(-1, 258)
    mean = np.mean(padded_data_reshaped, axis=0)
    std = np.std(padded_data_reshaped, axis=0)
    std[std == 0] = 1
    padded_data_reshaped = (padded_data_reshaped - mean) / std
    padded_data = padded_data_reshaped.reshape(max_len, 258)
    
    # Reshape for model input (1, max_len, 258)
    return np.expand_dims(padded_data, axis=0)

# Function to predict the adjective
def predict_adjective(landmarks):
    if landmarks.shape[0] == 0:
        return "Error: No landmarks detected", 0.0
    
    # Preprocess landmarks
    processed_landmarks = preprocess_landmarks(landmarks)
    
    # Make prediction
    prediction = model.predict(processed_landmarks, verbose=0)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_label = adjectives[predicted_class]
    confidence = prediction[0][predicted_class]
    
    return predicted_label, confidence

# Function to play video, plot landmarks, and save the output
def play_video_with_landmarks(video_path, save_output=True, output_path='output_video1.mp4'):
    # First, extract landmarks and predict the adjective
    landmarks = extract_landmarks_from_video(video_path)
    predicted_label, confidence = predict_adjective(landmarks)
    print(f"Predicted adjective: {predicted_label} (Confidence: {confidence:.4f})")
    
    # Reopen the video to play it with landmarks
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties for saving
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Initialize video writer if saving is enabled
    if save_output:
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)
        
        # Convert back to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        # Draw landmarks with enhanced visibility
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame_bgr, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=4, circle_radius=6),  # Green points, thicker
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=3)  # Red connections
            )
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame_bgr, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 165, 0), thickness=4, circle_radius=6),  # Orange points
                mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=3)  # Yellow connections
            )
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame_bgr, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 165, 0), thickness=4, circle_radius=6),  # Orange points
                mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=3)  # Yellow connections
            )
        
        # Display the prediction on the frame with a background box for better visibility
        text = f"Prediction: {predicted_label} ({confidence:.2f})"
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.rectangle(frame_bgr, (10, 10), (10 + text_width, 30 + text_height), (0, 0, 0), -1)  # Black background
        cv2.putText(frame_bgr, text, (10, 30 + text_height), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Show the frame
        cv2.imshow('Video with Landmarks', frame_bgr)
        
        # Write the frame to the output video
        if save_output:
            out.write(frame_bgr)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    if save_output:
        out.release()
        print(f"Output video saved as {output_path}")
    cv2.destroyAllWindows()
    holistic.close()

# Test on a new video
video_path = 'D:/code/Mini/Adjectives/4. sad/MVI_9565.MOV'
play_video_with_landmarks(video_path, save_output=True, output_path='D:/code/Mini/pro v2/output_video1.mp4')