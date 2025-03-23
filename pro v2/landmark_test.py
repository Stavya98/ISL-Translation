import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Load your video (replace with your file path)
video_path = 'D:/code/Mini/Adjectives/5. Beautiful/MVI_9569.MOV'  # Adjust this
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video at {video_path}")
    exit()

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
    
    # Get frame dimensions for scaling coordinates
    h, w, _ = image.shape
    
    # Draw and label Pose landmarks (first 5 points)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        for i, lm in enumerate(results.pose_landmarks.landmark[:5]):  # First 5 points
            x, y = int(lm.x * w), int(lm.y * h)
            label = f"{i}: ({lm.x:.2f}, {lm.y:.2f}, {lm.z:.2f})"
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    
    # Draw and label Left Hand landmarks (first 5 points)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        for i, lm in enumerate(results.left_hand_landmarks.landmark[:5]):
            x, y = int(lm.x * w), int(lm.y * h)
            label = f"{i}: ({lm.x:.2f}, {lm.y:.2f}, {lm.z:.2f})"
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
    
    # Draw and label Right Hand landmarks (first 5 points)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        for i, lm in enumerate(results.right_hand_landmarks.landmark[:5]):
            x, y = int(lm.x * w), int(lm.y * h)
            label = f"{i}: ({lm.x:.2f}, {lm.y:.2f}, {lm.z:.2f})"
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    
    # Display the frame with landmarks and labels
    cv2.imshow('Landmarks on Video', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
holistic.close()
print(f"Processed {frame_count} frames.")


