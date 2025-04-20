import cv2
import os

# 📁 Path to your dataset (organized by words)
dataset_path = "D:/code/Mini/Final data"  # e.g. "./dataset/"

# 📊 Initialize
max_frames = 0
longest_video = ""

# 📁 Loop through folders (each word)
for label in os.listdir(dataset_path):
    word_folder = os.path.join(dataset_path, label)
    if not os.path.isdir(word_folder):
        continue
    
    # 🎥 Loop through video files in each folder
    for video_file in os.listdir(word_folder):
        video_path = os.path.join(word_folder, video_file)

        # Open video
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Track max
        if frame_count > max_frames:
            max_frames = frame_count
            longest_video = video_path

print(f"✅ Max frame count: {max_frames}")
print(f"🎥 Longest video: {longest_video}")
