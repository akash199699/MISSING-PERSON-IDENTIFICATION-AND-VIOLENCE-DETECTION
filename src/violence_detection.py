import os
import cv2
import torch
import numpy as np
import tempfile
from PIL import Image
import torchvision.models.video as models
import torchvision.transforms as transforms
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import from your custom modules
from utils import select_files
from report_generation import export_violence_report

### SECTION 3: VIOLENCE DETECTION

class ViolenceDetectionModel(nn.Module):
    """3D CNN model for violence detection in video clips"""
    def __init__(self, num_classes=2):
        super(ViolenceDetectionModel, self).__init__()
        self.base_model = models.r3d_18(weights="DEFAULT")
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

def preprocess_clip(clip):
    """Preprocess video clip for violence detection"""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    processed_clip = []
    for frame in clip:
        processed_frame = transform(frame)
        processed_clip.append(processed_frame)

    clip_tensor = torch.stack(processed_clip, dim=0)
    clip_tensor = clip_tensor.permute(1, 0, 2, 3).unsqueeze(0)

    return clip_tensor

def extract_video_clips(video_path, clip_length=16, overlap=8):
    """Extract clips from a video with optional overlap"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    clips = []
    clip_start_times = []
    buffer = []

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        buffer.append(rgb_frame)

        if len(buffer) == clip_length:
            clips.append(buffer.copy())
            clip_start_times.append(frame_count - clip_length + 1)

            if overlap > 0:
                buffer = buffer[clip_length - overlap:]
            else:
                buffer = []

        frame_count += 1

    cap.release()
    return clips, clip_start_times, fps

def load_violence_detection_model(device):
    """Load or create violence detection model"""
    print("Setting up Violence Detection Model...")
    model = ViolenceDetectionModel().to(device)
    model.eval()
    return model

def detect_violence_in_clip(clip, start_time, fps, model, device, threshold=0.65):
    """Detect violence in a single clip"""
    clip_tensor = preprocess_clip(clip)
    clip_tensor = clip_tensor.to(device)

    with torch.no_grad():
        outputs = model(clip_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        violence_prob = probabilities[0][1].item()  # Assuming class 1 is violence

        if violence_prob > threshold:
            time_in_seconds = start_time / fps
            return {
                'time': time_in_seconds,
                'probability': violence_prob,
                'frame_idx': start_time,
                'thumbnail': clip[0]  # First frame as thumbnail
            }
    return None

def detect_violence_in_video(video_path, model, device, threshold=0.65):
    """Detect violence in a video file using multi-threading"""
    clips, clip_start_times, fps = extract_video_clips(video_path)

    violence_detections = []
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(detect_violence_in_clip, clip, start_time, fps, model, device, threshold)
            for clip, start_time in zip(clips, clip_start_times)
        ]
        for future in as_completed(futures):
            result = future.result()
            if result:
                violence_detections.append(result)

    # Open a window to display the video
    cv2.namedWindow("Violence Detection", cv2.WINDOW_NORMAL)
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Display the frame with detection information
        for det in violence_detections:
            if det['frame_idx'] == frame_idx:
                cv2.putText(frame, f"Violence Detected: {det['probability']:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Violence Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    return violence_detections

def run_violence_detection(video_files):
    """Main function to run the violence detection pipeline"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_violence_detection_model(device)

    for video_file in video_files:
        print(f"Analyzing {video_file} for violence...")
        violence_detections = detect_violence_in_video(video_file, model, device)

        if violence_detections:
            print(f"Found {len(violence_detections)} violent clips in {video_file}")
            export_violence_report(violence_detections, video_file)
        else:
            print(f"No violence detected in {video_file}")

    print("Violence detection complete!")
    return