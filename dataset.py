import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm
import mediapipe as mp


class ExerciseDataset(Dataset):
    """PyTorch dataset for exercise sequences."""
    
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # labels are expected to be a multi-hot vector (list/ndarray)
        return torch.FloatTensor(self.sequences[idx]), torch.FloatTensor(self.labels[idx])


class VideoProcessor:
    """Process videos to extract pose landmarks."""
    
    def __init__(self, sequence_length, frame_skip, visualize=False):
        self.sequence_length = sequence_length
        self.frame_skip = frame_skip
        self.visualize = visualize
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def extract_landmarks(self, video_path):
        """Extract pose landmarks from a video file."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return []
        
        landmarks_list = []
        frame_count = 0
        
        if self.visualize:
            print(f"  Processing: {video_path.name}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames based on frame_skip parameter
            if frame_count % self.frame_skip != 0:
                frame_count += 1
                continue
            
            # Convert BGR to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)
            
            if results.pose_landmarks:
                # Extract x, y coordinates and visibility score
                landmarks = [[lm.x, lm.y, lm.visibility] for lm in results.pose_landmarks.landmark]
                landmarks_list.append(np.array(landmarks, dtype=np.float32))
                
                # Visualization mode
                if self.visualize:
                    # Draw landmarks on frame
                    annotated_frame = frame.copy()
                    self.mp_drawing.draw_landmarks(
                        annotated_frame,
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                    
                    # Add text overlay
                    cv2.putText(annotated_frame, f"Landmarks: 33 detected", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"Video: {video_path.name}", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Show frame
                    cv2.imshow('Pose Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
            frame_count += 1
        
        cap.release()
        if self.visualize:
            cv2.destroyAllWindows()
        
        return landmarks_list
    
    def create_sequences(self, landmarks_list):
        """Create fixed-length sequences from landmarks using sliding window."""
        sequences = []
        flattened = [lm.flatten() for lm in landmarks_list]
        
        for i in range(len(flattened) - self.sequence_length + 1):
            sequences.append(np.array(flattened[i:i + self.sequence_length]))
        
        return sequences
    
    def process_video(self, video_path):
        """Process a single video file and return sequences."""
        landmarks = self.extract_landmarks(video_path)
        if len(landmarks) < self.sequence_length:
            return []
        return self.create_sequences(landmarks)
    
    def __del__(self):
        """Cleanup MediaPipe pose detector."""
        self.pose.close()


class DataLoader:
    """Load training data from organized video folders."""
    
    def __init__(self, training_dir, processor):
        self.training_dir = Path(training_dir)
        self.processor = processor
    
    def load(self):
        """Load all videos and extract sequences."""
        exercise_folders = [d for d in self.training_dir.iterdir() if d.is_dir()]
        exercise_names = sorted([d.name for d in exercise_folders])

        if not exercise_names:
            raise ValueError(f"No exercise folders found in {self.training_dir}")

        # For each exercise, detect its defect/label subfolders (e.g. 'good', 'rounded_back')
        exercise_data = {}

        for exercise_name in exercise_names:
            exercise_dir = self.training_dir / exercise_name

            # find subfolders inside the exercise folder; these represent labels
            label_folders = [d for d in exercise_dir.iterdir() if d.is_dir()]

            # If no label subfolders, treat video files directly as a single default label 'default'
            if not label_folders:
                label_names = ['default']
                label_paths = { 'default': exercise_dir }
            else:
                label_names = sorted([d.name for d in label_folders])
                label_paths = {name: exercise_dir / name for name in label_names}

            label_to_index = {name: idx for idx, name in enumerate(label_names)}
            class_to_label = {idx: name for name, idx in label_to_index.items()}

            all_sequences = []
            all_labels = []

            # For each label, collect videos
            for label_name, label_path in label_paths.items():
                video_files = list(label_path.glob('*.mp4')) + list(label_path.glob('*.avi')) + \
                              list(label_path.glob('*.mov')) + list(label_path.glob('*.mkv'))

                if not video_files:
                    # skip empty label folder but keep label mapping
                    continue

                for video_path in tqdm(video_files, desc=f"Processing {exercise_name}/{label_name}"):
                    sequences = self.processor.process_video(video_path)
                    if sequences:
                        all_sequences.extend(sequences)
                        # create multi-hot label vectors for each sequence (single-label examples set single bit)
                        idx = label_to_index[label_name]
                        for _ in range(len(sequences)):
                            lbl = np.zeros(len(label_names), dtype=np.float32)
                            lbl[idx] = 1.0
                            all_labels.append(lbl)

            if not all_sequences:
                print(f"⚠️  No sequences found for exercise '{exercise_name}' - skipping")
                continue

            exercise_data[exercise_name] = (all_sequences, all_labels, class_to_label)

        if not exercise_data:
            raise ValueError("No sequences extracted from any exercise folders")

        return exercise_data