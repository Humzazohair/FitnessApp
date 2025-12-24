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
        return torch.FloatTensor(self.sequences[idx]), torch.LongTensor([self.labels[idx]])[0]


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
        
        # Create class mappings
        exercise_to_class = {name: idx for idx, name in enumerate(exercise_names)}
        class_to_exercise = {idx: name for name, idx in exercise_to_class.items()}
        
        all_sequences = []
        all_labels = []
        
        # Process each exercise folder
        for exercise_name in exercise_names:
            exercise_dir = self.training_dir / exercise_name
            video_files = list(exercise_dir.glob('*.mp4')) + list(exercise_dir.glob('*.avi')) + \
                         list(exercise_dir.glob('*.mov')) + list(exercise_dir.glob('*.mkv'))
            
            if not video_files:
                print(f"⚠️  No videos found in {exercise_dir}")
                continue
            
            # Process each video
            for video_path in tqdm(video_files, desc=f"Processing {exercise_name}"):
                sequences = self.processor.process_video(video_path)
                if sequences:
                    all_sequences.extend(sequences)
                    all_labels.extend([exercise_to_class[exercise_name]] * len(sequences))
        
        if not all_sequences:
            raise ValueError("No sequences extracted from videos")
        
        return all_sequences, all_labels, class_to_exercise