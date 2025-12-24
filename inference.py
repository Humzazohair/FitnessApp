import sys
import cv2
import numpy as np
import torch
from pathlib import Path
from collections import deque
import mediapipe as mp

from models import create_model


class ExerciseClassifier:
    """Real-time exercise classification."""
    
    def __init__(self, model_path, sequence_length=30):
        self.sequence_length = sequence_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model = create_model(
            model_type=checkpoint['model_type'],
            input_size=checkpoint['input_size'],
            hidden_size=checkpoint['hidden_size'],
            num_classes=checkpoint['num_classes'],
            num_layers=checkpoint['num_layers']
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.class_to_exercise = checkpoint['class_to_exercise']
        print(f"Model loaded! Classes: {list(self.class_to_exercise.values())}")
        
        # MediaPipe setup
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Sequence buffer
        self.landmark_buffer = deque(maxlen=sequence_length)
        
    def extract_landmarks(self, frame):
        """Extract landmarks from a frame."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if results.pose_landmarks:
            landmarks = [[lm.x, lm.y, lm.visibility] for lm in results.pose_landmarks.landmark]
            return np.array(landmarks, dtype=np.float32), results.pose_landmarks
        return None, None
    
    def predict(self, landmarks):
        """Predict exercise type from landmark sequence."""
        if len(self.landmark_buffer) < self.sequence_length:
            return None, None
        
        # Prepare sequence
        sequence = np.array([lm.flatten() for lm in self.landmark_buffer])
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(sequence_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        exercise = self.class_to_exercise[predicted.item()]
        return exercise, confidence.item()
    
    def process_frame(self, frame):
        """Process a single frame and return annotated frame with prediction."""
        # Extract landmarks
        landmarks, pose_landmarks = self.extract_landmarks(frame)
        
        annotated_frame = frame.copy()
        
        if landmarks is not None:
            # Draw skeleton
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # Add to buffer
            self.landmark_buffer.append(landmarks)
            
            # Predict if we have enough frames
            exercise, confidence = self.predict(landmarks)
            
            if exercise:
                # Display prediction
                text = f"{exercise.upper()}: {confidence*100:.1f}%"
                color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
                cv2.putText(annotated_frame, text, (10, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            else:
                cv2.putText(annotated_frame, "Collecting frames...", (10, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        else:
            cv2.putText(annotated_frame, "No pose detected", (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        # Show buffer status
        buffer_text = f"Buffer: {len(self.landmark_buffer)}/{self.sequence_length}"
        cv2.putText(annotated_frame, buffer_text, (10, annotated_frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_frame
    
    def run_video(self, video_path):
        """Run inference on a video file."""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        print(f"Processing video: {video_path}")
        print("Press 'q' to quit, 'r' to restart")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Video ended, restarting...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.landmark_buffer.clear()
                continue
            
            annotated_frame = self.process_frame(frame)
            cv2.imshow('Exercise Classification', annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.landmark_buffer.clear()
        
        cap.release()
        cv2.destroyAllWindows()
    
    def run_webcam(self):
        """Run inference on webcam feed."""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Starting webcam...")
        print("Press 'q' to quit, 'r' to reset buffer")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame")
                break
            
            annotated_frame = self.process_frame(frame)
            cv2.imshow('Exercise Classification', annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.landmark_buffer.clear()
                print("Buffer reset")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def __del__(self):
        """Cleanup."""
        self.pose.close()


def main():
    """Main inference function."""
    model_dir = Path(__file__).parent / 'model'
    
    # Find model file
    model_files = list(model_dir.glob('exercise_classifier_*.pth'))
    if not model_files:
        print("Error: No trained model found in 'model/' directory")
        print("Please train a model first using: python main.py")
        return
    
    model_path = model_files[0]
    
    # Create classifier
    classifier = ExerciseClassifier(model_path)
    
    # Run on video or webcam
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        if not Path(video_path).exists():
            print(f"Error: Video file not found: {video_path}")
            return
        classifier.run_video(video_path)
    else:
        classifier.run_webcam()


if __name__ == '__main__':
    main()