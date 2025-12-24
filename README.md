# Exercise Classifier Training

Train a neural network to recognize different exercises from video using pose detection.

## What It Does

1. **Extracts pose landmarks** from exercise videos using MediaPipe
2. **Creates temporal sequences** from body keypoints (33 landmarks × 3 features: x, y, visibility)
   - **Visibility** (0-1): confidence score for how clearly each landmark is visible
   - Helps model distinguish occluded/unclear poses from clear ones
3. **Trains an LSTM/GRU/Transformer** to classify exercise types
4. **Saves the trained model** for inference

## Project Structure

```
project/
├── config.py          # Training parameters
├── dataset.py         # Video processing and data loading
├── training.py        # Model training logic
├── train.py           # Main script
├── training/          # Training videos (you create this)
│   ├── squat/
│   │   ├── video1.mp4
│   │   └── video2.mp4
│   ├── bench/
│   │   └── video1.mp4
│   └── pullup/
│       └── video1.mp4
└── model/             # Saved models (auto-generated)
    ├── exercise_classifier_lstm.pth
    └── exercise_classes.json
```

## Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### 1. Organize Your Videos

Create a `training/` folder with subfolders for each exercise type:

```bash
mkdir -p training/squat training/bench training/pullup
# Add your .mp4 videos to each folder
```

### 2. Configure Training

Edit `config.py` to adjust parameters:

```python
MODEL_TYPE = 'lstm'      # or 'gru', 'transformer'
EPOCHS = 50              # Number of training epochs
BATCH_SIZE = 32          # Batch size
SEQUENCE_LENGTH = 30     # Frames per sequence
FRAME_SKIP = 2          # Process every Nth frame
INPUT_SIZE = 99         # 33 landmarks × 3 (x, y, visibility)
VISUALIZE = False       # Set to True to see landmark detection (slower, for debugging)
```

### 3. Train

```bash
python train.py
```

### 4. Output

The script generates:
- `model/exercise_classifier_lstm.pth` - Trained model weights + metadata
- `model/exercise_classes.json` - Exercise class mapping

## Testing Your Model

After training, test your model using `inference.py`:

### Test with Webcam (Real-time)
```bash
python inference.py
```

### Test with Video File
```bash
python inference.py path/to/your/video.mp4
```

### Controls
- **q** - Quit
- **r** - Reset/restart (clear buffer or restart video)

### What You'll See
- Live skeleton tracking overlay
- Exercise prediction with confidence percentage
- Buffer status (how many frames collected)
- Color-coded predictions:
  - **Green**: High confidence (>70%)
  - **Orange**: Lower confidence (<70%)
  - **Red**: No pose detected

## What Happens During Training

```
1. Load videos from training/ folders
2. Extract pose landmarks (33 keypoints × 3 features: x, y, visibility)
3. Create sliding window sequences (e.g., 30 frames each)
4. Split into train/validation sets (80/20)
5. Train model for specified epochs
6. Save best model based on validation loss
```

## Training Output

```
Device: cuda
Total sequences: 2450
Classes: ['bench', 'pullup', 'squat']
Training LSTM with 234,567 parameters
Epoch 1/50 | Train: 1.2543/45.32% | Val: 1.1234/52.18%
Epoch 2/50 | Train: 0.9821/68.45% | Val: 0.8943/71.23%
...
Model saved: model/exercise_classifier_lstm.pth
Best validation loss: 0.3421
```

## Tips

- **More videos = better accuracy** (aim for 10+ videos per exercise)
- **Varied angles and lighting** help the model generalize
- **Consistent exercise form** in videos improves results
- **Use GPU** for faster training (install CUDA-enabled PyTorch)
- **Enable visualization** (`VISUALIZE = True`) to debug landmark detection, but expect 2-3x slower processing
  - Shows video playback with detected landmarks
  - Displays current video filename and frame count
  - Press 'q' to skip to next video

## Troubleshooting

**"No sequences extracted"**
- Check video format (MP4, AVI, MOV, MKV supported)
- Ensure videos show clear body poses
- Try reducing `FRAME_SKIP` in config

**Out of memory**
- Reduce `BATCH_SIZE` in config
- Reduce `SEQUENCE_LENGTH`
- Use a smaller model (`HIDDEN_SIZE`)

**Low accuracy**
- Add more training videos
- Increase `EPOCHS`
- Try different `MODEL_TYPE` (transformer often works best)