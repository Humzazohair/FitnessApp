from pathlib import Path

SEQUENCE_LENGTH = 30
INPUT_SIZE = 99  # 33 landmarks × 3 (x, y, visibility)
HIDDEN_SIZE = 128
NUM_LAYERS = 2
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
TRAIN_TEST_SPLIT = 0.8
MODEL_TYPE = 'lstm'
FRAME_SKIP = 2
VISUALIZE = True       # Set to True to see landmark detection (slower, for debugging)


BASE_DIR = Path(__file__).parent
TRAINING_DIR = BASE_DIR / 'training'
MODEL_DIR = BASE_DIR / 'model'