import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from pathlib import Path
import json

# Import from our modules
import config
from models import create_model
from dataset import ExerciseDataset, VideoProcessor, DataLoader
from training import ModelTrainer


def split_data(sequences, labels, split_ratio):
    """Split data into training and validation sets."""
    indices = np.arange(len(sequences))
    np.random.shuffle(indices)
    
    split_idx = int(len(indices) * split_ratio)
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    
    return (
        [sequences[i] for i in train_idx],
        [labels[i] for i in train_idx],
        [sequences[i] for i in val_idx],
        [labels[i] for i in val_idx]
    )


def save_artifacts(model, model_path, model_config, class_mapping):
    """Save trained model and class mappings."""
    torch.save({
        'model_state_dict': model.state_dict(),
        **model_config,
        'class_to_label': class_mapping,
        'label_to_class': {v: k for k, v in class_mapping.items()}
    }, model_path)
    
    # Save per-model class mapping next to the model
    with open(model_path.parent / f'exercise_classes_{model_config.get("exercise_name","model")}.json', 'w') as f:
        json.dump(class_mapping, f, indent=2)


def main():
    """Main training function."""
    # Create directories
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config.TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Show visualization warning
    if config.VISUALIZE:
        print("\n⚠️  Visualization enabled - processing will be slower")
        print("   Press 'q' to skip current video\n")
    
    # Load and process training data (per-exercise)
    processor = VideoProcessor(config.SEQUENCE_LENGTH, config.FRAME_SKIP, config.VISUALIZE)
    loader = DataLoader(config.TRAINING_DIR, processor)

    exercise_data = loader.load()

    for exercise_name, (sequences, labels, class_to_label) in exercise_data.items():
        print(f"\nExercise: {exercise_name} | Total sequences: {len(sequences)}")
        print(f"Labels: {list(class_to_label.values())}\n")

        # Split data for this exercise
        train_seq, train_lbl, val_seq, val_lbl = split_data(sequences, labels, config.TRAIN_TEST_SPLIT)
        print(f"Train: {len(train_seq)}, Val: {len(val_seq)}\n")

        # Create datasets and loaders
        train_dataset = ExerciseDataset(train_seq, train_lbl)
        val_dataset = ExerciseDataset(val_seq, val_lbl)

        train_loader = TorchDataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = TorchDataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

        # Create model for this exercise
        num_classes = len(class_to_label)
        model = create_model(
            model_type=config.MODEL_TYPE,
            input_size=config.INPUT_SIZE,
            hidden_size=config.HIDDEN_SIZE,
            num_classes=num_classes,
            num_layers=config.NUM_LAYERS
        )

        print(f"Training {config.MODEL_TYPE.upper()} for '{exercise_name}' with {sum(p.numel() for p in model.parameters()):,} parameters\n")

        # Train model (multi-label)
        trainer = ModelTrainer(model, device, config.LEARNING_RATE)
        trainer.multi_label = True
        trainer.criterion = torch.nn.BCEWithLogitsLoss()

        best_loss = trainer.fit(train_loader, val_loader, config.EPOCHS)

        # Save model for this exercise
        model_config = {
            'model_type': config.MODEL_TYPE,
            'input_size': config.INPUT_SIZE,
            'hidden_size': config.HIDDEN_SIZE,
            'num_classes': num_classes,
            'num_layers': config.NUM_LAYERS,
            'sequence_length': config.SEQUENCE_LENGTH,
            'exercise_name': exercise_name
        }

        model_path = config.MODEL_DIR / f'exercise_classifier_{exercise_name}_{config.MODEL_TYPE}.pth'
        save_artifacts(trainer.model, model_path, model_config, class_to_label)

        print(f"\n✓ Model saved: {model_path}")
        print(f"✓ Best validation loss: {best_loss:.4f}\n")


if __name__ == '__main__':
    main()