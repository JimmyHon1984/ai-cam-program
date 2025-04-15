import yaml
import argparse
import os
import sys
from ultralytics import YOLO
from datetime import datetime

def load_config(config_path):
    """Loads training configuration from a YAML file."""
    print(f"Loading configuration from: {config_path}")
    if not os.path.exists(config_path):
        print(f"ERROR: Configuration file not found at {config_path}")
        sys.exit(1)
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("Configuration loaded successfully.")
        return config
    except Exception as e:
        print(f"ERROR: Failed to load or parse configuration file: {e}")
        sys.exit(1)

def check_paths(config):
    """Checks if essential paths in the config exist."""
    print("Checking paths...")
    dataset_yaml = config.get('dataset_yaml_path')
    output_dir = config.get('output_dir')

    if not dataset_yaml or not os.path.exists(dataset_yaml):
        print(f"ERROR: Dataset YAML path '{dataset_yaml}' not found or not specified in config.")
        return False

    if not output_dir:
        print("ERROR: Output directory 'output_dir' not specified in config.")
        return False

    # Create output directory if it doesn't exist
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory ensured: {output_dir}")
    except Exception as e:
        print(f"ERROR: Could not create output directory '{output_dir}': {e}")
        return False

    print("Path checks passed.")
    return True


def run_training(config):
    """Runs the YOLO training process based on the loaded configuration."""
    print("\n--- Starting Training ---")

    if not check_paths(config):
        print("Exiting due to path errors.")
        sys.exit(1)

    # Extract parameters from config, providing defaults if necessary
    base_model = config.get('base_model', 'yolov8n.pt')
    dataset_yaml = config['dataset_yaml_path']
    epochs = config.get('epochs', 50)
    img_size = config.get('img_size', 640)
    batch_size = config.get('batch_size', 16)
    output_dir = config['output_dir']
    run_name = config.get('run_name', f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    device = config.get('device', '0') # Default to GPU 0
    optimizer = config.get('optimizer', 'auto')
    lr0 = config.get('learning_rate', None) # Let Ultralytics handle if None
    patience = config.get('patience', 50) # Default patience for early stopping

    print(f"Parameters:")
    print(f"  Base Model: {base_model}")
    print(f"  Dataset YAML: {dataset_yaml}")
    print(f"  Epochs: {epochs}")
    print(f"  Image Size: {img_size}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Device: {device}")
    print(f"  Optimizer: {optimizer}")
    print(f"  Learning Rate (lr0): {'Default' if lr0 is None else lr0}")
    print(f"  Patience: {patience}")
    print(f"  Output Directory (Project): {output_dir}")
    print(f"  Run Name: {run_name}")

    try:
        # Load the base model
        model = YOLO(base_model)

        # Start training
        print("\nInitiating model.train()...")
        results = model.train(
            data=dataset_yaml,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            project=output_dir,
            name=run_name,
            device=device,
            optimizer=optimizer,
            lr0=lr0, # Pass learning rate if specified
            patience=patience,
            # Add other parameters from config as needed
            # workers=config.get('workers', 8)
        )
        print("\nTraining completed.")
        print(f"Results, models, and evaluation reports saved in: {os.path.join(output_dir, run_name)}")

        # The 'results' object contains various metrics, and files are saved automatically
        # including best.pt, last.pt, results.csv, confusion_matrix.png, etc.

    except Exception as e:
        print(f"\nERROR: An error occurred during training: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback
        sys.exit(1)

    print("--- Training Script Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO Training based on YAML configuration.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the training configuration YAML file (e.g., config.yaml)."
    )
    args = parser.parse_args()

    # Load configuration
    training_config = load_config(args.config)

    # Run training
    run_training(training_config)