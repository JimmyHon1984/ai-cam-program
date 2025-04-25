# YOLOv8 Trainer with Resource Monitoring

This project provides a Python script (`train.py`) to run Ultralytics YOLOv8 model training based on a YAML configuration file (`config.yaml`). It also includes a resource monitoring feature to log CPU, RAM, and GPU usage during the training process.

## Features

*   **Configuration-Driven Training**: Easily manage training parameters via `config.yaml`.
*   **Ultralytics Based**: Leverages the powerful and widely-used `ultralytics` YOLO library.
*   **Resource Monitoring**: Periodically logs CPU usage, RAM usage, and (if available) detailed GPU utilization, memory, temperature, and power consumption during training.
*   **CSV Logging**: Saves resource usage statistics to an easy-to-analyze `resource_usage.csv` file.
*   **Structured Output**: Saves training results (model weights, logs, plots) and the resource monitoring log in a specified output directory, organized by run name.
*   **Path Checking**: Verifies that essential file paths exist before starting training.
*   **Error Handling**: Includes basic error handling and traceback printing for easier debugging.

## File Structure

```bash
├── train.py             # Main training and monitoring script
├── config.yaml          # Training configuration file (user needs to modify)
├── dataset.yaml         # YOLO dataset configuration file (user needs to modify or provide)
└── (your_dataset_folder) # Place according to the path in dataset.yaml
```

## Prerequisites

1.  **Python**: Python 3.8 or higher is recommended.
2.  **Package Manager**: `pip`.
3.  **Ultralytics YOLO**: And its dependencies.
4.  **PyYAML**: For reading `.yaml` configuration files.
5.  **psutil**: For monitoring CPU and RAM.
6.  **(Optional but highly recommended)** **NVIDIA GPU**: For GPU training and detailed monitoring.
7.  **(Optional)** **pynvml**: Python bindings for the NVIDIA Management Library (NVML) for detailed GPU monitoring. If not installed, GPU monitoring will be limited (potentially only showing PyTorch allocated memory).

## Installation

1.  **Clone or download this project's files.**

2.  **Create and activate a virtual environment** (recommended):
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Linux / macOS
    source venv/bin/activate
    ```

3.  **Install the required Python packages**:
    ```bash
    pip install ultralytics pyyaml psutil
    ```

4.  **(Optional) Install pynvml for detailed GPU monitoring**:
    ```bash
    pip install nvidia-ml-py3
    ```
    *Note: Ensure your system has the NVIDIA drivers installed.*

5.  **Prepare your dataset**: Make sure your object detection dataset is prepared in the YOLO format, and the `dataset.yaml` file correctly points to the paths for training, validation (and optionally test) images and labels.

## Configuration

1.  **`dataset.yaml`**:
    *   This is the standard Ultralytics dataset configuration file.
    *   You need to create or modify this file to point to your dataset path, define the number of classes (`nc`), and the class names (`names`).
    *   Example structure:
        ```yaml
        path: /content/ai-cam-program/datasets/your_dataset # Dataset root directory
        train: images/train  # Train images path relative to 'path'
        val: images/val    # Val images path relative to 'path'
        # test: images/test # (Optional) Test images path relative to 'path'

        # Classes
        nc: 2  # Number of classes
        names: ['class1', 'class2'] # List of class names
        ```
    *   Refer to the [Ultralytics Datasets Documentation](https://docs.ultralytics.com/datasets/) for more details.

2.  **`config.yaml`**:
    *   This is the main configuration file for this training script.
    *   **You MUST modify** the paths marked with `*** MODIFY THIS ***` to point to the correct locations in your environment.
    *   Key parameters explained:
        *   `dataset_yaml_path`: **Absolute or relative path** to your `dataset.yaml` file.
        *   `base_model`: The base model to start training from (e.g., `yolov8s.pt`) or the path to a previously trained model weights file (`.pt`).
        *   `epochs`: Total number of training epochs.
        *   `img_size`: Input image size during training.
        *   `batch_size`: Number of images per batch (adjust based on your GPU memory).
        *   `optimizer`: Optimizer type (`Adam`, `SGD`, `auto`).
        *   `learning_rate`: Learning rate (if set to `None` or omitted, handled by `auto` or Ultralytics defaults).
        *   `patience`: Early stopping patience (number of epochs with no improvement on validation metrics before stopping).
        *   `device`: Training device (`0` for the first GPU, `cpu` for CPU, or `0,1` for multi-GPU).
        *   `output_dir`: The **root directory** where all training results will be saved.
        *   `run_name`: A specific name for this training run; a subfolder with this name will be created under `output_dir`.
        *   `monitor_interval` (Optional): Sampling interval (in seconds) for resource monitoring, defaults to 5.0 seconds.

## Usage

1.  Ensure your `config.yaml` and `dataset.yaml` are correctly configured for your environment and needs.
2.  Open a terminal or command prompt.
3.  Activate your virtual environment (if you created one).
4.  Run the training script, specifying your configuration file path using the `--config` argument:

    ```bash
    python train.py --config config.yaml
    ```
    (If your `config.yaml` is in the same directory as `train.py`, you can just use the filename.)

5.  The script will:
    *   Load the configuration.
    *   Check paths.
    *   Print the training parameters.
    *   Start the resource monitor (in a background thread).
    *   Load the YOLO model.
    *   Begin the Ultralytics `model.train()` process.
    *   Stop the resource monitor after training finishes or is interrupted.
    *   Save all results to `<output_dir>/<run_name>/`.

## Output

After training completes, you will find a folder named `run_name` (as specified in `config.yaml`) inside the `output_dir`. This folder will contain:

*   `weights/`: Contains `best.pt` (best model) and `last.pt` (last epoch model) weight files.
*   `events.out.tfevents.*`: TensorBoard log files.
*   `results.csv`: CSV file containing per-epoch training metrics (like mAP, loss).
*   `results.png`: Plot visualizing the training metrics.
*   `confusion_matrix.png`: Confusion matrix plot on the validation set.
*   `args.yaml`: The final arguments used for this training run (including Ultralytics defaults).
*   **`resource_usage.csv`**: The resource monitoring log file generated by this script, containing timestamps and CPU/RAM/GPU usage statistics.

## Resource Monitoring Details

*   The `ResourceMonitor` class uses `psutil` to monitor CPU and RAM.
*   If `pynvml` (nvidia-ml-py3) is available and initializes successfully, it uses NVML to monitor each NVIDIA GPU for:
    *   GPU Utilization (%)
    *   Used/Total GPU Memory (GB)
    *   Temperature (°C)
    *   Power Usage (W)
*   If `pynvml` is unavailable, but PyTorch can detect a CUDA device, the script attempts to log GPU memory allocated and reserved by PyTorch (but cannot get utilization, temperature, or power).
*   If neither is available or no GPU is detected, the GPU-related columns will show "NA".
*   Monitoring data is appended to the `<output_dir>/<run_name>/resource_usage.csv` file.

