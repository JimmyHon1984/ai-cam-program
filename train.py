import yaml
import argparse
import os
import sys
from ultralytics import YOLO
from datetime import datetime
import psutil
import threading
import time
import csv
import torch
import traceback

# For NVIDIA GPU monitoring
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("Warning: pynvml not available. GPU monitoring will be limited.")
    print("You may need to run: pip install nvidia-ml-py3")


class ResourceMonitor:
    """Monitors and logs system resource usage (CPU, RAM, GPU)."""
    
    def __init__(self, log_path, interval=1.0):
        """
        Initialize the resource monitor.
        
        Args:
            log_path: Path to save the resource usage log
            interval: Sampling interval in seconds
        """
        self.log_path = log_path
        self.interval = interval
        self.stop_flag = threading.Event()
        self.monitoring_thread = None
        
        # Initialize NVML if available
        self.nvml_initialized = False
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvml_initialized = True
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                print(f"GPU monitoring enabled for {self.gpu_count} devices")
            except Exception as e:
                print(f"Warning: Failed to initialize NVML: {e}")
        
    def _monitor_resources(self):
        """Monitor system resources and write to log file."""
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        
        # Create log file and write header
        with open(self.log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Timestamp', 
                'Elapsed_Time_Sec',
                'CPU_Usage_Percent', 
                'RAM_Used_GB', 
                'RAM_Total_GB',
                'GPU_ID', 
                'GPU_Utilization_Percent', 
                'GPU_Memory_Used_GB', 
                'GPU_Memory_Total_GB', 
                'GPU_Temperature_C',
                'GPU_Power_W'
            ])
        
        start_time = time.time()
        
        while not self.stop_flag.is_set():
            try:
                # Get current timestamp
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                elapsed_time = time.time() - start_time
                
                # Get CPU usage
                cpu_percent = psutil.cpu_percent(interval=0.1)
                
                # Get RAM usage
                ram = psutil.virtual_memory()
                ram_used_gb = ram.used / (1024 ** 3)
                ram_total_gb = ram.total / (1024 ** 3)
                
                # Open file for writing resource data
                with open(self.log_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                
                    # Get GPU stats if NVML is available
                    if self.nvml_initialized:
                        for gpu_id in range(self.gpu_count):
                            try:
                                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                                
                                # GPU utilization
                                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                                gpu_util = utilization.gpu
                                
                                # GPU memory
                                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                                gpu_memory_used = memory_info.used / (1024 ** 3)  # Convert to GB
                                gpu_memory_total = memory_info.total / (1024 ** 3)  # Convert to GB
                                
                                # GPU temperature
                                try:
                                    gpu_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                                except:
                                    gpu_temp = 0
                                
                                # GPU power usage
                                try:
                                    power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert mW to W
                                except:
                                    power_usage = 0
                                
                                writer.writerow([
                                    timestamp,
                                    round(elapsed_time, 1),
                                    cpu_percent,
                                    round(ram_used_gb, 2),
                                    round(ram_total_gb, 2),
                                    gpu_id,
                                    gpu_util,
                                    round(gpu_memory_used, 2),
                                    round(gpu_memory_total, 2),
                                    gpu_temp,
                                    round(power_usage, 2)
                                ])
                            except Exception as e:
                                print(f"Warning: Error monitoring GPU {gpu_id}: {e}")
                    else:
                        # If NVML not available, try to get basic GPU info from torch
                        if torch.cuda.is_available():
                            try:
                                gpu_count = torch.cuda.device_count()
                                for gpu_id in range(gpu_count):
                                    # Set the current device
                                    torch.cuda.set_device(gpu_id)
                                    
                                    # Get basic memory info
                                    gpu_memory_allocated = torch.cuda.memory_allocated(gpu_id) / (1024 ** 3)  # Convert to GB
                                    gpu_memory_reserved = torch.cuda.memory_reserved(gpu_id) / (1024 ** 3)  # Convert to GB
                                    
                                    writer.writerow([
                                        timestamp,
                                        round(elapsed_time, 1),
                                        cpu_percent,
                                        round(ram_used_gb, 2),
                                        round(ram_total_gb, 2),
                                        gpu_id,
                                        "NA",  # No utilization info
                                        round(gpu_memory_allocated, 2),
                                        round(gpu_memory_reserved, 2),
                                        "NA",  # No temperature info
                                        "NA"   # No power info
                                    ])
                            except Exception as e:
                                print(f"Warning: Error monitoring GPU with PyTorch: {e}")
                        else:
                            # No GPU monitoring available
                            writer.writerow([
                                timestamp,
                                round(elapsed_time, 1),
                                cpu_percent,
                                round(ram_used_gb, 2),
                                round(ram_total_gb, 2),
                                "NA",
                                "NA",
                                "NA",
                                "NA",
                                "NA",
                                "NA"
                            ])
                
                # Sleep until next interval
                time.sleep(self.interval)
                
            except Exception as e:
                print(f"Warning: Resource monitoring error: {e}")
                time.sleep(self.interval)  # Still sleep to avoid tight loops on errors
    
    def start(self):
        """Start the resource monitoring thread."""
        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            self.stop_flag.clear()
            self.monitoring_thread = threading.Thread(target=self._monitor_resources)
            self.monitoring_thread.daemon = True  # Thread will exit when main program exits
            self.monitoring_thread.start()
            print(f"Resource monitoring started. Logging to: {self.log_path}")
        else:
            print("Resource monitoring is already running")
    
    def stop(self):
        """Stop the resource monitoring thread."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.stop_flag.set()
            self.monitoring_thread.join(timeout=5)
            
            # Shutdown NVML if it was initialized
            if self.nvml_initialized:
                try:
                    pynvml.nvmlShutdown()
                except:
                    pass
                
            print("Resource monitoring stopped.")
        else:
            print("Resource monitoring is not running")


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
    device = config.get('device', '0')  # Default to GPU 0
    optimizer = config.get('optimizer', 'auto')
    lr0 = config.get('learning_rate', None)  # Let Ultralytics handle if None
    patience = config.get('patience', 50)  # Default patience for early stopping
    
    # Set up resource monitoring
    monitor_interval = config.get('monitor_interval', 5.0)  # Default to 5 seconds
    run_output_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_output_dir, exist_ok=True)
    resource_log_path = os.path.join(run_output_dir, 'resource_usage.csv')

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
    print(f"  Resource Monitoring Interval: {monitor_interval} seconds")
    print(f"  Resource Log Path: {resource_log_path}")

    # Initialize resource monitor
    resource_monitor = ResourceMonitor(resource_log_path, interval=monitor_interval)
    
    try:
        # Start resource monitoring
        resource_monitor.start()
        
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
            lr0=lr0,  # Pass learning rate if specified
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
        traceback.print_exc()  # Print detailed traceback
        sys.exit(1)
    finally:
        # Stop resource monitoring regardless of whether training succeeded or failed
        resource_monitor.stop()
        print(f"Resource monitoring data saved to: {resource_log_path}")

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