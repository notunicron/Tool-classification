# Cholec80 Surgical Tool Classification using YOLOv8

This project uses the Ultralytics YOLOv8 classification model, fine-tuned with PyTorch, to classify surgical tools (Clipper, Grasper, Hook, Scissor) present in images from a subset of the Cholec80 dataset.

## Overview

The goal is to train a convolutional neural network to automatically identify which of the four specified surgical tools is visible in a given image frame. This script handles:
1.  Loading a pre-trained YOLOv8 classification model (`yolov8n-cls.pt` by default).
2.  Fine-tuning the model on the provided training and validation sets.
3.  Evaluating the trained model's performance on the test set.
4.  Providing an example of how to predict the tool class for a single image.

## Features
*   Uses state-of-the-art YOLOv8 classification models.
*   Leverages transfer learning from ImageNet pre-training.
*   Simple setup using the `ultralytics` pip package.
*   Includes training, validation, and testing phases.
*   Calculates Top-1 accuracy on the full test set.
*   Demonstrates single-image prediction.

## Dataset

The project dataset was taken from

https://www.kaggle.com/datasets/aditya9790/chloectinytools

This project is designed for a specific subset of the Cholec80 dataset, structured as follows:
```
cholec-tinytools/
├── test/
│ ├── clipper/
│ │ └── *.png
│ ├── grasper/
│ │ └── *.png
│ ├── hook/
│ │ └── *.png
│ └── scissor/
│ └── *.png
├── train/
│ ├── clipper/
│ │ └── *.png
│ ├── grasper/
│ │ └── *.png
│ ├── hook/
│ │ └── *.png
│ └── scissor/
│ └── *.png
└── validation/
├── clipper/
│ └── *.png
├── grasper/
│ └── *.png
├── hook/
│ └── *.png
└── scissor/
└── *.png
```

*   The base directory name (`cholec-tinytools`) can be changed, but the `train`, `validation`, and `test` subdirectories are expected.
*   Each of these subdirectories must contain folders named exactly after the classes: `clipper`, `grasper`, `hook`, `scissor`.
*   Images for each class should reside within their respective class folders. Supported formats include `.png` and `.jpg`.

## Results:
- Least loss on train data: 0.0153
- Accuracy on test data: 0.9833

## Requirements

*   Python 3.8+ (tested with 3.12)
*   PyTorch (CPU or GPU version)
*   Torchvision
*   Ultralytics YOLO
*   Matplotlib (for displaying sample prediction)
*   tqdm (for progress bars)

**Installation:**

1.  **Create and activate a virtual environment (Recommended):**
    ```bash
    python -m venv myenv
    source myenv/bin/activate  # Linux/macOS
    # myenv\Scripts\activate  # Windows
    ```

2.  **Install PyTorch and Torchvision:**
    *   Visit the official PyTorch website: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
    *   Select your OS, Package (`Pip`), Language (`Python`), and Compute Platform (CPU or appropriate CUDA version).
    *   Run the generated command. For CPU, it will typically be:
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        ```
    *   *(Using the official command ensures compatibility between torch and torchvision, preventing potential runtime errors.)*

3.  **Install Ultralytics and other dependencies:**
    ```bash
    pip install ultralytics matplotlib tqdm
    ```

## Usage
1. Copy the repository.
2. Navigate to the project folder.
3. Execute following command in bash:
```bash
#!/bin/bash
curl -L -o chloectinytools.zip\
  https://www.kaggle.com/api/v1/datasets/download/aditya9790/chloectinytools
```
OR, create a python file and paste this code and execute it.
```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("aditya9790/chloectinytools")

print("Path to dataset files:", path)
```


4.  **Place your dataset:** Ensure the `cholec-tinytools` directory (or your renamed dataset directory) is accessible from where you run the script. Update the `DATASET_BASE_DIR` variable in the script if necessary.
5.  **Configure parameters (Optional):** Modify variables like `MODEL_NAME`, `EPOCHS`, `IMG_SIZE`, `BATCH_SIZE` at the top of the `classify_tools.py` script if needed.
6.  **Run the script:**
    ```bash
    python classify_tools.py
    ```

## Script Workflow

1.  **Setup:** Sets paths, configuration, and device (CPU/GPU).
2.  **Load Model:** Downloads (if needed) and loads the specified pre-trained YOLOv8 classification model.
3.  **Train:** Fine-tunes the model using the `train` and `validation` sets. Progress and results are saved to a `runs/classify/PROJECT_NAME/RUN_NAME` directory. The best model weights are saved as `best.pt`.
4.  **Evaluate:**
    *   Loads the `best.pt` weights.
    *   **Important:** It explicitly finds all images within the subdirectories of the `TEST_DIR` using `glob`.
    *   Runs `model.predict` on this *list* of files.
    *   Manually calculates the Top-1 accuracy by comparing the prediction for each image against its parent directory name (the true class).
    *   Prints the final accuracy on the full test set.
5.  **Predict Sample:** Selects a random image from the test set, performs prediction, and displays the image with the predicted class and confidence score using Matplotlib.

## Configuration Variables (in `classify_tools.py`)

*   `DATASET_BASE_DIR`: Path to the root directory containing `train/`, `validation/`, `test/`.
*   `MODEL_NAME`: YOLOv8 classification model to use (e.g., `yolov8n-cls.pt`, `yolov8s-cls.pt`).
*   `EPOCHS`: Number of training epochs.
*   `IMG_SIZE`: Image size (pixels) for input to the model.
*   `BATCH_SIZE`: Number of images per batch during training and evaluation. Reduce if you encounter memory errors.
*   `PROJECT_NAME`: Top-level folder name for saving runs.
*   `RUN_NAME`: Specific subfolder name for this training run's results.

## Troubleshooting

*   **`FileNotFoundError: No images or videos found in .../test` during Evaluation:** This script specifically uses `glob` to find files in subdirectories and passes the *list* to `model.predict()`. If you still see this, double-check the `TEST_DIR` path and ensure `glob` finds files correctly (check the console output right before evaluation starts).
*   **PyTorch/Torchvision Version Errors (e.g., `operator torchvision::nms does not exist`):** Make sure you installed PyTorch/Torchvision using the command from the official website for your system to ensure compatibility. Uninstall existing versions (`pip uninstall torch torchvision torchaudio -y`) before reinstalling if needed.
*   **CUDA/GPU Issues:** Ensure your NVIDIA drivers, CUDA toolkit, and the CUDA version specified during PyTorch installation are compatible. If issues persist, try running on CPU by setting `device='cpu'`.
*   **Memory Errors (`OutOfMemoryError`):** Reduce the `BATCH_SIZE`.

## Future Work / Improvements

*   Experiment with different YOLOv8 classification model sizes (s, m, l, x) for potentially higher accuracy at the cost of speed/resources.
*   Tune hyperparameters (learning rate, optimizer, epochs, patience).
*   Implement data augmentation during training.
*   Use the more standard Ultralytics evaluation method by creating a `data.yaml` file defining train/val/test paths and using `model.val(data='data.yaml', split='test')`.