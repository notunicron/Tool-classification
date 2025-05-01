import torch
import os
import random
import glob
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path # Added for getting parent directory name
from tqdm import tqdm # Added for progress bar

# --- Configuration ---
# 1. Set the base path to your dataset
DATASET_BASE_DIR = 'cholec-tinytools'

# 2. Define paths for train, validation, and test sets
TRAIN_DIR = os.path.join(DATASET_BASE_DIR, 'train')
VAL_DIR = os.path.join(DATASET_BASE_DIR, 'validation')
TEST_DIR = os.path.join(DATASET_BASE_DIR, 'test')

# 3. Choose YOLO model variant
MODEL_NAME = 'yolov8n-cls.pt'

# 4. Training Hyperparameters
EPOCHS = 50          # Number of training epochs
IMG_SIZE = 224       # Input image size
BATCH_SIZE = 16
PROJECT_NAME = 'cholec_tool_classification' # Folder name for saving results
RUN_NAME = f'yolov8n_cls_epochs{EPOCHS}'    # Specific name for this training run

# --- Check for GPU ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# --- 1. Load Pre-trained YOLOv8 Classification Model ---
print(f"\nLoading pre-trained model: {MODEL_NAME}")
model = YOLO(MODEL_NAME)

# --- 2. Train the Model ---
print("\nStarting training...")
results = model.train(
    data=DATASET_BASE_DIR, # Directory containing train/val folders
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    project=PROJECT_NAME,
    name=RUN_NAME,
    device=device,
)

print("Training finished.")
print(f"Best model weights saved at: {results.save_dir}/weights/best.pt")

# --- 3. Evaluate the Model on the Test Set ---
print("\nEvaluating model on the test set...")

# Load the *best* weights from your training run
best_model_path = os.path.join(PROJECT_NAME, RUN_NAME, 'weights', 'best.pt')
if not os.path.exists(best_model_path):
     print(f"Error: Best model weights not found at {best_model_path}")
     last_model_path = os.path.join(PROJECT_NAME, RUN_NAME, 'weights', 'last.pt')
     if os.path.exists(last_model_path):
         print("Warning: Using last weights instead of best weights for evaluation.")
         best_model_path = last_model_path
     else:
          print("Error: Neither best nor last model weights found. Cannot evaluate.")
          model_to_evaluate = None
else:
     print(f"Loading best weights from: {best_model_path}")
     model_to_evaluate = YOLO(best_model_path)

# Proceed with evaluation only if a model was loaded
if model_to_evaluate:
    # --- Get List of All Image Files using glob ---
    print(f"Globbing for image files in subdirectories of: {TEST_DIR}")
    # Ensure we look inside the class subdirectories
    all_image_files = glob.glob(os.path.join(TEST_DIR, '*', '*.png')) + \
                      glob.glob(os.path.join(TEST_DIR, '*', '*.jpg'))

    num_files = len(all_image_files)

    if num_files == 0:
        # This check is crucial - if glob fails, predict will also fail.
        print(f"Error: glob could not find any image files in the subdirectories of {TEST_DIR}. Check paths and file extensions.")
        print(f"Searched pattern: {os.path.join(TEST_DIR, '*', '*.png')} and similar for jpg.")
    else:
        print(f"Found {num_files} image files via glob. Starting evaluation by predicting on the explicit file list...")
        try:
            # --- Perform Prediction directly on the List of Files ---
            results_generator = model_to_evaluate.predict(
                source=all_image_files,
                imgsz=IMG_SIZE,
                batch=BATCH_SIZE,
                stream=True,            # Stream is efficient for many files
                device=device,
                verbose=False           # Keep output clean during loop
            )

            # --- Manually Calculate Accuracy ---
            correct_predictions = 0
            total_images = 0
            class_names = model_to_evaluate.names

            # Iterate through prediction results with a progress bar
            for result in tqdm(results_generator, total=num_files, desc="Evaluating Test Set"):
                if result.probs is None:
                    print(f"Warning: Could not get prediction probability for image: {result.path}")
                    total_images += 1
                    continue

                predicted_class_index = result.probs.top1
                predicted_class_name = class_names[predicted_class_index]

                try:
                    true_class_name = Path(result.path).parent.name
                    if true_class_name not in class_names.values():
                         print(f"Warning: Unexpected parent directory name '{true_class_name}' for {result.path}. Skipping accuracy check.")
                         total_images += 1
                         continue
                except IndexError:
                     print(f"Warning: Could not determine true class for image: {result.path}. Skipping accuracy check.")
                     total_images += 1
                     continue

                if predicted_class_name == true_class_name:
                    correct_predictions += 1

                total_images += 1

            # --- Calculate and Print Final Accuracy ---
            if total_images > 0:
                if total_images != num_files:
                    print(f"Warning: Processed {total_images} images, but initially found {num_files} via glob. Some might have had issues during prediction.")

                accuracy = correct_predictions / total_images
                print(f"\n--- Test Set Evaluation Results ---")
                print(f"Total images evaluated: {total_images}")
                print(f"Correct predictions: {correct_predictions}")
                print(f"Accuracy (Top-1): {accuracy:.4f} ({accuracy*100:.2f}%)")
            else:
                 print("No images were successfully processed during evaluation loop, though files were found by glob.")

        except Exception as e:
             print(f"\n--- UNEXPECTED ERROR DURING PREDICTION ON FILE LIST ---")
             print(f"An error occurred: {e}")
             import traceback
             traceback.print_exc()

else:
    print("Skipping evaluation because the model weights could not be loaded.")

# --- 4. Predict on a Single Image (Remains the same) ---
print("\nRunning prediction on a sample image...")

# Use the same model loaded for evaluation if available
if model_to_evaluate:
    try:
        test_image_files = glob.glob(os.path.join(TEST_DIR, '*', '*.png')) + \
                           glob.glob(os.path.join(TEST_DIR, '*', '*.jpg'))
        if not test_image_files:
            print("No image files found in the test directory for single prediction.")
        else:
            random_image_path = random.choice(test_image_files)
            print(f"Predicting on: {random_image_path}")

            # Run prediction
            pred_results = model_to_evaluate.predict(random_image_path, imgsz=IMG_SIZE, verbose=False)

            # Process results
            result = pred_results[0]
            class_names_pred = result.names # Get names from this result object
            predicted_class_index = result.probs.top1
            predicted_class_name = class_names_pred[predicted_class_index]
            confidence_score = result.probs.top1conf.item()

            print(f"Predicted Class: {predicted_class_name}")
            print(f"Confidence: {confidence_score:.4f}")

            # # Display image with prediction (optional)
            # img = Image.open(random_image_path)
            # plt.imshow(img)
            # plt.title(f"Predicted: {predicted_class_name} ({confidence_score:.2f})")
            # plt.axis('off')
            # plt.show()

    except FileNotFoundError:
        print(f"Error: Could not find test images in {TEST_DIR} or its subdirectories.")
    except Exception as e:
        print(f"An error occurred during single image prediction: {e}")
else:
    print("Skipping single image prediction because the model weights could not be loaded.")


print("\nScript finished.")