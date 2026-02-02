"""
Advanced Face Mask Detection Training with MobileNetV2
Optimized for FPGA deployment with 128x128 input size
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam

print("=" * 70)
print("  FACE MASK DETECTION - ADVANCED TRAINING WITH MOBILENETV2")
print("=" * 70)

# ==========================================
# GPU CONFIGURATION & CHECK
# ==========================================
# List all physical devices (GPUs)
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        # Enable memory growth to prevent allocating all VRAM at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"\n[INFO] ✅ NVIDIA GPU Found: {len(gpus)} device(s)")
        print(f"[INFO] Device Name: {gpus[0].name}\n")
    except RuntimeError as e:
        print(f"[ERROR] GPU Initialization failed: {e}")
else:
    print("\n[WARNING] ❌ No GPU detected. Training will be slow on CPU.\n")
# ==========================================

# Configuration
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
INPUT_SHAPE = (128, 128, 3)

# Dataset paths
DATASET_PATH = 'dataset'
WITH_MASK_PATH = os.path.join(DATASET_PATH, 'with_mask')
WITHOUT_MASK_PATH = os.path.join(DATASET_PATH, 'without_mask')
MODEL_OUTPUT = 'models/mask_detector_mobilenet.h5'

print(f"\n[INFO] Configuration:")
print(f"  Base Model: MobileNetV2 (ImageNet weights, frozen)")
print(f"  Input Shape: {INPUT_SHAPE}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Learning Rate: {LEARNING_RATE}")

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Load dataset
print(f"\n[INFO] Loading dataset from {DATASET_PATH}...")

data = []
labels = []

# Load images with mask (Label = 1)
print(f"[INFO] Loading images from {WITH_MASK_PATH}...")
with_mask_files = os.listdir(WITH_MASK_PATH)
with_mask_count = 0
for img_file in with_mask_files:
    if img_file.startswith('.'):
        continue
    img_path = os.path.join(WITH_MASK_PATH, img_file)
    try:
        image = cv2.imread(img_path)
        if image is None:
            continue
        # Resize to 128x128
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        data.append(image)
        labels.append(1)  # with_mask = 1
        with_mask_count += 1
    except Exception as e:
        print(f"[WARNING] Failed to load {img_path}: {e}")
        continue

print(f"[SUCCESS] Loaded {with_mask_count} images with masks")

# Load images without mask (Label = 0)
print(f"[INFO] Loading images from {WITHOUT_MASK_PATH}...")
without_mask_files = os.listdir(WITHOUT_MASK_PATH)
without_mask_count = 0
for img_file in without_mask_files:
    if img_file.startswith('.'):
        continue
    img_path = os.path.join(WITHOUT_MASK_PATH, img_file)
    try:
        image = cv2.imread(img_path)
        if image is None:
            continue
        # Resize to 128x128
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        data.append(image)
        labels.append(0)  # without_mask = 0
        without_mask_count += 1
    except Exception as e:
        print(f"[WARNING] Failed to load {img_path}: {e}")
        continue

print(f"[SUCCESS] Loaded {without_mask_count} images without masks")

# Convert to NumPy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

print(f"\n[INFO] Dataset Summary:")
print(f"  Total images: {len(data)}")
print(f"  With mask: {with_mask_count}")
print(f"  Without mask: {without_mask_count}")
print(f"  Data shape: {data.shape}")

# Normalize pixel values to [0, 1]
print(f"\n[INFO] Normalizing pixel values...")
data = data / 255.0

# One-hot encode labels
print(f"[INFO] One-hot encoding labels...")
labels = to_categorical(labels, num_classes=2)

# Split dataset: 80% Training, 20% Testing
print(f"[INFO] Splitting dataset: 80% Training, 20% Testing...")
(trainX, testX, trainY, testY) = train_test_split(
    data, labels,
    test_size=0.20,
    stratify=labels,
    random_state=42
)

print(f"[SUCCESS] Dataset split complete:")
print(f"  Training samples: {len(trainX)}")
print(f"  Testing samples: {len(testX)}")

# Build model with MobileNetV2
print(f"\n[INFO] Building model with MobileNetV2 base...")

# Load MobileNetV2 base model
baseModel = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=INPUT_SHAPE
)

# Freeze all layers in base model
print(f"[INFO] Freezing all base model layers...")
for layer in baseModel.layers:
    layer.trainable = False

# Build custom head
print(f"[INFO] Adding custom classification head...")
headModel = baseModel.output
headModel = GlobalAveragePooling2D()(headModel)
headModel = Dense(128, activation='relu')(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation='softmax')(headModel)

# Construct the final model
model = Model(inputs=baseModel.input, outputs=headModel)

# Print model summary
print(f"\n[INFO] Model Architecture:")
model.summary()

# Count trainable and non-trainable parameters
trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
non_trainable_params = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
print(f"\n[INFO] Model Parameters:")
print(f"  Total parameters: {trainable_params + non_trainable_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"  Non-trainable parameters: {non_trainable_params:,}")

# Compile model
print(f"\n[INFO] Compiling model...")
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
print(f"\n[INFO] Starting training...")
print("=" * 70)

history = model.fit(
    trainX, trainY,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(testX, testY),
    verbose=1
)

print("=" * 70)
print(f"[SUCCESS] Training complete!")

# Evaluate on test set
print(f"\n[INFO] Evaluating model on test set...")
test_loss, test_accuracy = model.evaluate(testX, testY, verbose=0)

print(f"\n" + "=" * 70)
print(f"  FINAL RESULTS")
print("=" * 70)
print(f"  Test Loss: {test_loss:.4f}")
print(f"  Test Accuracy: {test_accuracy * 100:.2f}%")
print("=" * 70)

# Save model
print(f"\n[INFO] Saving model to {MODEL_OUTPUT}...")
model.save(MODEL_OUTPUT)
print(f"[SUCCESS] Model saved successfully!")

# Save training history
history_path = 'models/training_history_mobilenet.npy'
np.save(history_path, history.history)
print(f"[SUCCESS] Training history saved to {history_path}")

print(f"\n" + "=" * 70)
print(f"  TRAINING COMPLETE!")
print(f"  Final Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"  Model saved to: {MODEL_OUTPUT}")
print("=" * 70)
