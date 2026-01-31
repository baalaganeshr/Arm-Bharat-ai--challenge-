"""
Improved Face Mask Detection Training
With data augmentation, better architecture, and proper validation
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

print("=" * 70)
print("  FACE MASK DETECTION - IMPROVED TRAINING WITH GPU")
print("=" * 70)

# Check GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"[SUCCESS] GPU Available: {gpus[0].name}")
else:
    print("[WARNING] No GPU found, using CPU")

# Paths
DATASET_PATH = '/app/dataset'
WITH_MASK_PATH = os.path.join(DATASET_PATH, 'with_mask')
WITHOUT_MASK_PATH = os.path.join(DATASET_PATH, 'without_mask')
MODEL_DIR = '/app/models'
os.makedirs(MODEL_DIR, exist_ok=True)

# Parameters
IMG_SIZE = 128  # Increased from 64 for better accuracy
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.0001

print(f"\n[INFO] Configuration:")
print(f"  Image Size: {IMG_SIZE}x{IMG_SIZE}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Learning Rate: {LEARNING_RATE}")

# Load and prepare data
print(f"\n[INFO] Loading dataset from {DATASET_PATH}...")

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # 80% train, 20% validation
)

# No augmentation for validation
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Training generator
train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

# Validation generator
val_generator = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

print(f"\n[INFO] Dataset loaded:")
print(f"  Training samples: {train_generator.samples}")
print(f"  Validation samples: {val_generator.samples}")
print(f"  Classes: {train_generator.class_indices}")

# Build improved model
print("\n[INFO] Building improved CNN model...")

model = keras.Sequential([
    # Input layer
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
    
    # Conv Block 1
    layers.Conv2D(32, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Conv Block 2
    layers.Conv2D(64, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Conv Block 3
    layers.Conv2D(128, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Conv Block 4
    layers.Conv2D(256, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Flatten and Dense
    layers.Flatten(),
    layers.Dense(512),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.5),
    
    layers.Dense(256),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.5),
    
    # Output
    layers.Dense(1, activation='sigmoid')
])

model.summary()

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

# Callbacks
callbacks = [
    keras.callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR, 'best_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

# Train
print("\n[INFO] Starting training...")
print("=" * 70)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# Save final model
final_model_path = os.path.join(MODEL_DIR, f'mask_detector_{IMG_SIZE}x{IMG_SIZE}_final.h5')
model.save(final_model_path)
print(f"\n[SUCCESS] Final model saved to {final_model_path}")

# Evaluate
print("\n[INFO] Evaluating model...")
results = model.evaluate(val_generator, verbose=0)
print("\n" + "=" * 70)
print("FINAL RESULTS:")
print("=" * 70)
print(f"Validation Loss:      {results[0]:.4f}")
print(f"Validation Accuracy:  {results[1]:.4f} ({results[1]*100:.2f}%)")
print(f"Validation Precision: {results[2]:.4f}")
print(f"Validation Recall:    {results[3]:.4f}")
print("=" * 70)

# Save training history
history_path = os.path.join(MODEL_DIR, 'training_history.npy')
np.save(history_path, history.history)
print(f"\n[INFO] Training history saved to {history_path}")

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plot_path = os.path.join(MODEL_DIR, 'training_plots.png')
plt.savefig(plot_path)
print(f"[INFO] Training plots saved to {plot_path}")

print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)
