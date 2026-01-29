"""
Simplified Face Mask Detection Training Script
Optimized for FPGA deployment with 64x64 grayscale images

Features:
- Smaller 64x64 input size (vs 224x224) for FPGA compatibility
- Grayscale images (1 channel vs 3) for reduced memory
- Simplified CNN architecture suitable for hardware acceleration
- Data augmentation for better generalization
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

# ============== CONFIGURATION ==============
IMG_SIZE = 64       # Smaller for FPGA (original uses 224)
BATCH_SIZE = 32
EPOCHS = 25
INIT_LR = 1e-4
DATASET_PATH = "dataset"
MODEL_PATH = "models"

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Face Mask Detection Model')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--img_size', type=int, default=IMG_SIZE, help='Image size (square)')
    parser.add_argument('--lr', type=float, default=INIT_LR, help='Initial learning rate')
    parser.add_argument('--dataset', type=str, default=DATASET_PATH, help='Dataset path')
    return parser.parse_args()


def load_dataset(dataset_path, img_size):
    """Load and preprocess dataset"""
    print("[INFO] Loading images...")
    data = []
    labels = []
    
    categories = ['with_mask', 'without_mask']
    
    for category in categories:
        path = os.path.join(dataset_path, category)
        if not os.path.exists(path):
            print(f"[WARN] Path not found: {path}")
            continue
            
        label = "mask" if category == "with_mask" else "no_mask"
        
        for img_name in os.listdir(path):
            # Skip non-image files
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                continue
                
            try:
                img_path = os.path.join(path, img_name)
                
                # Load as grayscale and resize
                img = keras.preprocessing.image.load_img(
                    img_path, 
                    target_size=(img_size, img_size), 
                    color_mode='grayscale'
                )
                img = keras.preprocessing.image.img_to_array(img)
                
                data.append(img)
                labels.append(label)
                
            except Exception as e:
                print(f"[WARN] Could not load {img_name}: {e}")
    
    print(f"[INFO] Loaded {len(data)} images")
    print(f"[INFO] Mask images: {labels.count('mask')}, No-mask images: {labels.count('no_mask')}")
    return np.array(data, dtype="float32"), np.array(labels)


def build_model(img_size):
    """
    Build simplified CNN model optimized for FPGA
    
    Architecture:
    - 3 Conv2D layers with MaxPooling
    - BatchNormalization for stable training
    - Dense layers with Dropout for regularization
    - Softmax output for binary classification
    """
    model = Sequential([
        # First Conv Block
        Conv2D(32, (3, 3), activation='relu', padding='same', 
               input_shape=(img_size, img_size, 1)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Second Conv Block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Third Conv Block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Fourth Conv Block (deeper features)
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Dense Layers
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(2, activation='softmax')
    ])
    
    return model


def get_callbacks(model_path):
    """Create training callbacks"""
    callbacks = [
        # Save best model
        ModelCheckpoint(
            filepath=os.path.join(model_path, 'mask_detector_64x64_best.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    return callbacks


def plot_training_history(history, save_path):
    """Plot and save training metrics"""
    plt.style.use("ggplot")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot Loss
    axes[0].plot(history.history["loss"], label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    
    # Plot Accuracy
    axes[1].plot(history.history["accuracy"], label="Train Accuracy")
    axes[1].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Training plot saved to {save_path}")


def main():
    args = parse_args()
    
    # Create model directory
    os.makedirs(args.dataset, exist_ok=True)
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    # Load dataset
    data, labels = load_dataset(args.dataset, args.img_size)
    
    if len(data) == 0:
        print("[ERROR] No images found! Please add images to dataset/with_mask and dataset/without_mask")
        print("[INFO] You can download the dataset from:")
        print("       https://www.kaggle.com/datasets/omkargurav/face-mask-dataset")
        return
    
    # Normalize pixel values
    data = data / 255.0
    
    # Encode labels
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = tf.keras.utils.to_categorical(labels)
    
    print(f"[INFO] Classes: {lb.classes_}")
    print(f"[INFO] Data shape: {data.shape}")
    print(f"[INFO] Labels shape: {labels.shape}")
    
    # Split dataset
    trainX, testX, trainY, testY = train_test_split(
        data, labels, 
        test_size=0.20, 
        stratify=labels, 
        random_state=42
    )
    
    print(f"[INFO] Training samples: {len(trainX)}")
    print(f"[INFO] Testing samples: {len(testX)}")
    
    # Data augmentation
    aug = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )
    
    # Build model
    print("[INFO] Building model...")
    model = build_model(args.img_size)
    model.summary()
    
    # Compile
    print("[INFO] Compiling model...")
    opt = Adam(learning_rate=args.lr)
    model.compile(
        loss="binary_crossentropy", 
        optimizer=opt, 
        metrics=["accuracy"]
    )
    
    # Get callbacks
    callbacks = get_callbacks(MODEL_PATH)
    
    # Train
    print("[INFO] Training model...")
    history = model.fit(
        aug.flow(trainX, trainY, batch_size=args.batch_size),
        steps_per_epoch=len(trainX) // args.batch_size,
        validation_data=(testX, testY),
        validation_steps=len(testX) // args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks
    )
    
    # Evaluate
    print("\n[INFO] Evaluating model...")
    predictions = model.predict(testX, batch_size=args.batch_size)
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(testY, axis=1)
    
    print("\nClassification Report:")
    print(classification_report(true_classes, pred_classes, target_names=lb.classes_))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(true_classes, pred_classes))
    
    # Save final model
    final_model_path = os.path.join(MODEL_PATH, "mask_detector_64x64.h5")
    model.save(final_model_path, save_format="h5")
    print(f"\n[INFO] Model saved to {final_model_path}")
    
    # Save model summary
    with open(os.path.join(MODEL_PATH, "model_summary.txt"), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    # Plot training history
    plot_training_history(history, os.path.join(MODEL_PATH, "training_plot.png"))
    
    # Save TFLite version (for edge deployment)
    print("[INFO] Converting to TFLite...")
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        tflite_path = os.path.join(MODEL_PATH, "mask_detector_64x64.tflite")
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"[INFO] TFLite model saved to {tflite_path}")
    except Exception as e:
        print(f"[WARN] TFLite conversion failed: {e}")
    
    print("\n[SUCCESS] Training complete!")
    print(f"[INFO] Model saved to: {final_model_path}")
    print(f"[INFO] Use 'python modified/detect_cpu.py' to test detection")


if __name__ == "__main__":
    main()
