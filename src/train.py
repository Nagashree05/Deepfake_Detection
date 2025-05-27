import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models, callbacks
from sklearn.utils import class_weight
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

# Environment configuration
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Suppress INFO/WARNING logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TensorFlow logs

tf.get_logger().setLevel('ERROR')

# ========== 1. Input Validation ==========
def validate_inputs(train_df, val_df, test_df):
    for df in [train_df, val_df, test_df]:
        df['filepath'] = df['filepath'].str.replace('\\', '/')

    # Check dataframe validity
    for df, name in zip([train_df, val_df, test_df], ['Train', 'Validation', 'Test']):
        if df.empty:
            raise ValueError(f"{name} DataFrame is empty!")
        if 'filepath' not in df.columns or 'label' not in df.columns:
            raise ValueError(f"{name} DataFrame missing required columns!")
        
    def filter_missing(df, name):
        initial_count = len(df)
        df = df[df['filepath'].apply(os.path.exists)].copy()
        removed = initial_count - len(df)
        if removed > 0:
            print(f"Removed {removed} missing files from {name} set")
        return df
                    
# Load splits with validation
train_df = pd.read_csv('data/splits/train.csv')
val_df = pd.read_csv('data/splits/val.csv')
test_df = pd.read_csv('data/splits/test.csv')
print("Loaded dataframes")
validate_inputs(train_df, val_df, test_df)
print("Validation complete")

# Convert labels to strings for Keras compatibility
train_df['label'] = train_df['label'].astype(str)
val_df['label'] = val_df['label'].astype(str)
test_df['label'] = test_df['label'].astype(str)

# ========== 2. Enhanced Data Pipeline ==========
train_datagen = ImageDataGenerator(
    rotation_range=15,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.9, 1.1],
    shear_range=0.1,
    zoom_range=0.1,
    fill_mode='nearest',  # Fixed missing comma here
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input
)

val_test_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input
)

# Create generators with balanced class detection
train_gen = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filepath',
    y_col='label',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=True,
    seed=42
)

val_gen = val_test_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='filepath',
    y_col='label',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

test_gen = val_test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='filepath',
    y_col='label',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# ========== 3. Class Handling & Weighting ==========
# Verify class distribution
print("\nClass distribution:")
print("Train:", train_df['label'].value_counts())
print("Validation:", val_df['label'].value_counts())
print("Test:", test_df['label'].value_counts())

# Compute class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_df['label']),
    y=train_df['label']
)
class_weights = dict(enumerate(class_weights))
print(f"\nClass weights: {class_weights}")

# ========== 4. Enhanced Model Architecture ==========
def build_model():
    # Verify GPU availability
    print("\nGPU Available:", len(tf.config.list_physical_devices('GPU')) > 0)
    
    # Mixed precision policy (requires GPU with compute capability 7.0+)
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Strategic layer unfreezing
    for layer in base_model.layers[:100]:
        layer.trainable = False
    for layer in base_model.layers[100:]:
        layer.trainable = True  # Enable fine-tuning of later layers
        
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid', dtype='float32')  # Output in float32
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    return model

model = build_model()
model.summary()

# ========== 5. Enhanced Training Configuration ==========
# Learning rate scheduler
def lr_scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

cb_list = [
    callbacks.EarlyStopping(
        patience=5,
        restore_best_weights=True,
        monitor='val_auc',
        mode='max'
    ),
    callbacks.ModelCheckpoint(
        'best_model.h5',
        save_best_only=True,
        monitor='val_auc',
        mode='max'
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_auc',
        factor=0.2,
        patience=2,
        mode='max'
    ),
    callbacks.LearningRateScheduler(lr_scheduler)
]

# ========== 6. Train with Steps Per Epoch ==========
# Calculate steps per epoch
train_steps = len(train_gen)
val_steps = len(val_gen)

history = model.fit(
    train_gen,
    epochs=30,
    steps_per_epoch=train_steps,
    validation_data=val_gen,
    validation_steps=val_steps,
    class_weight=class_weights,
    callbacks=cb_list,
    verbose=1
    )

# ========== 7. Comprehensive Evaluation ==========
print("\nEvaluating on test set:")
test_loss, test_acc, test_auc, test_precision, test_recall = model.evaluate(test_gen)
print(f"""
Test Metrics:
- Loss:     {test_loss:.4f}
- Accuracy: {test_acc:.4f}
- AUC:      {test_auc:.4f}
- Precision: {test_precision:.4f}
- Recall:    {test_recall:.4f}
""")

# ========== 8. Save & Visualize ==========
# Save final model
model.save('final_resnet50_deepfake.h5')

# Enhanced visualization
plt.figure(figsize=(18, 6))

# Accuracy plot
plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.legend()

# Loss plot
plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss')
plt.xlabel('Epoch')
plt.legend()

# AUC plot
plt.subplot(1, 3, 3)
plt.plot(history.history['auc'], label='Train')
plt.plot(history.history['val_auc'], label='Validation')
plt.title('AUC')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.savefig('training_metrics.png', dpi=300)
plt.show()

