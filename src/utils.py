import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_training_history(history, save_path=None):
    """Plot and optionally save training/validation accuracy and loss."""
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.legend()
    plt.title('Accuracy')
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def load_and_preprocess_image(img_path, target_size=(224,224)):
    """Load and preprocess a single image for ResNet50 prediction."""
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.resnet50.preprocess_input(x)
    return x

def load_trained_model(model_path):
    """Load a trained Keras model from disk."""
    return load_model(model_path)

def print_classification_report(y_true, y_pred):
    """Prints a classification report and confusion matrix."""
    from sklearn.metrics import classification_report, confusion_matrix
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
