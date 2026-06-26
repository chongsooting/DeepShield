import numpy as np
import cv2
import tensorflow as tf

IMG_SIZE = (224, 224)

def preprocess_for_model(img_rgb):
    """
    Resize and apply EfficientNet-specific preprocessing.
    MUST match what Member 2 used during training.
    """
    img_resized = cv2.resize(img_rgb, IMG_SIZE)
    img_float = img_resized.astype(np.float32)
    # EfficientNet preprocessing scales to [-1, 1]
    img_preprocessed = tf.keras.applications.efficientnet.preprocess_input(img_float)
    return np.expand_dims(img_preprocessed, axis=0)  # shape: (1, 224, 224, 3)

def get_last_conv_layer(model):
    """Auto-detect the last convolutional layer for Grad-CAM."""
    for layer in reversed(model.layers):
        if hasattr(layer, 'output') and len(layer.output.shape) == 4:
            return layer.name
    raise ValueError("No convolutional layer found in model.")