import numpy as np
import cv2
import tensorflow as tf


def compute_gradcam(img_preprocessed, model, last_conv_layer_name):
    """
    Compute Grad-CAM heatmap.

    This implementation runs the model layer-by-layer instead of using
    model.output, making it fully compatible with Keras 3 Sequential models.

    Args:
        img_preprocessed : shape (1, 224, 224, 3), already preprocessed
        model             : loaded Keras model
        last_conv_layer_name : name of the last conv layer (from model.summary())

    Returns:
        heatmap : numpy array (h, w), values in [0, 1]
    """
    img_tensor = tf.cast(img_preprocessed, tf.float32)

    # Find the index of the target conv layer
    layer_names = [layer.name for layer in model.layers]
    if last_conv_layer_name not in layer_names:
        raise ValueError(
            f"Layer '{last_conv_layer_name}' not found. "
            f"Available layers: {layer_names}"
        )
    target_idx = layer_names.index(last_conv_layer_name)

    # Run layer-by-layer, watching the conv output for gradients
    with tf.GradientTape() as tape:
        x = img_tensor
        conv_outputs = None

        for i, layer in enumerate(model.layers):
            x = layer(x, training=False)
            if i == target_idx:
                conv_outputs = x
                tape.watch(conv_outputs)  # track gradients from here

        predictions = x  # final model output after all layers

        # Handle sigmoid output (shape [1,1]) and softmax (shape [1,2])
        if predictions.shape[-1] == 1:
            class_score = predictions[:, 0]
        else:
            pred_idx = tf.argmax(predictions[0])
            class_score = predictions[:, pred_idx]

    if conv_outputs is None:
        raise ValueError(
            f"Failed to capture output from layer '{last_conv_layer_name}'."
        )

    # Gradient of class score w.r.t. conv layer activations
    grads = tape.gradient(class_score, conv_outputs)

    # Global average pool over spatial dimensions → per-channel importance
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight conv outputs by importance and collapse channels
    conv_outputs = conv_outputs[0]                          # (h, w, channels)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]  # (h, w, 1)
    heatmap = tf.squeeze(heatmap)                           # (h, w)

    # ReLU: keep only positive activations
    heatmap = tf.nn.relu(heatmap)

    # Normalise to [0, 1]
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()


def overlay_heatmap(original_img_rgb, heatmap, alpha=0.45):
    """
    Blend Grad-CAM heatmap onto the original image.

    Args:
        original_img_rgb : numpy RGB image (H, W, 3)
        heatmap          : Grad-CAM array (h, w), values 0-1
        alpha            : heatmap opacity

    Returns:
        RGB image with heatmap blended in
    """
    H, W = original_img_rgb.shape[:2]

    # Resize heatmap to match the original image
    heatmap_resized = cv2.resize(heatmap, (W, H))

    # Apply JET colormap (blue = low activation, red = high activation)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colored_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)

    # Weighted blend with original image
    img_uint8 = np.clip(original_img_rgb, 0, 255).astype(np.uint8)
    superimposed = cv2.addWeighted(img_uint8, 1 - alpha, colored_rgb, alpha, 0)
    return superimposed
