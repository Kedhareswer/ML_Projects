# ai_art_style_transfer/src/utils.py
# This file will contain utility functions for image loading,
# preprocessing (resizing, normalization), and postprocessing (denormalization, saving).

import numpy as np
import cv2 # OpenCV for image manipulation

def load_image(image_path: str, max_dim: int = 512) -> np.ndarray:
    """
    Loads an image from the given file path, resizes it so its largest dimension
    is max_dim (maintaining aspect ratio), and adds a batch dimension.

    Args:
        image_path (str): Path to the image file.
        max_dim (int): The maximum size of the image's longer dimension.

    Returns:
        np.ndarray: The loaded and processed image as a float32 NumPy array
                    in RGB format, with shape (1, height, width, 3).
                    Returns None if the image cannot be loaded.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Image not found or corrupted at {image_path}")
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert from BGR (OpenCV default) to RGB
        img = img.astype(np.float32)

        # Resize while maintaining aspect ratio
        shape = np.array(img.shape[:-1], dtype=np.float32) # height, width
        long_dim = max(shape)
        scale = max_dim / long_dim

        new_shape = tuple((shape * scale).astype(int))

        img = cv2.resize(img, (new_shape[1], new_shape[0])) # cv2.resize expects (width, height)

        img = img / 255.0 # Normalize to [0, 1] range
        img = np.expand_dims(img, axis=0) # Add batch dimension

        return img
    except Exception as e:
        print(f"An error occurred while loading image {image_path}: {e}")
        return None

def preprocess_image_vgg(image_tensor: np.ndarray) -> np.ndarray:
    """
    Preprocesses an image tensor for VGG-like models.
    This typically involves subtracting mean pixel values and converting RGB to BGR.

    Args:
        image_tensor (np.ndarray): An image tensor, typically the output of load_image
                                   (shape: 1, height, width, 3), pixels in [0,1] range.

    Returns:
        np.ndarray: The preprocessed image tensor.
    """
    if image_tensor is None:
        return None

    # VGG networks expect BGR images and mean pixel subtraction
    # Mean pixel values (RGB) for VGG19 trained on ImageNet
    vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32) # RGB

    processed_img = image_tensor * 255.0 # Scale back to [0, 255] if it was [0,1]

    # Subtract VGG mean (RGB order)
    processed_img_rgb = processed_img.copy()
    processed_img_rgb[..., 0] -= vgg_mean[0]
    processed_img_rgb[..., 1] -= vgg_mean[1]
    processed_img_rgb[..., 2] -= vgg_mean[2]

    # Convert RGB to BGR
    processed_img_bgr = processed_img_rgb[..., ::-1]

    return processed_img_bgr

def deprocess_image_vgg(processed_tensor: np.ndarray) -> np.ndarray:
    """
    Deprocesses an image tensor that was preprocessed for VGG models.
    Reverses mean subtraction, BGR to RGB conversion, and clips to [0, 255].

    Args:
        processed_tensor (np.ndarray): The processed image tensor from the model.
                                       Shape (1, height, width, 3), BGR order.

    Returns:
        np.ndarray: The deprocessed image as a uint8 NumPy array,
                    RGB format, values in [0, 255].
    """
    if processed_tensor is None:
        return None

    img = processed_tensor.copy()
    img = np.squeeze(img, axis=0) # Remove batch dimension

    # Add VGG mean (BGR order for the array, but mean values are for RGB)
    # VGG mean pixel values (RGB)
    vgg_mean_rgb = np.array([123.68, 116.779, 103.939], dtype=np.float32)

    # The tensor is BGR, so add mean in BGR order
    img[..., 0] += vgg_mean_rgb[2] # Blue
    img[..., 1] += vgg_mean_rgb[1] # Green
    img[..., 2] += vgg_mean_rgb[0] # Red

    # Clip to [0, 255] and convert to uint8
    img = np.clip(img, 0, 255).astype('uint8')

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img_rgb


if __name__ == '__main__':
    print(f"Utilities for Neural Style Transfer in {__file__}")
    # Example Usage (requires an image file named 'test_image.jpg' in the script's directory)
    # Create a dummy image for testing if one doesn't exist
    dummy_image_path = "dummy_test_image.jpg"
    if not cv2.imread(dummy_image_path):
        dummy_array = np.random.randint(0, 256, (400, 600, 3), dtype=np.uint8)
        cv2.imwrite(dummy_image_path, dummy_array)
        print(f"Created dummy image at {dummy_image_path} for testing.")

    print(f"Attempting to load image: {dummy_image_path}")
    loaded_img_tensor = load_image(dummy_image_path, max_dim=256)
    if loaded_img_tensor is not None:
        print(f"Loaded image shape: {loaded_img_tensor.shape}")

        preprocessed_tensor = preprocess_image_vgg(loaded_img_tensor)
        if preprocessed_tensor is not None:
            print(f"Preprocessed image tensor shape: {preprocessed_tensor.shape}")
            print(f"Preprocessed image min/max: {preprocessed_tensor.min()}, {preprocessed_tensor.max()}")

            deprocessed_img = deprocess_image_vgg(preprocessed_tensor)
            if deprocessed_img is not None:
                print(f"Deprocessed image shape: {deprocessed_img.shape}")
                print(f"Deprocessed image min/max/dtype: {deprocessed_img.min()}, {deprocessed_img.max()}, {deprocessed_img.dtype}")
                # cv2.imwrite("deprocessed_test_output.jpg", cv2.cvtColor(deprocessed_img, cv2.COLOR_RGB2BGR))
                # print("Saved deprocessed_test_output.jpg")
    else:
        print(f"Failed to load {dummy_image_path}")

    # Clean up dummy image
    # import os
    # if os.path.exists(dummy_image_path):
    #     os.remove(dummy_image_path)
    #     print(f"Cleaned up {dummy_image_path}")
