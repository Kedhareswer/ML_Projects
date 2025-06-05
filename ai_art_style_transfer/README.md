# AI Art Style Transfer App

**Type:** DL (CV + GANs) - More accurately, primarily Deep Learning with Convolutional Neural Networks (CNNs). GANs can be an extension but are not core to basic style transfer.

**Value Summary:** Creative industry, AR filters, photo apps

## Description

This project aims to implement Neural Style Transfer (NST), a technique that takes two images—a **content image** and a **style image**—and blends them together so the output image retains the core content of the content image but is painted in the artistic style of the style image. This is typically achieved by using a pre-trained Convolutional Neural Network (CNN) to extract content and style representations from the images and then optimizing a new image to match these representations.

## Key Technologies

- **Programming Language:** Python
- **Core Libraries:**
    - **TensorFlow/Keras** or **PyTorch:** For building and potentially training the neural network components.
    - **NumPy:** For numerical operations, especially image manipulation.
    - **OpenCV (cv2):** For image loading, preprocessing (resizing, color space conversion), and postprocessing.
    - **Matplotlib/Pillow (PIL):** For image display and saving.
- **Pre-trained Model:** VGG16, VGG19, or similar, for feature extraction.

## Project Outline

1.  **Setup & Configuration:**
    - Install necessary libraries.
    - Set up project structure (already done).

2.  **Image Loading and Preprocessing (`src/utils.py`):**
    - Function to load content and style images from specified paths.
    - Resize images to a consistent dimension for processing.
    - Normalize pixel values (e.g., to [0, 1] range or VGG-specific normalization).
    - Convert images to the format expected by the chosen deep learning framework (e.g., TensorFlow tensor).

3.  **Model Definition (`src/nst_model.py`):**
    - Load a pre-trained CNN (e.g., VGG19) without its classification head.
    - Define layers from which content features will be extracted (typically one or more deeper layers).
    - Define layers from which style features will be extracted (typically a set of layers across different depths).
    - Create a model that takes an input image and outputs the feature maps from these selected content and style layers.

4.  **Loss Functions (`src/nst_model.py` or `src/transfer_style.py`):**
    - **Content Loss:** Measures how different the content representation of the generated image is from the content representation of the content image. Usually L2 distance between feature maps.
    - **Style Loss:** Measures how different the style representation of the generated image is from the style representation of the style image. This involves calculating Gram matrices of the feature maps and finding the L2 distance between them.
    - **Total Variation Loss (Optional):** A regularization term to encourage spatial smoothness in the generated image.

5.  **Optimization Loop (`src/transfer_style.py`):**
    - Initialize the generated image (e.g., from the content image or random noise).
    - Define an optimizer (e.g., Adam, L-BFGS).
    - Iteratively:
        - Pass the generated image through the model to get its content and style features.
        - Calculate content loss, style loss, and total variation loss.
        - Compute the total loss (weighted sum of individual losses).
        - Calculate gradients of the loss with respect to the pixels of the generated image.
        - Apply gradients to update the generated image.
        - Periodically save or display the intermediate results.

6.  **Postprocessing and Output (`src/utils.py`, `src/transfer_style.py`):**
    - Denormalize the generated image.
    - Convert back to a standard image format (e.g., uint8 NumPy array).
    - Save the final stylized image.
    - Display the content, style, and generated images.

7.  **Main Script (`src/transfer_style.py`):**
    - Argument parsing for content image path, style image path, output path, number of iterations, loss weights, etc.
    - Orchestrate the entire process: load images, build model, run optimization, save output.

## Potential Next Steps & Enhancements

- **User Interface:** Develop a simple GUI (e.g., using Tkinter, PyQt, or a web framework like Flask/Streamlit) to allow users to easily select images and run the style transfer.
- **Optimization:**
    - Experiment with different optimizers and learning rates.
    - Explore techniques like Fast Neural Style Transfer (feed-forward generative networks) for real-time performance after initial model training.
- **Model Variations:**
    - Try different pre-trained models (e.g., Inception, ResNet).
    - Implement adaptive style transfer techniques that offer more control over style elements.
- **Hyperparameter Tuning:** Systematically tune weights for content, style, and total variation losses.
- **Video Style Transfer:** Extend the concept to stylize video frames.

## TODO

- [x] Elaborate on project description.
- [x] List specific technologies.
- [x] Detail each step in the project outline.
- [ ] Add code to `src/` for `nst_model.py`, `utils.py`, and `transfer_style.py`.
- [ ] Add example notebooks to `notebooks/` demonstrating usage.
- [ ] Add utility scripts to `scripts/` if needed (e.g., for downloading pre-trained models).
