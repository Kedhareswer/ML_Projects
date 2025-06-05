# ai_art_style_transfer/src/transfer_style.py
# This is the main script to orchestrate the neural style transfer process.
# It will handle argument parsing, loading images, building the model,
# running the optimization loop, and saving the output.

import argparse
import time
import os
import tensorflow as tf
import numpy as np

# Ensure utils and nst_model can be imported
# This setup assumes the script might be run directly, or the 'src' dir is in PYTHONPATH
try:
    import utils
    import nst_model
except ImportError:
    # Attempt to add parent directory to sys.path if running from src
    # to find sibling modules, though direct execution from root is preferred.
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        import utils
        import nst_model
    except ImportError:
        print("Error: Ensure 'utils.py' and 'nst_model.py' are accessible.")
        print("If running from project root: python -m ai_art_style_transfer.src.transfer_style ...")
        print("If running from 'src/': python transfer_style.py ... (make sure parent of 'src' is in PYTHONPATH or use relative imports if 'src' is a package)")
        exit(1)


def main(args):
    """
    Main function to run the Neural Style Transfer.
    """
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Eager execution: {tf.executing_eagerly()}")

    # Load content and style images
    print(f"Loading content image from: {args.content_image_path}")
    content_image_np = utils.load_image(args.content_image_path, max_dim=args.max_dim)
    print(f"Loading style image from: {args.style_image_path}")
    style_image_np = utils.load_image(args.style_image_path, max_dim=args.max_dim)

    if content_image_np is None or style_image_np is None:
        print("Error loading images. Please check paths and image integrity.")
        return

    # Preprocess images for VGG19
    preprocessed_content_image = utils.preprocess_image_vgg(content_image_np)
    preprocessed_style_image = utils.preprocess_image_vgg(style_image_np)

    if preprocessed_content_image is None or preprocessed_style_image is None:
        print("Error preprocessing images.")
        return

    print(f"Content image shape after preprocessing: {preprocessed_content_image.shape}")
    print(f"Style image shape after preprocessing: {preprocessed_style_image.shape}")

    # Create the StyleContentModel
    extractor = nst_model.StyleContentModel(nst_model.STYLE_LAYERS, nst_model.CONTENT_LAYERS)

    # Get target content and style representations
    print("Extracting target content and style features...")
    target_content_features = extractor(preprocessed_content_image)['content']
    target_style_features = extractor(preprocessed_style_image)['style']
    print("Target features extracted.")

    # Initialize the generated image (starting from the content image)
    generated_image_var = tf.Variable(preprocessed_content_image, dtype=tf.float32)

    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, beta_1=0.99, epsilon=1e-1)

    # Loss weights
    loss_weights_dict = {
        'content': args.content_weight,
        'style': args.style_weight,
        'total_variation': args.tv_weight
    }

    # Function to compute total loss (can be moved to nst_model.py if preferred)
    def compute_loss(model, current_loss_weights, init_image, current_target_content_features, current_target_style_features):
        model_outputs = model(init_image)

        current_content_features = model_outputs['content']
        c_loss = nst_model.content_loss(current_content_features, current_target_content_features)

        current_style_features = model_outputs['style']
        s_loss = nst_model.style_loss(current_style_features, current_target_style_features)

        tv_l = nst_model.total_variation_loss(init_image)

        total_l = (current_loss_weights['content'] * c_loss +
                   current_loss_weights['style'] * s_loss +
                   current_loss_weights['total_variation'] * tv_l)
        return total_l, c_loss, s_loss, tv_l

    # Define the train_step with tf.function for performance
    # Ensure the input signature matches the expected shape of generated_image_var
    @tf.function(input_signature=[
        tf.TensorSpec(shape=preprocessed_content_image.shape, dtype=tf.float32),
    ])
    def train_step_fn(image_variable):
        with tf.GradientTape() as tape:
            total_l, c_l, s_l, tv_l = compute_loss(
                extractor, loss_weights_dict, image_variable,
                target_content_features, target_style_features
            )
        gradients = tape.gradient(total_l, image_variable)
        optimizer.apply_gradients([(gradients, image_variable)])
        return total_l, c_l, s_l, tv_l

    # Optimization loop
    print(f"Starting optimization with {args.epochs} epochs and {args.steps_per_epoch} steps per epoch.")
    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        for step in range(args.steps_per_epoch):
            total_loss_val, c_loss_val, s_loss_val, tv_loss_val = train_step_fn(generated_image_var)
            if (step + 1) % args.log_interval == 0:
                print(f"  Epoch {epoch + 1}/{args.epochs}, Step {step + 1}/{args.steps_per_epoch} - "
                      f"Total Loss: {total_loss_val:.2f} "
                      f"(Content: {c_loss_val * args.content_weight:.2f}, "
                      f"Style: {s_loss_val * args.style_weight:.2f}, "
                      f"TV: {tv_loss_val * args.tv_weight:.2f})")

        print(f"Epoch {epoch + 1} completed in {time.time() - epoch_start_time:.2f}s. Last Total Loss: {total_loss_val:.2f}")

    total_time = time.time() - start_time
    print(f"Total optimization time: {total_time:.2f} seconds.")

    # Deprocess and save the final image
    print("Deprocessing final image...")
    final_image_deprocessed_np = utils.deprocess_image_vgg(generated_image_var.numpy())

    output_dir = os.path.dirname(args.output_image_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    final_image_bgr = utils.cv2.cvtColor(final_image_deprocessed_np, utils.cv2.COLOR_RGB2BGR)
    utils.cv2.imwrite(args.output_image_path, final_image_bgr)
    print(f"Stylized image saved to: {args.output_image_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Neural Style Transfer Command-Line Tool")
    parser.add_argument("--content_image_path", type=str, required=True, help="Path to the content image.")
    parser.add_argument("--style_image_path", type=str, required=True, help="Path to the style image.")
    parser.add_argument("--output_image_path", type=str, required=True, help="Path to save the generated stylized image.")

    parser.add_argument("--max_dim", type=int, default=512, help="Maximum dimension for loading images.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for optimization.")
    parser.add_argument("--steps_per_epoch", type=int, default=100, help="Number of steps per epoch.")
    parser.add_argument("--learning_rate", type=float, default=0.02, help="Learning rate for the Adam optimizer.")

    parser.add_argument("--content_weight", type=float, default=1e3, help="Weight for the content loss.")
    parser.add_argument("--style_weight", type=float, default=1e-2, help="Weight for the style loss.")
    parser.add_argument("--tv_weight", type=float, default=30, help="Weight for the total variation loss.")
    parser.add_argument("--log_interval", type=int, default=20, help="Log loss every N steps.")

    cli_args = parser.parse_args()

    # Check if image paths exist before trying to create dummies
    paths_to_check = {}
    if not os.path.exists(cli_args.content_image_path):
        paths_to_check[cli_args.content_image_path] = "content_image_path"
    if not os.path.exists(cli_args.style_image_path):
        paths_to_check[cli_args.style_image_path] = "style_image_path"

    for img_path, arg_name in paths_to_check.items():
        default_data_path = os.path.join("ai_art_style_transfer", "data", os.path.basename(img_path))
        if os.path.exists(default_data_path):
            print(f"Warning: Image at {img_path} (from arg {arg_name}) not found. Using default {default_data_path} instead.")
            if arg_name == "content_image_path":
                cli_args.content_image_path = default_data_path
            elif arg_name == "style_image_path":
                cli_args.style_image_path = default_data_path
        else:
            print(f"Warning: Image at {img_path} (from arg {arg_name}) not found, and default {default_data_path} not found. Creating a dummy image for testing at {img_path}.")
            dummy_img_dir = os.path.dirname(img_path)
            if dummy_img_dir and not os.path.exists(dummy_img_dir):
                os.makedirs(dummy_img_dir)

            if hasattr(utils, 'cv2'):
                dummy_array = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
                utils.cv2.imwrite(img_path, dummy_array)
                print(f"Created dummy image at {img_path}")
            else:
                print(f"Error: cv2 module not available in utils. Cannot create dummy image for {img_path}.")
                exit(1)
    main(cli_args)
