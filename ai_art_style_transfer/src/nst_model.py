# ai_art_style_transfer/src/nst_model.py
# This file contains the neural style transfer model definition,
# including loading pre-trained networks and defining loss functions.

import tensorflow as tf
import numpy as np

# Define the layers to be used for content and style extraction
# Content layer where will pull our feature maps
CONTENT_LAYERS = ['block5_conv2']

# Style layers
STYLE_LAYERS = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1',
    'block4_conv1',
    'block5_conv1'
]
NUM_STYLE_LAYERS = len(STYLE_LAYERS)
NUM_CONTENT_LAYERS = len(CONTENT_LAYERS)


def get_vgg_layers(layer_names):
    """ Creates a VGG19 model that returns a list of intermediate output values."""
    # Load our model. We load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False # We don't want to train VGG weights

    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model


class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = get_vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        """Expects float input in range [0,1]"""
        # Keras VGG19 application preprocesses inputs by default (subtracting ImageNet mean & BGR conversion)
        # if the input is a NumPy array or a tf.Tensor directly.
        # If using a symbolic tensor from tf.keras.Input, preprocessing is part of the model.
        # Our utils.preprocess_image_vgg already does this, so the input here should be preprocessed.
        # However, tf.keras.applications.vgg19.preprocess_input is idempotent if mean is already subtracted
        # and channels are already BGR. For safety, or if utils.py changes, we can call it.
        # inputs = tf.keras.applications.vgg19.preprocess_input(inputs * 255.0) # if inputs are [0,1] RGB
        # For this implementation, we assume inputs are already preprocessed as per utils.preprocess_image_vgg

        outputs = self.vgg(inputs)
        style_outputs = outputs[:self.num_style_layers]
        content_outputs = outputs[self.num_style_layers:]

        # Package the outputs in a dictionary
        style_outputs_dict = {name: value for name, value in zip(self.style_layers, style_outputs)}
        content_outputs_dict = {name: value for name, value in zip(self.content_layers, content_outputs)}

        return {'content': content_outputs_dict, 'style': style_outputs_dict}


def content_loss(content_features, target_content_features):
    """
    Computes the content loss.
    Args:
        content_features: A dict mapping layer names to content features of the generated image.
        target_content_features: A dict mapping layer names to content features of the content image.
    Returns:
        The content loss (a scalar tensor).
    """
    loss = tf.add_n([
        tf.reduce_sum(tf.square(content_features[name] - target_content_features[name])) * 0.5
        for name in content_features.keys()
    ])
    # Normalize by the number of content layers (though often not done, or weights are used)
    # loss /= len(CONTENT_LAYERS)
    return loss


def gram_matrix(input_tensor):
    """Computes the Gram matrix of a tensor (style representation)."""
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)


def style_loss(style_features, target_style_features):
    """
    Computes the style loss.
    Args:
        style_features: A dict mapping layer names to style features (Gram matrices) of the generated image.
        target_style_features: A dict mapping layer names to style features (Gram matrices) of the style image.
    Returns:
        The style loss (a scalar tensor).
    """
    s_loss = tf.add_n([
        tf.reduce_sum(tf.square(gram_matrix(style_features[name]) - gram_matrix(target_style_features[name])))
        for name in style_features.keys()
    ])
    # Normalize by the number of style layers
    s_loss /= NUM_STYLE_LAYERS
    return s_loss


def total_variation_loss(image):
    """
    Computes the total variation loss, encouraging spatial smoothness.
    Args:
        image: The generated image tensor of shape (batch, height, width, channels).
    Returns:
        The total variation loss (a scalar tensor).
    """
    # Kernel for x-derivative (Sobel_x)
    # kernel_x = tf.constant([[[[1, 1, 1], [-2, -2, -2], [1, 1, 1]]]], dtype=tf.float32)
    # Kernel for y-derivative (Sobel_y)
    # kernel_y = tf.constant([[[[1], [-2], [1]], [[1], [-2], [1]], [[1], [-2], [1]]]], dtype=tf.float32)

    # Simpler version using differences
    x_deltas = image[:, :, 1:, :] - image[:, :, :-1, :]
    y_deltas = image[:, 1:, :, :] - image[:, :-1, :, :]

    return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))


if __name__ == '__main__':
    print(f"Core NST model components (TensorFlow-based) in {__file__}")
    # Basic check: try to instantiate the model
    try:
        style_content_model_instance = StyleContentModel(style_layers=STYLE_LAYERS, content_layers=CONTENT_LAYERS)
        # Create a dummy input (batch, height, width, channels) - preprocessed (e.g. BGR, mean subtracted)
        # VGG19 expects 3 channels.
        dummy_preprocessed_input = tf.random.normal((1, 224, 224, 3), dtype=tf.float32)
        outputs = style_content_model_instance(dummy_preprocessed_input)
        print("StyleContentModel instantiated and called successfully.")
        print("Content feature keys:", list(outputs['content'].keys()))
        print("Style feature keys:", list(outputs['style'].keys()))
    except Exception as e:
        print(f"Error during StyleContentModel instantiation or call: {e}")
