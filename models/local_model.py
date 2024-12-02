import tensorflow as tf
from tensorflow.keras import layers, models, Input
import numpy as np
import os

def build_model(image_shape=(128, 128, 3), aux_input_dim=4, output_dim=3):
    """
    Builds a multimodal model for training on local data.

    Parameters:
    - image_shape: Shape of input images (e.g., (128, 128, 3)).
    - aux_input_dim: Dimension of auxiliary inputs (e.g., frame, hlc, light, measurements).
    - output_dim: Dimension of the output (e.g., controls with [throttle, steer, brake]).

    Returns:
    - model: Compiled Keras model.
    """
    # Image processing branch
    image_input = Input(shape=image_shape, name='image_input')
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(image_input)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)

    # Auxiliary inputs branch
    aux_input = Input(shape=(aux_input_dim,), name='aux_input')
    y = layers.Dense(64, activation='relu')(aux_input)
    y = layers.Dense(32, activation='relu')(y)

    # Concatenate and output
    combined = layers.concatenate([x, y])
    z = layers.Dense(64, activation='relu')(combined)
    z = layers.Dense(output_dim, activation='linear', name='output')(z)

    model = models.Model(inputs=[image_input, aux_input], outputs=z)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    return model


