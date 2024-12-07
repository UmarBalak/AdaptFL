from keras import layers, models

def build_model(input_shapes):
    """
    Builds a multi-input model for predicting controls using multiple modalities.

    Parameters:
    - input_shapes: Dictionary of input shapes for each modality.

    Returns:
    - model: A Keras model.
    """
    inputs = []

    # RGB images input
    rgb_input = layers.Input(shape=input_shapes['rgb'], name="rgb")
    rgb_branch = layers.Conv2D(32, (3, 3), activation='relu')(rgb_input)
    rgb_branch = layers.MaxPooling2D()(rgb_branch)
    rgb_branch = layers.Flatten()(rgb_branch)
    inputs.append(rgb_input)

    # Segmentation masks input
    segmentation_input = layers.Input(shape=input_shapes['segmentation'], name="segmentation")
    segmentation_branch = layers.Conv2D(32, (3, 3), activation='relu')(segmentation_input)
    segmentation_branch = layers.MaxPooling2D()(segmentation_branch)
    segmentation_branch = layers.Flatten()(segmentation_branch)
    inputs.append(segmentation_input)

    # High-Level Command input
    hlc_input = layers.Input(shape=input_shapes['hlc'], name="hlc")
    hlc_branch = layers.Dense(32, activation='relu')(hlc_input)
    inputs.append(hlc_input)

    # Traffic Light Status input
    light_input = layers.Input(shape=input_shapes['light'], name="light")
    light_branch = layers.Dense(32, activation='relu')(light_input)
    inputs.append(light_input)

    # Measurements input (speed)
    measurements_input = layers.Input(shape=input_shapes['measurements'], name="measurements")
    measurements_branch = layers.Dense(32, activation='relu')(measurements_input)
    inputs.append(measurements_input)

    # Concatenate all branches
    concatenated = layers.concatenate([rgb_branch, segmentation_branch, hlc_branch, light_branch, measurements_branch])

    # Output layer (for control predictions)
    output = layers.Dense(3, activation='tanh', name="controls")(concatenated)

    # Define the model
    model = models.Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    return model

# Example usage with input shapes
input_shapes = {
    "rgb": (128, 128, 3),  # Adjust size as per preprocessing
    "segmentation": (128, 128, 3),
    "hlc": (1,),
    "light": (1,),
    "measurements": (1,)
}