import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, Dropout
)
from tensorflow.keras.models import Model

def build_multi_input_model(input_shape_rgb=(128, 128, 3), input_shape_seg=(128, 128, 3), input_shape_meta=(3,)):
    """
    Build a multi-input neural network model.
    
    Parameters:
    - input_shape_rgb: Shape of the RGB input (e.g., (128, 128, 3)).
    - input_shape_seg: Shape of the segmentation input (e.g., (128, 128, 3)).
    - input_shape_meta: Shape of the metadata input (e.g., (3,)).
    
    Returns:
    - model: Compiled Keras model.
    """
    
    # RGB Branch
    rgb_input = Input(shape=input_shape_rgb, name="rgb_input")
    x_rgb = Conv2D(32, (3, 3), activation='relu', padding='same')(rgb_input)
    x_rgb = MaxPooling2D(pool_size=(2, 2))(x_rgb)
    x_rgb = Conv2D(64, (3, 3), activation='relu', padding='same')(x_rgb)
    x_rgb = MaxPooling2D(pool_size=(2, 2))(x_rgb)
    x_rgb = Flatten()(x_rgb)

    # Segmentation Branch
    seg_input = Input(shape=input_shape_seg, name="seg_input")
    x_seg = Conv2D(32, (3, 3), activation='relu', padding='same')(seg_input)
    x_seg = MaxPooling2D(pool_size=(2, 2))(x_seg)
    x_seg = Conv2D(64, (3, 3), activation='relu', padding='same')(x_seg)
    x_seg = MaxPooling2D(pool_size=(2, 2))(x_seg)
    x_seg = Flatten()(x_seg)

    # Metadata Branch
    meta_input = Input(shape=input_shape_meta, name="meta_input")
    x_meta = Dense(32, activation='relu')(meta_input)
    x_meta = Dropout(0.2)(x_meta)

    # Fusion Layer
    x = Concatenate()([x_rgb, x_seg, x_meta])
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)

    # Output Layer
    output = Dense(3, activation='tanh', name="controls_output")(x)  # Predict throttle, steer, brake

    # Define Model
    model = Model(inputs=[rgb_input, seg_input, meta_input], outputs=output)

    # Compile Model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse',  # Mean Squared Error for regression
                  metrics=['mae'])  # Mean Absolute Error

    return model

# Build and summarize the model
model = build_multi_input_model()
model.summary()
