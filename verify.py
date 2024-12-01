# verify preprocessing.py
import numpy as np
rgb_images = np.load("data/processed/rgb/rgb_images.npy")
print(rgb_images.shape)  # Should be (total_frames, 128, 128, 3)

segmentation_masks = np.load("data/processed/segmentation/segmentation_masks.npy")
print(segmentation_masks.shape)  # Should match total_frames

hlc_data = np.load("data/processed/metadata/hlc.npy")
print(hlc_data.shape)  # Should match total_frames

# verify model
from models import build_multi_input_model

model = build_multi_input_model()
# print(model.summary())
