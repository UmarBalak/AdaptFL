# compression_utils.py

import numpy as np
import zlib
import pickle
from typing import List, Tuple

def compress_weights(weights: List[np.ndarray]) -> bytes:
    """Compress model weights using quantization and zlib."""
    compressed_weights = []
    
    for w in weights:
        # Store original shape and dtype
        shape = w.shape
        dtype = str(w.dtype)
        
        # Quantize to float16 for smaller size
        quantized = w.astype(np.float16)
        
        # Pack metadata and weights
        layer_data = (shape, dtype, quantized)
        compressed_weights.append(layer_data)
    
    # Serialize and compress
    serialized = pickle.dumps(compressed_weights)
    compressed = zlib.compress(serialized)
    
    return compressed

def decompress_weights(compressed_data: bytes) -> List[np.ndarray]:
    """Decompress weights back to original format."""
    # Decompress and deserialize
    decompressed = zlib.decompress(compressed_data)
    weight_data = pickle.loads(decompressed)
    
    # Reconstruct weights
    weights = []
    for shape, dtype, quantized in weight_data:
        # Convert back to original dtype
        w = quantized.astype(dtype)
        weights.append(w)
    
    return weights