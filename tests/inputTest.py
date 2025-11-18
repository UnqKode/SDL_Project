import numpy as np
def validate_shape(input:np.ndarray):
    if input.ndim != 2:
            raise ValueError("Sample must be 2D: shape (seq_len, features).")
    
def validate_dim(input:np.ndarray):
      
      if input.shape != (6,128):
            print(input.shape)
            raise ValueError("Sample must have shape (128, 6) for UCI-HAR dataset.")
        