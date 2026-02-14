import h5py
import numpy as np

with h5py.File('demo_data.h5', 'r') as f:
    # Use 'action' singular as seen in your logs
    actions = f['data/demo_0/action'][:] 
    
    print(f"Action Shape: {actions.shape}")
    print(f"First 5 Actions (XYZ + Gripper):\n{actions[:5]}")
    
    # Check if the values are Absolute (large) or Relative (tiny)
    max_val = np.max(np.abs(actions[:, :3]))
    print(f"\nMax XYZ Magnitude: {max_val}")
    
    if max_val < 0.1:
        print("RESULT: Data is RELATIVE (Deltas).")
    else:
        print("RESULT: Data is ABSOLUTE (Coordinates).")