import scipy.io
import numpy as np
import glob
import os

files = glob.glob(r"D:\MMFi_Unified\S01\A01\wifi-csi\frame*.mat")[:10]
for path in files:
    data = scipy.io.loadmat(path)
    amp = data["CSIamp"]
    print(f"File: {os.path.basename(path)}")
    print(f"  Shape: {amp.shape}, Dtype: {amp.dtype}")
    print(f"  Any Inf: {np.isinf(amp).any()}, Any NaN: {np.isnan(amp).any()}")
    print(f"  Min: {np.min(amp)}, Max: {np.max(amp)}")
