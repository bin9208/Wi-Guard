import scipy.io
import os
import numpy as np

path = r"D:\MMFi_Unified\S01\A01\wifi-csi\frame001.mat"
if os.path.exists(path):
    data = scipy.io.loadmat(path)
    if "CSIamp" in data:
        amp = data["CSIamp"]
        print(f"CSIamp shape: {amp.shape}")
        print(f"CSIamp dtype: {amp.dtype}")
    if "CSIphase" in data:
        phase = data["CSIphase"]
        print(f"CSIphase shape: {phase.shape}")
else:
    print(f"File not found: {path}")
