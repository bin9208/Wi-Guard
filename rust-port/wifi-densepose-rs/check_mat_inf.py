import scipy.io
import numpy as np

path = r"D:\MMFi_Unified\S01\A01\wifi-csi\frame001.mat"
data = scipy.io.loadmat(path)
amp = data["CSIamp"]
print(f"MAT CSIamp shape: {amp.shape}")
print(f"MAT CSIamp Any Inf: {np.isinf(amp).any()}")
print(f"MAT CSIamp Any NaN: {np.isnan(amp).any()}")
if np.isinf(amp).any():
    print(f"MAT CSIamp Inf locations: {np.where(np.isinf(amp))}")
