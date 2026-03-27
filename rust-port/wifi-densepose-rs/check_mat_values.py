import scipy.io
import numpy as np

path = r"D:\MMFi_Unified\S01\A01\wifi-csi\frame118.mat"
data = scipy.io.loadmat(path)
amp = data["CSIamp"]
print(f"MAT CSIamp shape: {amp.shape}")
print(f"MAT CSIamp min: {np.min(amp)}")
print(f"MAT CSIamp max: {np.max(amp)}")
print(f"Inf count: {np.isinf(amp).sum()}")
print(f"NaN count: {np.isnan(amp).sum()}")
