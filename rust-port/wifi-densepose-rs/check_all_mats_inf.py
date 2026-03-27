import scipy.io
import numpy as np
import glob
import os

files = glob.glob(r"D:\MMFi_Unified\S01\A01\wifi-csi\frame*.mat")
inf_files = []
for path in files:
    data = scipy.io.loadmat(path)
    amp = data["CSIamp"]
    if np.isinf(amp).any() or np.isnan(amp).any():
        inf_files.append(os.path.basename(path))

print(f"Total files checked: {len(files)}")
print(f"Files with inf/nan: {len(inf_files)}")
if inf_files:
    print(f"Sample corrupt files: {inf_files[:5]}")
