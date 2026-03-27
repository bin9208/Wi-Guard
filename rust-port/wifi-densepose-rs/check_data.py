import numpy as np
import os

path = r"D:\MMFi_Unified\S01\A01\wifi_csi.npy"
if os.path.exists(path):
    data = np.load(path)
    print(f"Data shape: {data.shape}")
    print(f"Any NaN: {np.isnan(data).any()}")
    print(f"Any Inf: {np.isinf(data).any()}")
    print(f"Max value: {np.max(data)}")
    print(f"Min value: {np.min(data)}")
    print(f"Mean: {np.mean(data)}")
else:
    print("File not found.")

path_ph = r"D:\MMFi_Unified\S01\A01\wifi_csi_phase.npy"
if os.path.exists(path_ph):
    data_ph = np.load(path_ph)
    print(f"Phase shape: {data_ph.shape}")
    print(f"Any NaN: {np.isnan(data_ph).any()}")
    print(f"Any Inf: {np.isinf(data_ph).any()}")
    print(f"Max value: {np.max(data_ph)}")
    print(f"Min value: {np.min(data_ph)}")
