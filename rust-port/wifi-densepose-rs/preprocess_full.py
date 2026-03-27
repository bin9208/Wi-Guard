import os
import glob
import numpy as np
import scipy.io
from multiprocessing import Pool
import time

def process_action_folder(action_path):
    """
    Processes a single action folder: combines all frame*.mat into wifi_csi.npy and wifi_csi_phase.npy
    """
    csi_dir = os.path.join(action_path, "wifi-csi")
    if not os.path.exists(csi_dir):
        return None
    
    mat_files = sorted(glob.glob(os.path.join(csi_dir, "frame*.mat")))
    if not mat_files:
        return None
    
    print(f"Processing {action_path} ({len(mat_files)} frames)...")
    
    all_amp = []
    all_phase = []
    for mat_file in mat_files:
        try:
            data = scipy.io.loadmat(mat_file)
            if "CSIamp" in data:
                # Clamp non-finite values (inf, nan) immediately
                amp_raw = data["CSIamp"]
                amp_fixed = np.nan_to_num(amp_raw, nan=0.0, posinf=100.0, neginf=-100.0)
                all_amp.append(amp_fixed)
            if "CSIphase" in data:
                phase_raw = data["CSIphase"]
                phase_fixed = np.nan_to_num(phase_raw, nan=0.0, posinf=np.pi, neginf=-np.pi)
                all_phase.append(phase_fixed)
        except Exception as e:
            print(f"Error reading {mat_file}: {e}")
            
    if not all_amp or not all_phase:
        return None
        
    # Average over the last dimension (packets) if it exists
    # Raw shape: (1, 3, 114, 10) or (3, 114, 10)
    # We want (1, 3, 114) per frame.
    
    processed_amp = []
    processed_phase = []
    
    for a, p in zip(all_amp, all_phase):
        # Average over last dim if 4D, or just use as is if 3D
        if a.ndim == 4:
            a = np.mean(a, axis=-1)
            p = np.mean(p, axis=-1)
        elif a.ndim == 3 and a.shape[-1] == 10: # (3, 114, 10) case
            a = np.mean(a, axis=-1)
            p = np.mean(p, axis=-1)
            
        # Ensure (1, 3, 114)
        if a.shape == (3, 114):
            a = a.reshape(1, 3, 114)
            p = p.reshape(1, 3, 114)
        
        processed_amp.append(a)
        processed_phase.append(p)

    amp_combined = np.array(processed_amp, dtype=np.float32) # Shape: (N, 1, 3, 114)
    phase_combined = np.array(processed_phase, dtype=np.float32)
    
    amp_output = os.path.join(action_path, "wifi_csi.npy")
    phase_output = os.path.join(action_path, "wifi_csi_phase.npy")
    
    np.save(amp_output, amp_combined)
    np.save(phase_output, phase_combined)
    
    return (amp_output, phase_output)

def main():
    base_dir = r"D:\MMFi_Unified"
    
    # 1. Find all action folders (A??)
    # The structure mentioned by user is S??/A??/wifi-csi/
    action_folders = []
    for s_folder in glob.glob(os.path.join(base_dir, "S*")):
        for a_folder in glob.glob(os.path.join(s_folder, "A*")):
            if os.path.isdir(os.path.join(a_folder, "wifi-csi")):
                action_folders.append(a_folder)
    
    if not action_folders:
        print(f"No action folders found in {base_dir} using S??/A?? structure.")
        # Fallback to recursively finding wifi-csi
        for root, dirs, files in os.walk(base_dir):
            if "wifi-csi" in dirs:
                action_folders.append(root)
    
    print(f"Found {len(action_folders)} action folders to process.")
    
    start_time = time.time()
    
    # 2. Process in parallel using 32 threads (5950X)
    with Pool(processes=32) as pool:
        results = pool.map(process_action_folder, action_folders)
    
    end_time = time.time()
    processed_count = len([r for r in results if r is not None])
    
    print(f"\nDone!")
    print(f"Processed {processed_count} folders.")
    print(f"Total time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
