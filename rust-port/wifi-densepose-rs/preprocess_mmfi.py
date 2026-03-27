import os
import glob
import numpy as np
import scipy.io
from multiprocessing import Pool
import time

def process_action_folder(action_path):
    """
    Processes a single action folder: combines all frame*.mat into wifi_csi.npy
    """
    csi_dir = os.path.join(action_path, "wifi-csi")
    if not os.path.exists(csi_dir):
        return None
    
    mat_files = sorted(glob.glob(os.path.join(csi_dir, "frame*.mat")))
    if not mat_files:
        return None
    
    print(f"Processing {action_path} ({len(mat_files)} frames)...")
    
    all_csi = []
    for mat_file in mat_files:
        try:
            data = scipy.io.loadmat(mat_file)
            if "CSIamp" in data:
                all_csi.append(data["CSIamp"])
        except Exception as e:
            print(f"Error reading {mat_file}: {e}")
            
    if not all_csi:
        return None
        
    combined = np.array(all_csi) # Shape: (N, 1, 3, 114) or similar
    output_path = os.path.join(action_path, "wifi_csi.npy")
    np.save(output_path, combined)
    return output_path

def main():
    base_dir = r"D:\MMFi_Dataset"
    
    # 1. Find all action folders (A??)
    # Based on earlier check: D:\MMFi_Dataset\E??\S??\A??
    action_folders = []
    for e_folder in glob.glob(os.path.join(base_dir, "E*")):
        for s_folder in glob.glob(os.path.join(e_folder, "S*")):
            for a_folder in glob.glob(os.path.join(s_folder, "A*")):
                if os.path.isdir(os.path.join(a_folder, "wifi-csi")):
                    action_folders.append(a_folder)
    
    print(f"Found {len(action_folders)} action folders to process.")
    
    start_time = time.time()
    
    # 2. Process in parallel using 16 cores (5950X)
    with Pool(processes=16) as pool:
        results = pool.map(process_action_folder, action_folders)
    
    end_time = time.time()
    processed_count = len([r for r in results if r is not None])
    
    print(f"\nDone!")
    print(f"Processed {processed_count} folders.")
    print(f"Total time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
