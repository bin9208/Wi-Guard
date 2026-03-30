#!/usr/bin/env python3
"""
ESP32 CSI Fine-tuning Pipeline for WiFi-DensePose
==================================================

Collects ESP32 CSI data + smartphone camera ground truth, then fine-tunes
the ModalityTranslator to adapt the model from MM-Fi (5GHz, 3-antenna)
to ESP32 (2.4GHz, 1-antenna) CSI distribution.

Usage:
  # Step 1: Record CSI from the server (30-60 seconds per action)
  python scripts/finetune_esp32.py record --server http://NAS_IP:3000 --duration 30

  # Step 2: Extract keypoints from smartphone video
  python scripts/finetune_esp32.py extract-keypoints --video path/to/video.mp4

  # Step 3: Align CSI + keypoints by timestamp
  python scripts/finetune_esp32.py align --csi data/finetune/csi_rec.jsonl --keypoints data/finetune/keypoints.npy

  # Step 4: Fine-tune (runs on the server via Rust training pipeline)
  python scripts/finetune_esp32.py finetune --dataset data/finetune/aligned/ --epochs 50

  # All-in-one: record + extract + align + train
  python scripts/finetune_esp32.py all --server http://NAS_IP:3000 --video video.mp4 --epochs 50

Requirements:
  pip install mediapipe opencv-python numpy requests
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────

N_SUB = 56          # subcarriers
N_KEYPOINTS = 17    # COCO keypoints
WINDOW_FRAMES = 100 # temporal window for model input
FPS_CSI = 20        # ESP32 CSI rate after rate-limiting (edge_processing.c)

COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# MediaPipe Pose → COCO 17 mapping
# MediaPipe has 33 landmarks; we pick the 17 that match COCO
MP_TO_COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: Record CSI from the running server
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_record(args):
    """Start/stop CSI recording on the server via REST API."""
    import requests

    base = args.server.rstrip("/")
    out_dir = ensure_dir(Path(args.output))
    rec_id = args.id or f"finetune_{int(time.time())}"

    print(f"[1/3] Starting CSI recording on {base} ...")
    print(f"      Recording ID: {rec_id}")
    print(f"      Duration: {args.duration}s")
    print()
    print("=" * 60)
    print("  NOW: Perform your actions in front of the ESP32 nodes!")
    print("  Actions: stand, walk, sit, wave, raise arms, turn around")
    print("=" * 60)
    print()

    # Start recording
    resp = requests.post(f"{base}/api/v1/recording/start", json={"id": rec_id})
    data = resp.json()
    if not data.get("success"):
        print(f"ERROR: {data.get('error', 'unknown')}")
        sys.exit(1)

    # Wait for duration
    for remaining in range(args.duration, 0, -1):
        print(f"\r  Recording... {remaining}s remaining  ", end="", flush=True)
        time.sleep(1)
    print()

    # Stop recording
    resp = requests.post(f"{base}/api/v1/recording/stop")
    data = resp.json()
    frames = data.get("frames_recorded", "?")
    print(f"  Recording stopped: {frames} frames captured")
    print(f"  File: data/recordings/{rec_id}.jsonl (on the server)")
    print()
    print(f"[INFO] Download the JSONL from the server:")
    print(f"       scp NAS_IP:/app/data/recordings/{rec_id}.jsonl {out_dir}/")
    print(f"       OR: docker cp CONTAINER:/app/data/recordings/{rec_id}.jsonl {out_dir}/")


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: Extract COCO 17-keypoints from video using MediaPipe
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_extract_keypoints(args):
    """Extract COCO 17-keypoint coordinates from a video file using MediaPipe."""
    try:
        import cv2
        import mediapipe as mp
    except ImportError:
        print("ERROR: Install dependencies first:")
        print("  pip install mediapipe opencv-python")
        sys.exit(1)

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)

    out_dir = ensure_dir(Path(args.output))
    out_file = out_dir / "keypoints.npy"
    out_vis = out_dir / "keypoints_vis.npy"  # visibility/confidence

    print(f"[2/3] Extracting keypoints from: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps > 0 else 0

    print(f"      Video: {total_frames} frames @ {video_fps:.1f} FPS ({duration:.1f}s)")

    # Target FPS to match CSI rate
    target_fps = FPS_CSI
    frame_skip = max(1, int(video_fps / target_fps))
    print(f"      Sampling every {frame_skip} frames → ~{video_fps/frame_skip:.1f} FPS")

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,  # highest accuracy
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    all_keypoints = []
    all_visibility = []
    frame_idx = 0
    processed = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                kps = np.zeros((N_KEYPOINTS, 2), dtype=np.float32)
                vis = np.zeros(N_KEYPOINTS, dtype=np.float32)

                for coco_idx, mp_idx in enumerate(MP_TO_COCO):
                    lm = results.pose_landmarks.landmark[mp_idx]
                    # MediaPipe: x=[0,1] left→right, y=[0,1] top→bottom
                    # This matches COCO/MM-Fi convention
                    kps[coco_idx] = [lm.x, lm.y]
                    vis[coco_idx] = lm.visibility

                all_keypoints.append(kps)
                all_visibility.append(vis)
            else:
                # No detection → zero keypoints (will be masked during training)
                all_keypoints.append(np.zeros((N_KEYPOINTS, 2), dtype=np.float32))
                all_visibility.append(np.zeros(N_KEYPOINTS, dtype=np.float32))

            processed += 1
            if processed % 100 == 0:
                print(f"\r      Processed {processed} frames...", end="", flush=True)

        frame_idx += 1

    cap.release()
    pose.close()

    keypoints = np.array(all_keypoints, dtype=np.float32)  # [T, 17, 2]
    visibility = np.array(all_visibility, dtype=np.float32)  # [T, 17]

    np.save(out_file, keypoints)
    np.save(out_vis, visibility)

    print(f"\n      Saved {len(keypoints)} keypoint frames")
    print(f"      Keypoints: {out_file}")
    print(f"      Visibility: {out_vis}")

    # Stats
    valid = (visibility > 0.5).sum(axis=0)
    print(f"\n      Per-joint detection rate:")
    for i, name in enumerate(COCO_KEYPOINT_NAMES):
        pct = 100 * valid[i] / len(visibility) if len(visibility) > 0 else 0
        print(f"        {name:20s}: {pct:5.1f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3: Convert JSONL CSI recording → numpy arrays
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_convert_csi(args):
    """Convert server JSONL recording to numpy arrays matching MM-Fi format."""
    jsonl_path = Path(args.csi)
    if not jsonl_path.exists():
        print(f"ERROR: JSONL not found: {jsonl_path}")
        sys.exit(1)

    out_dir = ensure_dir(Path(args.output))

    print(f"[2.5] Converting CSI: {jsonl_path}")

    # Parse JSONL frames
    frames_by_node = {}
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Extract node_id and CSI data from the broadcast format
            node_id = obj.get("node_id", obj.get("esp32_node_id", 1))
            amp = obj.get("amplitudes", obj.get("amplitude", []))
            ph = obj.get("phases", obj.get("phase", []))
            ts = obj.get("timestamp", obj.get("ts", 0))

            if len(amp) >= N_SUB and len(ph) >= N_SUB:
                if node_id not in frames_by_node:
                    frames_by_node[node_id] = []
                frames_by_node[node_id].append({
                    "amp": np.array(amp[:N_SUB], dtype=np.float64),
                    "phase": np.array(ph[:N_SUB], dtype=np.float64),
                    "ts": ts,
                })

    if not frames_by_node:
        print("ERROR: No valid CSI frames found in JSONL")
        sys.exit(1)

    # Use the node with most frames
    best_node = max(frames_by_node, key=lambda k: len(frames_by_node[k]))
    frames = frames_by_node[best_node]
    print(f"      Using node {best_node}: {len(frames)} frames")
    print(f"      Other nodes: {[(k, len(v)) for k, v in frames_by_node.items() if k != best_node]}")

    # Stack into arrays
    # Training format: [T, n_tx=1, n_rx=3, n_sub=56]
    # ESP32 has 1 antenna, replicate to 3 for training compatibility
    T = len(frames)
    wifi_csi_amp = np.zeros((T, 1, 3, N_SUB), dtype=np.float64)
    wifi_csi_phase = np.zeros((T, 1, 3, N_SUB), dtype=np.float64)

    for t, fr in enumerate(frames):
        for rx in range(3):
            wifi_csi_amp[t, 0, rx, :] = fr["amp"]
            wifi_csi_phase[t, 0, rx, :] = fr["phase"]

    # Apply training normalization:
    # Amplitude: 20*log10(amp) → clamp(-15, 65) → (dB + 15) / 80
    amp_db = 20.0 * np.log10(np.abs(wifi_csi_amp) + 1e-6)
    amp_norm = np.clip(amp_db, -15, 65)
    wifi_csi_amp = (amp_norm + 15.0) / 80.0

    # Phase: raw radians → / pi
    wifi_csi_phase = wifi_csi_phase / np.pi

    np.save(out_dir / "wifi_csi.npy", wifi_csi_amp.astype(np.float32))
    np.save(out_dir / "wifi_csi_phase.npy", wifi_csi_phase.astype(np.float32))

    print(f"      Saved: wifi_csi.npy shape={wifi_csi_amp.shape}")
    print(f"      Saved: wifi_csi_phase.npy shape={wifi_csi_phase.shape}")

    # Save timestamps for alignment
    timestamps = np.array([fr["ts"] for fr in frames])
    np.save(out_dir / "csi_timestamps.npy", timestamps)
    print(f"      Saved: csi_timestamps.npy ({len(timestamps)} entries)")


# ═══════════════════════════════════════════════════════════════════════════════
# Step 4: Align CSI windows + keypoints into training samples
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_align(args):
    """Align CSI frames with keypoint labels into training-ready windows."""
    out_dir = ensure_dir(Path(args.output))
    data_dir = Path(args.data_dir)

    csi_amp = np.load(data_dir / "wifi_csi.npy")        # [T_csi, 1, 3, 56]
    csi_phase = np.load(data_dir / "wifi_csi_phase.npy") # [T_csi, 1, 3, 56]
    keypoints = np.load(data_dir / "keypoints.npy")      # [T_kp, 17, 2]
    visibility = np.load(data_dir / "keypoints_vis.npy") # [T_kp, 17]

    T_csi = len(csi_amp)
    T_kp = len(keypoints)

    print(f"[3/4] Aligning data:")
    print(f"      CSI frames: {T_csi}")
    print(f"      Keypoint frames: {T_kp}")

    # Simple alignment: linearly map keypoint frames to CSI frames
    # Both should cover the same time span
    T_min = min(T_csi, T_kp)
    if T_csi > T_kp:
        # Downsample CSI to match keypoints
        indices = np.linspace(0, T_csi - 1, T_min, dtype=int)
        csi_amp = csi_amp[indices]
        csi_phase = csi_phase[indices]
    elif T_kp > T_csi:
        # Downsample keypoints to match CSI
        indices = np.linspace(0, T_kp - 1, T_min, dtype=int)
        keypoints = keypoints[indices]
        visibility = visibility[indices]

    T = len(csi_amp)
    print(f"      Aligned length: {T} frames")

    # Create sliding windows of WINDOW_FRAMES
    stride = WINDOW_FRAMES // 2  # 50% overlap
    samples_amp = []
    samples_phase = []
    samples_kp = []
    samples_vis = []

    for start in range(0, T - WINDOW_FRAMES + 1, stride):
        end = start + WINDOW_FRAMES
        mid = (start + end) // 2  # keypoint label at window midpoint

        samples_amp.append(csi_amp[start:end])
        samples_phase.append(csi_phase[start:end])
        samples_kp.append(keypoints[mid])
        samples_vis.append(visibility[mid])

    if not samples_amp:
        print("ERROR: Not enough frames for even one window")
        print(f"  Need {WINDOW_FRAMES} frames, have {T}")
        sys.exit(1)

    X_amp = np.array(samples_amp, dtype=np.float32)   # [N, 100, 1, 3, 56]
    X_phase = np.array(samples_phase, dtype=np.float32)
    Y_kp = np.array(samples_kp, dtype=np.float32)     # [N, 17, 2]
    Y_vis = np.array(samples_vis, dtype=np.float32)    # [N, 17]

    # Save in MM-Fi compatible format for each sample
    dataset_dir = ensure_dir(out_dir / "dataset")
    for i in range(len(X_amp)):
        sample_dir = ensure_dir(dataset_dir / f"S{i:04d}")
        np.save(sample_dir / "wifi_csi.npy", X_amp[i])
        np.save(sample_dir / "wifi_csi_phase.npy", X_phase[i])
        np.save(sample_dir / "ground-truth.npy", Y_kp[i])
        np.save(sample_dir / "visibility.npy", Y_vis[i])

    print(f"      Created {len(X_amp)} training samples (stride={stride})")
    print(f"      Saved to: {dataset_dir}/")

    # Also save as single arrays for easy loading
    np.save(out_dir / "X_amp.npy", X_amp)
    np.save(out_dir / "X_phase.npy", X_phase)
    np.save(out_dir / "Y_keypoints.npy", Y_kp)
    np.save(out_dir / "Y_visibility.npy", Y_vis)
    print(f"      Bulk arrays: X_amp={X_amp.shape}, Y_keypoints={Y_kp.shape}")


# ═══════════════════════════════════════════════════════════════════════════════
# Step 5: Fine-tune the model
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_finetune(args):
    """Fine-tune WiFiDensePoseModel's ModalityTranslator on ESP32 CSI data."""
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        print("ERROR: PyTorch required. Install with:")
        print("  pip install torch")
        sys.exit(1)

    data_dir = Path(args.dataset)
    checkpoint = Path(args.checkpoint)
    out_dir = ensure_dir(Path(args.output))

    if not checkpoint.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint}")
        sys.exit(1)

    # Load training data
    X_amp = np.load(data_dir / "X_amp.npy")          # [N, 100, 1, 3, 56]
    X_phase = np.load(data_dir / "X_phase.npy")
    Y_kp = np.load(data_dir / "Y_keypoints.npy")     # [N, 17, 2]
    Y_vis = np.load(data_dir / "Y_visibility.npy")    # [N, 17]

    N = len(X_amp)
    print(f"[4/4] Fine-tuning on {N} samples")
    print(f"      Checkpoint: {checkpoint}")
    print(f"      Epochs: {args.epochs}")
    print(f"      LR: {args.lr}")

    # Reshape to model input format: [N, T*n_ant, n_sub] = [N, 300, 56]
    # Flatten [100, 1, 3, 56] → [100, 3, 56] → [300, 56] in time-major order
    amp_flat = X_amp.reshape(N, WINDOW_FRAMES, 3, N_SUB).reshape(N, -1, N_SUB)
    phase_flat = X_phase.reshape(N, WINDOW_FRAMES, 3, N_SUB).reshape(N, -1, N_SUB)

    amp_t = torch.from_numpy(amp_flat).float()
    phase_t = torch.from_numpy(phase_flat).float()
    kp_t = torch.from_numpy(Y_kp).float()
    vis_t = torch.from_numpy(Y_vis).float()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"      Device: {device}")

    # Load the safetensors model
    # We use tch-rs format, so load with safetensors library
    try:
        from safetensors import safe_open
        from safetensors.torch import save_file

        tensors = {}
        with safe_open(str(checkpoint), framework="pt", device=str(device)) as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

        print(f"      Loaded {len(tensors)} parameters from checkpoint")

        # Identify ModalityTranslator parameters (to unfreeze)
        translator_keys = [k for k in tensors if "translator" in k.lower() or "conv1" in k or "gn1" in k or "conv2" in k or "gn2" in k or "out_conv" in k]
        backbone_keys = [k for k in tensors if k not in translator_keys]

        print(f"      Translator params: {len(translator_keys)}")
        print(f"      Backbone params (frozen): {len(backbone_keys)}")

    except ImportError:
        print("ERROR: safetensors library required:")
        print("  pip install safetensors")
        sys.exit(1)

    print()
    print("=" * 60)
    print("  Fine-tuning must run via the Rust training pipeline.")
    print("  Use the server's --train flag with --dataset pointing")
    print("  to the aligned data directory.")
    print()
    print("  Command:")
    print(f"    sensing-server --train \\")
    print(f"      --dataset {data_dir / 'dataset'} \\")
    print(f"      --epochs {args.epochs} \\")
    print(f"      --checkpoint-dir {out_dir} \\")
    print(f"      --model-checkpoint {checkpoint}")
    print()
    print("  Or via Docker:")
    print(f"    docker exec -it ruview-nas sensing-server --train \\")
    print(f"      --dataset /app/data/finetune/dataset \\")
    print(f"      --epochs {args.epochs} \\")
    print(f"      --checkpoint-dir /app/checkpoint \\")
    print(f"      --model-checkpoint /app/checkpoint/model.safetensors")
    print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
# Main CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="ESP32 CSI Fine-tuning Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # record
    p_rec = sub.add_parser("record", help="Record CSI data from the server")
    p_rec.add_argument("--server", required=True, help="Server URL (e.g. http://192.168.1.100:3000)")
    p_rec.add_argument("--duration", type=int, default=60, help="Recording duration in seconds")
    p_rec.add_argument("--id", help="Recording ID (auto-generated if omitted)")
    p_rec.add_argument("--output", default="data/finetune", help="Output directory")

    # extract-keypoints
    p_kp = sub.add_parser("extract-keypoints", help="Extract keypoints from video")
    p_kp.add_argument("--video", required=True, help="Path to smartphone video")
    p_kp.add_argument("--output", default="data/finetune", help="Output directory")

    # convert-csi
    p_conv = sub.add_parser("convert-csi", help="Convert JSONL recording to numpy")
    p_conv.add_argument("--csi", required=True, help="Path to recording JSONL file")
    p_conv.add_argument("--output", default="data/finetune", help="Output directory")

    # align
    p_align = sub.add_parser("align", help="Align CSI + keypoints into training samples")
    p_align.add_argument("--data-dir", default="data/finetune", help="Directory with CSI + keypoint numpy files")
    p_align.add_argument("--output", default="data/finetune", help="Output directory")

    # finetune
    p_ft = sub.add_parser("finetune", help="Fine-tune the model")
    p_ft.add_argument("--dataset", required=True, help="Aligned dataset directory")
    p_ft.add_argument("--checkpoint", default="rust-port/wifi-densepose-rs/checkpoint/model.safetensors",
                       help="Path to pretrained model checkpoint")
    p_ft.add_argument("--epochs", type=int, default=50, help="Number of fine-tuning epochs")
    p_ft.add_argument("--lr", type=float, default=1e-4, help="Learning rate (lower than initial training)")
    p_ft.add_argument("--output", default="data/finetune/checkpoint", help="Output checkpoint directory")

    args = parser.parse_args()

    if args.command == "record":
        cmd_record(args)
    elif args.command == "extract-keypoints":
        cmd_extract_keypoints(args)
    elif args.command == "convert-csi":
        cmd_convert_csi(args)
    elif args.command == "align":
        cmd_align(args)
    elif args.command == "finetune":
        cmd_finetune(args)


if __name__ == "__main__":
    main()
