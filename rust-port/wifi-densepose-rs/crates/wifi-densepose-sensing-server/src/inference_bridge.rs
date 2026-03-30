//! Real-time GPU inference bridge for WiFi-DensePose.
//!
//! Supports two checkpoint formats:
//! - `.safetensors` — VarStore weights saved by the training pipeline (no zscore.json needed)
//! - `.pt` — TorchScript traced/scripted models (optional zscore.json for external normalization)

use std::collections::VecDeque;
use std::fs;
use std::path::PathBuf;
use serde::Deserialize;
use tokio::sync::{mpsc, watch};
use tch::{Device, Tensor, CModule, Kind};
use tracing::{info, warn};
use rand::Rng;

use wifi_densepose_train::config::TrainingConfig;
use wifi_densepose_train::model::WiFiDensePoseModel;

use super::{PersonDetection, PoseKeypoint, BoundingBox};

pub const N_SUB: usize = 56;
pub const N_ANTENNAS: usize = 1;
pub const WINDOW_SIZE: usize = 100;
pub const N_CHANNELS: usize = 6;

/// Stack-allocated CSI frame data for inference.
#[derive(Debug, Clone, Copy)]
pub struct CsiFrameSlim {
    pub node_id: u8,
    pub data: [f32; 336],
}

impl CsiFrameSlim {
    /// Build a frame from raw ESP32 CSI, applying the same preprocessing as
    /// the MM-Fi training pipeline (trainer.rs):
    ///
    /// **Amplitude:** `20·log10(amp)` → clamp(−15, 65) → `(dB + 15) / 80` → [0, 1]
    /// **Phase:**     raw radians → `phase / π`                             → [−1, 1]
    ///
    /// These are the exact transforms applied in `trainer.rs` before feeding
    /// batches to the model. Applying them here ensures inference inputs sit
    /// in the same distribution the model was trained on.
    pub fn from_esp32(node_id: u8, amplitudes: &[f64], phases: &[f64]) -> Self {
        use std::f64::consts::PI;
        let mut data = [0.0f32; 336];

        // ── Amplitude: Channel 0 (indices 0..56) ─────────────────────────
        // Training: amp_t = (amp_t.clamp(-15, 65) + 15) / 80
        for (i, &amp_raw) in amplitudes.iter().take(56).enumerate() {
            let db = 20.0 * (amp_raw.abs() + 1e-6).log10();
            let db_clamped = db.max(-15.0).min(65.0);
            data[i] = ((db_clamped + 15.0) / 80.0) as f32;
        }

        // ── Phase: Channel 3 (indices 168..224) ──────────────────────────
        // Training: ph_t = ph_t / π  →  range [−1, 1]
        for (i, &phase) in phases.iter().take(56).enumerate() {
            data[i + 168] = (phase / PI) as f32;
        }

        // ── Antenna replication with independent noise ────────────────────
        // The model was trained with n_rx=3 (MM-Fi dataset).  ESP32 has 1
        // antenna, so we replicate + add small independent Gaussian noise
        // to each slot. This prevents GroupNorm degeneracy (identical channels
        // → zero inter-channel variance) and mimics the training augmentation
        // (virtual_aug.rs: σ_amp=0.03, σ_phase=0.02).
        //
        // Layout: [ch0_amp(0..56), ch1_amp(56..112), ch2_amp(112..168),
        //          ch3_ph(168..224), ch4_ph(224..280), ch5_ph(280..336)]
        let mut rng = rand::rng();
        for i in 0..N_SUB {
            // Amplitude noise: uniform ±0.02 (training aug used σ=0.03 Gaussian)
            let noise_a1: f32 = rng.random_range(-0.02_f32..0.02_f32);
            let noise_a2: f32 = rng.random_range(-0.02_f32..0.02_f32);
            data[i + 56]  = (data[i] + noise_a1).clamp(0.0, 1.0);
            data[i + 112] = (data[i] + noise_a2).clamp(0.0, 1.0);

            // Phase noise: uniform ±0.015
            let noise_p1: f32 = rng.random_range(-0.015_f32..0.015_f32);
            let noise_p2: f32 = rng.random_range(-0.015_f32..0.015_f32);
            data[i + 224] = (data[i + 168] + noise_p1).clamp(-1.0, 1.0);
            data[i + 280] = (data[i + 168] + noise_p2).clamp(-1.0, 1.0);
        }

        Self { node_id, data }
    }
}

pub struct CsiWindowBuffer {
    buffer: VecDeque<CsiFrameSlim>,
}

impl CsiWindowBuffer {
    pub fn new() -> Self {
        Self { buffer: VecDeque::with_capacity(WINDOW_SIZE) }
    }

    pub fn push(&mut self, frame: CsiFrameSlim) {
        if self.buffer.len() >= WINDOW_SIZE {
            self.buffer.pop_front();
        }
        self.buffer.push_back(frame);
    }

    pub fn is_ready(&self) -> bool {
        self.buffer.len() == WINDOW_SIZE
    }

    /// Build a raw CSI tensor `[1, 6, 100, 56]` from the window buffer.
    pub fn construct_input(&self, device: Device) -> Tensor {
        let mut flat = Vec::with_capacity(WINDOW_SIZE * 336);
        for frame in &self.buffer {
            flat.extend_from_slice(&frame.data);
        }

        Tensor::from_slice(&flat)
            .view([WINDOW_SIZE as i64, 6, 56])
            .permute(&[1, 0, 2])
            .unsqueeze(0)
            .to_device(device)
    }
}

/// Multi-node antenna array: combines CSI from 3 different ESP32 nodes
/// into a single `[1, 6, 100, 56]` tensor where each node acts as a
/// separate RX antenna (ch0=node_a, ch1=node_b, ch2=node_c).
///
/// This provides REAL spatial diversity instead of antenna replication.
struct MultiNodeAntennaArray {
    /// Per-node amplitude ring buffers: [node_idx][time][56 subcarriers]
    amp_bufs: [VecDeque<[f32; N_SUB]>; 3],
    /// Per-node phase ring buffers
    phase_bufs: [VecDeque<[f32; N_SUB]>; 3],
    /// Mapping: slot index → node_id
    node_slots: [Option<u8>; 3],
}

impl MultiNodeAntennaArray {
    fn new() -> Self {
        Self {
            amp_bufs: [
                VecDeque::with_capacity(WINDOW_SIZE),
                VecDeque::with_capacity(WINDOW_SIZE),
                VecDeque::with_capacity(WINDOW_SIZE),
            ],
            phase_bufs: [
                VecDeque::with_capacity(WINDOW_SIZE),
                VecDeque::with_capacity(WINDOW_SIZE),
                VecDeque::with_capacity(WINDOW_SIZE),
            ],
            node_slots: [None; 3],
        }
    }

    /// Assign a node_id to an antenna slot (0, 1, or 2).
    /// First 3 unique nodes are auto-assigned.
    fn slot_for(&mut self, node_id: u8) -> Option<usize> {
        // Check if already assigned
        for (i, slot) in self.node_slots.iter().enumerate() {
            if *slot == Some(node_id) {
                return Some(i);
            }
        }
        // Assign to first empty slot
        for (i, slot) in self.node_slots.iter_mut().enumerate() {
            if slot.is_none() {
                *slot = Some(node_id);
                return Some(i);
            }
        }
        None // All 3 slots taken by other nodes
    }

    /// Push a frame's amplitude and phase into the appropriate antenna slot.
    fn push(&mut self, frame: &CsiFrameSlim) {
        let slot = match self.slot_for(frame.node_id) {
            Some(s) => s,
            None => return, // Node not in this array's 3 slots
        };

        // Extract amp (ch0: indices 0..56) and phase (ch3: indices 168..224)
        let mut amp = [0.0f32; N_SUB];
        let mut phase = [0.0f32; N_SUB];
        for i in 0..N_SUB {
            amp[i] = frame.data[i];
            phase[i] = frame.data[i + 168];
        }

        if self.amp_bufs[slot].len() >= WINDOW_SIZE {
            self.amp_bufs[slot].pop_front();
            self.phase_bufs[slot].pop_front();
        }
        self.amp_bufs[slot].push_back(amp);
        self.phase_bufs[slot].push_back(phase);
    }

    /// How many antenna slots have at least 1 frame?
    fn active_slots(&self) -> usize {
        self.node_slots.iter().filter(|s| s.is_some()).count()
    }

    /// Are all 3 antenna slots ready (have WINDOW_SIZE frames)?
    fn is_ready(&self) -> bool {
        self.active_slots() >= 2 && // at least 2 nodes
        self.amp_bufs.iter().enumerate().all(|(i, buf)| {
            self.node_slots[i].is_none() || buf.len() >= WINDOW_SIZE
        })
    }

    /// Build `[1, 6, 100, 56]` tensor where channels are real multi-node CSI.
    ///
    /// Layout: [ch0_amp(node_a), ch1_amp(node_b), ch2_amp(node_c),
    ///          ch3_phase(node_a), ch4_phase(node_b), ch5_phase(node_c)]
    fn construct_input(&self, device: Device) -> Tensor {
        // [100, 6, 56] → permute → [6, 100, 56] → unsqueeze → [1, 6, 100, 56]
        let mut flat = vec![0.0f32; WINDOW_SIZE * 6 * N_SUB];

        for t in 0..WINDOW_SIZE {
            for slot in 0..3 {
                let amp_data = if self.amp_bufs[slot].len() > t {
                    &self.amp_bufs[slot][t]
                } else if !self.amp_bufs[slot].is_empty() {
                    // Pad with last available frame
                    self.amp_bufs[slot].back().unwrap()
                } else {
                    // Empty slot: replicate from slot 0
                    if self.amp_bufs[0].len() > t {
                        &self.amp_bufs[0][t]
                    } else {
                        continue;
                    }
                };

                let phase_data = if self.phase_bufs[slot].len() > t {
                    &self.phase_bufs[slot][t]
                } else if !self.phase_bufs[slot].is_empty() {
                    self.phase_bufs[slot].back().unwrap()
                } else {
                    if self.phase_bufs[0].len() > t {
                        &self.phase_bufs[0][t]
                    } else {
                        continue;
                    }
                };

                let base = t * 6 * N_SUB;
                for s in 0..N_SUB {
                    flat[base + slot * N_SUB + s] = amp_data[s];
                    flat[base + (3 + slot) * N_SUB + s] = phase_data[s];
                }
            }
        }

        Tensor::from_slice(&flat)
            .view([WINDOW_SIZE as i64, 6, N_SUB as i64])
            .permute([1, 0, 2])
            .unsqueeze(0)
            .to_device(device)
    }
}

#[derive(Debug, Clone)]
pub struct InferencePipelineConfig {
    pub checkpoint_path: PathBuf,
    pub zscore_path: PathBuf,
    pub inference_stride: usize,
    pub confidence_threshold: f64,
    pub canvas_width: f64,
    pub canvas_height: f64,
}

#[derive(Deserialize)]
struct ZScoreStats {
    mean: Vec<f32>,
    std: Vec<f32>,
}

/// Loaded model — either a TorchScript CModule or a VarStore-based model.
enum LoadedModel {
    TorchScript {
        module: CModule,
        /// Z-score normalization tensors (mean, std) — `[1, 6, 1, 1]` each.
        mean: Tensor,
        std: Tensor,
    },
    SafeTensors(WiFiDensePoseModel),
}

impl LoadedModel {
    /// Run inference on raw CSI input `[1, 6, 100, 56]` and return keypoint
    /// heatmaps `[B, 17, H, W]`.
    fn infer(&self, raw_input: &Tensor) -> Result<Tensor, String> {
        match self {
            LoadedModel::TorchScript { module, mean, std } => {
                // TorchScript path: apply external z-score normalization
                let input_norm = (raw_input - mean) / (std + 1e-6);
                let amp_t = input_norm.narrow(1, 0, 3).contiguous();
                let phase_t = input_norm.narrow(1, 3, 3).contiguous();
                let ivalue_args = [tch::IValue::Tensor(amp_t), tch::IValue::Tensor(phase_t)];
                let output = module.forward_is(&ivalue_args).map_err(|e| e.to_string())?;
                extract_keypoints(&output).ok_or_else(|| "no keypoints in output".to_string())
            }
            LoadedModel::SafeTensors(model) => {
                // Safetensors path: 모델 내부 ModalityTranslator가 정규화 처리.
                // zscore.json 불필요 — 학습 시 원시 데이터 사용.
                //
                // 학습 데이터 순서 (trainer.rs collate): TIME-MAJOR
                //   flat_idx = t * n_ant + a  →  [t0_a0, t0_a1, t0_a2, t1_a0, ...]
                // ModalityTranslator: reshape([B, 300, 56]) → [B, T=100, n_ant=3, 56]
                //
                // raw_input: [1, 3, 100, 56] = [B, antenna, time, subcarrier]
                // reshape만 하면 antenna-major: [a0_t0..a0_t99, a1_t0..] ← 잘못됨!
                // permute(0,2,1,3) → [1, 100, 3, 56] = [B, time, antenna, sub]
                // 이후 reshape → [1, 300, 56] = time-major ← 학습과 일치!
                let amp_t = raw_input.narrow(1, 0, 3)
                    .permute([0, 2, 1, 3])   // [1,3,100,56] → [1,100,3,56]
                    .contiguous()
                    .reshape([-1, 300, 56]);  // [1,300,56] time-major
                let phase_t = raw_input.narrow(1, 3, 3)
                    .permute([0, 2, 1, 3])
                    .contiguous()
                    .reshape([-1, 300, 56]);
                let output = model.forward_inference(&amp_t, &phase_t);
                Ok(output.keypoints)
            }
        }
    }
}

fn is_safetensors(path: &std::path::Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("safetensors"))
        .unwrap_or(false)
}

pub fn spawn_inference_pipeline(
    config: InferencePipelineConfig,
) -> Result<(mpsc::Sender<CsiFrameSlim>, watch::Receiver<Vec<PersonDetection>>), Box<dyn std::error::Error>> {
    let device = Device::cuda_if_available();
    info!("Starting CSI Inference Pipeline on {:?}", device);

    let model = if is_safetensors(&config.checkpoint_path) {
        // ── Safetensors: VarStore 가중치 로딩 (zscore 불필요) ──
        info!("Loading safetensors checkpoint: {:?}", config.checkpoint_path);
        info!("Safetensors mode: z-score normalization skipped (model handles internally)");
        let train_cfg = TrainingConfig::default();
        let mut m = WiFiDensePoseModel::new(&train_cfg, device);
        m.load(&config.checkpoint_path)
            .map_err(|e| format!("safetensors load failed: {e}"))?;
        info!("Safetensors model loaded — {} parameters",
              m.var_store().variables().len());
        LoadedModel::SafeTensors(m)
    } else {
        // ── TorchScript: CModule + optional z-score normalization ──
        info!("Loading TorchScript checkpoint: {:?}", config.checkpoint_path);
        let cm = CModule::load_on_device(&config.checkpoint_path, device)?;

        let zscore: ZScoreStats = fs::read_to_string(&config.zscore_path)
            .and_then(|s| serde_json::from_str(&s)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e)))
            .unwrap_or_else(|_| {
                warn!("Z-score stats not found at {:?}, using identity normalization",
                      config.zscore_path);
                ZScoreStats { mean: vec![0.0; 6], std: vec![1.0; 6] }
            });

        let mean_t = Tensor::from_slice(&zscore.mean).to_device(device).view([1, 6, 1, 1]);
        let std_t = Tensor::from_slice(&zscore.std).to_device(device).view([1, 6, 1, 1]);

        LoadedModel::TorchScript { module: cm, mean: mean_t, std: std_t }
    };

    let (tx, mut rx) = mpsc::channel::<CsiFrameSlim>(100);
    let (pose_tx, pose_rx) = watch::channel(vec![]);

    std::thread::spawn(move || {
        // ── Multi-Node Antenna Array Mode ─────────────────────────────
        // 6 ESP32 nodes → 2 sets of 3-node antenna arrays
        //   Set A: first 3 nodes  → inference result A
        //   Set B: next  3 nodes  → inference result B
        //   Final: confidence-weighted ensemble of A + B
        //
        // Each set produces a `[1, 6, 100, 56]` tensor where channels
        // are REAL spatial diversity from physically separated nodes.
        let mut array_a = MultiNodeAntennaArray::new();
        let mut array_b = MultiNodeAntennaArray::new();

        // EMA-smoothed keypoints (persists across frames)
        let mut ema_keypoints: Option<Vec<(f64, f64)>> = None;
        const EMA_ALPHA: f64 = 0.3;

        let mut total_frames: usize = 0;
        let mut total_infer_count: usize = 0;

        // Per-set latest keypoints for ensemble
        let mut set_a_kpts: Option<Vec<PoseKeypoint>> = None;
        let mut set_b_kpts: Option<Vec<PoseKeypoint>> = None;

        // Fallback: single-node mode buffer (for ≤1 node scenarios)
        let mut fallback_buffer: Option<CsiWindowBuffer> = None;
        let mut fallback_node_id: Option<u8> = None;

        info!("Inference thread started — Multi-Node Antenna Array mode (up to 6 nodes → 2×3 ensemble)");

        while let Some(frame) = rx.blocking_recv() {
            total_frames += 1;

            // ── Route frame to Array A, B, or fallback ──────────────
            let assigned_to = if array_a.slot_for(frame.node_id).is_some() {
                array_a.push(&frame);
                'A'
            } else if array_b.slot_for(frame.node_id).is_some() {
                array_b.push(&frame);
                'B'
            } else {
                // More than 6 nodes? Use fallback single-node mode
                if fallback_buffer.is_none() {
                    fallback_buffer = Some(CsiWindowBuffer::new());
                    fallback_node_id = Some(frame.node_id);
                }
                if fallback_node_id == Some(frame.node_id) {
                    fallback_buffer.as_mut().unwrap().push(frame);
                }
                'F'
            };

            // Log node assignments once
            if total_frames == 1 || total_frames == 50 {
                info!(
                    "Node {} → Set {}  |  Array A slots: {:?}  Array B slots: {:?}",
                    frame.node_id, assigned_to,
                    array_a.node_slots, array_b.node_slots
                );
            }

            // ── Inference stride check ──────────────────────────────
            if total_frames % config.inference_stride != 0 {
                continue;
            }

            // ── Try inference on Array A ────────────────────────────
            if array_a.is_ready() {
                let raw_input = array_a.construct_input(device);
                total_infer_count += 1;

                // Diagnostics every 50 inferences
                if total_infer_count % 50 == 1 {
                    log_input_diagnostics("SetA", &raw_input);
                }

                match model.infer(&raw_input) {
                    Ok(kp_tensor) => {
                        if total_infer_count % 50 == 1 {
                            log_heatmap_diagnostics("SetA", &kp_tensor);
                        }
                        let kpts = spatial_soft_argmax(
                            kp_tensor, config.canvas_width, config.canvas_height);
                        if !kpts.is_empty() {
                            if total_infer_count % 100 == 1 {
                                log_keypoints("SetA", &kpts);
                            }
                            set_a_kpts = Some(kpts);
                        }
                    }
                    Err(e) => warn!("Set A inference failed: {}", e),
                }
            }

            // ── Try inference on Array B ────────────────────────────
            if array_b.is_ready() {
                let raw_input = array_b.construct_input(device);
                total_infer_count += 1;

                if total_infer_count % 50 == 1 {
                    log_input_diagnostics("SetB", &raw_input);
                }

                match model.infer(&raw_input) {
                    Ok(kp_tensor) => {
                        if total_infer_count % 50 == 1 {
                            log_heatmap_diagnostics("SetB", &kp_tensor);
                        }
                        let kpts = spatial_soft_argmax(
                            kp_tensor, config.canvas_width, config.canvas_height);
                        if !kpts.is_empty() {
                            if total_infer_count % 100 == 1 {
                                log_keypoints("SetB", &kpts);
                            }
                            set_b_kpts = Some(kpts);
                        }
                    }
                    Err(e) => warn!("Set B inference failed: {}", e),
                }
            }

            // ── Fallback: single-node with antenna replication ──────
            if set_a_kpts.is_none() && set_b_kpts.is_none() {
                if let Some(ref buf) = fallback_buffer {
                    if buf.is_ready() {
                        let raw_input = buf.construct_input(device);
                        if let Ok(kp_tensor) = model.infer(&raw_input) {
                            let kpts = spatial_soft_argmax(
                                kp_tensor, config.canvas_width, config.canvas_height);
                            if !kpts.is_empty() {
                                set_a_kpts = Some(kpts);
                            }
                        }
                    }
                }
            }

            // ── Ensemble: confidence-weighted merge of Set A + B ────
            let sources: Vec<&Vec<PoseKeypoint>> = [&set_a_kpts, &set_b_kpts]
                .iter()
                .filter_map(|opt| opt.as_ref())
                .collect();

            if sources.is_empty() {
                continue;
            }

            let n_sources = sources.len() as f64;
            let mut fused_kpts = Vec::with_capacity(17);

            for joint_idx in 0..17 {
                let mut sum_x = 0.0;
                let mut sum_y = 0.0;
                let mut sum_conf = 0.0;
                let mut name = String::new();

                for src in &sources {
                    if let Some(kp) = src.get(joint_idx) {
                        let w = kp.confidence;
                        sum_x += kp.x * w;
                        sum_y += kp.y * w;
                        sum_conf += w;
                        if name.is_empty() { name = kp.name.clone(); }
                    }
                }

                if sum_conf > 0.0 {
                    fused_kpts.push(PoseKeypoint {
                        name,
                        x: sum_x / sum_conf,
                        y: sum_y / sum_conf,
                        z: 0.0,
                        confidence: sum_conf / n_sources,
                    });
                }
            }

            // ── EMA temporal smoothing ──────────────────────────────
            match &mut ema_keypoints {
                Some(ema) => {
                    for (i, kp) in fused_kpts.iter_mut().enumerate() {
                        if let Some(prev) = ema.get_mut(i) {
                            prev.0 = EMA_ALPHA * kp.x + (1.0 - EMA_ALPHA) * prev.0;
                            prev.1 = EMA_ALPHA * kp.y + (1.0 - EMA_ALPHA) * prev.1;
                            kp.x = prev.0;
                            kp.y = prev.1;
                        }
                    }
                }
                None => {
                    ema_keypoints = Some(
                        fused_kpts.iter().map(|k| (k.x, k.y)).collect()
                    );
                }
            }

            // ── Broadcast fused detection ───────────────────────────
            if !fused_kpts.is_empty() {
                let min_x = fused_kpts.iter().map(|k| k.x).fold(f64::MAX, f64::min);
                let max_x = fused_kpts.iter().map(|k| k.x).fold(f64::MIN, f64::max);
                let min_y = fused_kpts.iter().map(|k| k.y).fold(f64::MAX, f64::min);
                let max_y = fused_kpts.iter().map(|k| k.y).fold(f64::MIN, f64::max);

                let n_active = array_a.active_slots() + array_b.active_slots();
                let detection = PersonDetection {
                    id: 1,
                    confidence: fused_kpts.iter()
                        .map(|k| k.confidence).sum::<f64>() / 17.0,
                    keypoints: fused_kpts,
                    bbox: BoundingBox {
                        x: min_x,
                        y: min_y,
                        width: (max_x - min_x).max(1.0),
                        height: (max_y - min_y).max(1.0),
                    },
                    zone: format!("array_{}nodes_{}sets",
                        n_active, sources.len()),
                };
                let _ = pose_tx.send(vec![detection]);
            }
        }
    });

    Ok((tx, pose_rx))
}

// ── Diagnostic helpers ──────────────────────────────────────────────────

fn log_input_diagnostics(label: &str, raw_input: &Tensor) {
    let amp_ch = raw_input.narrow(1, 0, 3);
    let amp_min  = amp_ch.min().double_value(&[]);
    let amp_max  = amp_ch.max().double_value(&[]);
    let amp_mean = amp_ch.mean(Kind::Float).double_value(&[]);

    let ph_ch  = raw_input.narrow(1, 3, 3);
    let ph_min  = ph_ch.min().double_value(&[]);
    let ph_max  = ph_ch.max().double_value(&[]);
    let ph_mean = ph_ch.mean(Kind::Float).double_value(&[]);

    info!(
        "{} input shape={:?}  \
         amp[min={:.3} max={:.3} mean={:.3}] (expect [0,1])  \
         phase[min={:.3} max={:.3} mean={:.3}] (expect [-1,1])",
        label, raw_input.size(),
        amp_min, amp_max, amp_mean,
        ph_min, ph_max, ph_mean
    );
}

fn log_heatmap_diagnostics(label: &str, kp_tensor: &Tensor) {
    let kp_min  = kp_tensor.min().double_value(&[]);
    let kp_max  = kp_tensor.max().double_value(&[]);
    let kp_mean = kp_tensor.mean(Kind::Float).double_value(&[]);
    info!(
        "{} heatmap shape={:?}  min={:.4} max={:.4} mean={:.4}",
        label, kp_tensor.size(), kp_min, kp_max, kp_mean
    );
}

fn log_keypoints(label: &str, kpts: &[PoseKeypoint]) {
    let coords: Vec<String> = kpts.iter().map(|k|
        format!("{}({:.3},{:.3} c={:.2})", k.name, k.x, k.y, k.confidence)
    ).collect();
    info!("{} keypoints: {}", label, coords.join(" "));
}

fn extract_keypoints(output: &tch::IValue) -> Option<Tensor> {
    match output {
        tch::IValue::Tensor(t) => Some(t.shallow_clone()),
        tch::IValue::GenericDict(items) => {
            for (k, v) in items {
                if let tch::IValue::String(name) = k {
                    if name == "keypoints" {
                        if let tch::IValue::Tensor(t) = v {
                            return Some(t.shallow_clone());
                        }
                    }
                }
            }
            None
        }
        tch::IValue::Tuple(items) => {
            if let Some(tch::IValue::Tensor(t)) = items.first() {
                return Some(t.shallow_clone());
            }
            None
        }
        _ => None,
    }
}

/// Decode keypoint heatmaps `[1, 17, H, W]` → normalized `(x, y)` coordinates.
///
/// Uses spatial soft-argmax (differentiable weighted-mean position):
///   x = Σ softmax(h) · col_grid  /  (W−1)  →  [0, 1]
///   y = Σ softmax(h) · row_grid  /  (H−1)  →  [0, 1]
///
/// Convention (matches MM-Fi training labels):
///   x increases left → right  (column index)
///   y increases top  → bottom (row index)
///
/// `canvas_w` and `canvas_h` are multiplied into the output; pass 1.0 to
/// keep coordinates normalized. The JS frontend scales them to pixels.
fn spatial_soft_argmax(keypoints: Tensor, canvas_w: f64, canvas_h: f64) -> Vec<PoseKeypoint> {
    const COCO_NAMES: &[&str] = &[
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    ];

    let size = keypoints.size();
    if size.len() < 4 { return vec![]; }

    let n_joints = size[1].min(17);
    let h = size[2];
    let w = size[3];
    let device = keypoints.device();

    let mut results = Vec::with_capacity(n_joints as usize);

    // col_grid[row*W + col] = col  (x direction)
    let x_grid = Tensor::arange(w, (Kind::Float, device))
        .view([1, w])
        .expand([h, w], false)
        .reshape([-1]);

    // row_grid[row*W + col] = row  (y direction)
    let y_grid = Tensor::arange(h, (Kind::Float, device))
        .view([h, 1])
        .expand([h, w], false)
        .reshape([-1]);

    for i in 0..n_joints {
        // heatmap: [H, W] — raw logits from the model
        let heatmap = keypoints.get(0).get(i);
        let raw_max = heatmap.max().double_value(&[]);

        // Softmax over the full HxW spatial map → probability distribution
        let flat_softmax = heatmap.reshape([-1]).softmax(0, Kind::Float);

        // Weighted-mean position (soft-argmax)
        let x_coord = (&flat_softmax * &x_grid).sum(Kind::Float).double_value(&[]);
        let y_coord = (&flat_softmax * &y_grid).sum(Kind::Float).double_value(&[]);

        // Normalize to [0, 1] using grid extent
        let x_norm = (x_coord / (w - 1).max(1) as f64).clamp(0.0, 1.0);
        let y_norm = (y_coord / (h - 1).max(1) as f64).clamp(0.0, 1.0);

        // Sigmoid of peak logit → confidence in [0, 1]
        let confidence = 1.0 / (1.0 + (-raw_max).exp());

        results.push(PoseKeypoint {
            name: COCO_NAMES.get(i as usize).unwrap_or(&"joint").to_string(),
            x: x_norm * canvas_w,
            y: y_norm * canvas_h,
            z: 0.0,
            confidence,
        });
    }
    results
}
