//! Real-time GPU inference bridge for WiFi-DensePose.

use std::collections::VecDeque;
use std::fs;
use std::path::PathBuf;
use serde::Deserialize;
use tokio::sync::{mpsc, watch};
use tch::{Device, Tensor, CModule, Kind};
use tracing::{info, error, warn};

use super::{PersonDetection, PoseKeypoint, BoundingBox};

pub const N_SUB: usize = 56;
pub const N_ANTENNAS: usize = 3;
pub const WINDOW_SIZE: usize = 100;
pub const N_CHANNELS: usize = 6;

/// Stack-allocated CSI frame data for inference.
#[derive(Debug, Clone, Copy)]
pub struct CsiFrameSlim {
    pub data: [f32; 336],
}

impl CsiFrameSlim {
    pub fn from_esp32(amplitudes: &[f64], phases: &[f64]) -> Self {
        let mut data = [0.0f32; 336];
        for (i, &amp) in amplitudes.iter().take(168).enumerate() {
            data[i] = (20.0 * (amp + 1e-6).log10()) as f32;
        }
        for (i, &phase) in phases.iter().take(168).enumerate() {
            data[i + 168] = phase as f32;
        }
        Self { data }
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

pub fn spawn_inference_pipeline(
    config: InferencePipelineConfig,
) -> Result<(mpsc::Sender<CsiFrameSlim>, watch::Receiver<Vec<PersonDetection>>), Box<dyn std::error::Error>> {
    let device = Device::cuda_if_available();
    info!("Starting CSI Inference Pipeline on {:?}", device);
    
    let model = CModule::load_on_device(&config.checkpoint_path, device)?;
    
    let zscore: ZScoreStats = fs::read_to_string(&config.zscore_path)
        .and_then(|s| serde_json::from_str(&s).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e)))
        .unwrap_or_else(|_| {
            warn!("Z-score stats not found at {:?}, using identity", config.zscore_path);
            ZScoreStats { mean: vec![0.0; 6], std: vec![1.0; 6] }
        });
        
    let mean_t = Tensor::from_slice(&zscore.mean).to_device(device).view([1, 6, 1, 1]);
    let std_t = Tensor::from_slice(&zscore.std).to_device(device).view([1, 6, 1, 1]);
    
    let (tx, mut rx) = mpsc::channel::<CsiFrameSlim>(100);
    let (pose_tx, pose_rx) = watch::channel(vec![]);
    
    std::thread::spawn(move || {
        let mut buffer = CsiWindowBuffer::new();
        let mut frame_count = 0;
        
        while let Some(frame) = rx.blocking_recv() {
            buffer.push(frame);
            frame_count += 1;
            
            if frame_count % config.inference_stride == 0 && buffer.is_ready() {
                let input = buffer.construct_input(device);
                let input_norm = (input - &mean_t) / (&std_t + 1e-6);
                
                let amp_t = input_norm.narrow(1, 0, 3).contiguous();
                let phase_t = input_norm.narrow(1, 3, 3).contiguous();
                
                let ivalue_args = [tch::IValue::Tensor(amp_t), tch::IValue::Tensor(phase_t)];
                
                match model.forward_is(&ivalue_args) {
                    Ok(output) => {
                        if let Some(keypoints_tensor) = extract_keypoints(&output) {
                            let pose_kpts = spatial_soft_argmax(keypoints_tensor, config.canvas_width, config.canvas_height);
                            
                            if !pose_kpts.is_empty() {
                                let min_x = pose_kpts.iter().map(|k| k.x).fold(f64::MAX, f64::min);
                                let max_x = pose_kpts.iter().map(|k| k.x).fold(f64::MIN, f64::max);
                                let min_y = pose_kpts.iter().map(|k| k.y).fold(f64::MAX, f64::min);
                                let max_y = pose_kpts.iter().map(|k| k.y).fold(f64::MIN, f64::max);

                                let detection = PersonDetection {
                                    id: 1,
                                    confidence: pose_kpts.iter().map(|k| k.confidence).sum::<f64>() / 17.0,
                                    keypoints: pose_kpts,
                                    bbox: BoundingBox {
                                        x: min_x,
                                        y: min_y,
                                        width: (max_x - min_x).max(1.0),
                                        height: (max_y - min_y).max(1.0),
                                    },
                                    zone: "detected".to_string(),
                                };
                                let _ = pose_tx.send(vec![detection]);
                            }
                        }
                    }
                    Err(e) => {
                        warn!("Model inference failed: {}", e);
                        let _ = pose_tx.send(vec![]);
                    }
                }
            }
        }
    });
    
    Ok((tx, pose_rx))
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
    
    let x_grid = Tensor::arange(w, (Kind::Float, device)).view([1, w]).repeat([h, 1]).view([-1]);
    let y_grid = Tensor::arange(h, (Kind::Float, device)).view([h, 1]).repeat([1, w]).view([-1]);

    for i in 0..n_joints {
        let heatmap = keypoints.get(0).get(i);
        let flat_softmax = heatmap.view([-1]).softmax(0, Kind::Float);
        
        let x_coord = (&flat_softmax * &x_grid).sum(Kind::Float).double_value(&[]);
        let y_coord = (&flat_softmax * &y_grid).sum(Kind::Float).double_value(&[]);
        
        let x_norm = x_coord / (w - 1).max(1) as f64;
        let y_norm = y_coord / (h - 1).max(1) as f64;
        
        results.push(PoseKeypoint {
            name: COCO_NAMES.get(i as usize).unwrap_or(&"joint").to_string(),
            x: x_norm * canvas_w,
            y: y_norm * canvas_h,
            z: 0.0,
            confidence: heatmap.max().double_value(&[]),
        });
    }
    results
}
