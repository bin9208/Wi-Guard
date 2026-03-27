//! Physics-based CSI Data Augmentation for WiFi pose estimation.
//!
//! Contains two layers:
//! - [`Xorshift64`]: lightweight PRNG, always available (no torch dependency).
//! - [`CsiAugmentor`]: GPU tensor augmentation, requires `torch-backend` feature.
//!
//! When `torch-backend` is enabled, [`CsiAugmentor`] operates directly on GPU
//! tensors in batch — no per-sample CPU loops. All augmentations model
//! real-world WiFi signal phenomena.
//!
//! Input tensors: `amp [B, T*ant, sub]`, `phase [B, T*ant, sub]`
//! where `T*ant = 300` (100 frames × 3 antennas), `sub = 56`.
//!
//! The 300-row dimension is interleaved as:
//! `[t0_a0, t0_a1, t0_a2, t1_a0, t1_a1, t1_a2, ..., t99_a0, t99_a1, t99_a2]`

// =========================================================================
// Xorshift64 PRNG — always available (no torch dependency)
// =========================================================================

/// Lightweight 64-bit Xorshift PRNG for deterministic augmentation.
pub struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    /// Create a new PRNG. Seed `0` is replaced with a fixed non-zero value.
    pub fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 0x853c49e6748fea9b } else { seed } }
    }

    /// Advance the state and return the next `u64`.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    /// Return a uniformly distributed `f32` in `[0, 1)`.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Return a uniformly distributed `f32` in `[lo, hi)`.
    #[inline]
    pub fn next_f32_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.next_f32() * (hi - lo)
    }

    /// Return a uniformly distributed `usize` in `[lo, hi]` (inclusive).
    #[inline]
    pub fn next_usize_range(&mut self, lo: usize, hi: usize) -> usize {
        if lo >= hi { return lo; }
        lo + (self.next_u64() % (hi - lo + 1) as u64) as usize
    }

    /// Sample an approximate Gaussian (mean=0, std=1) via Box-Muller.
    #[inline]
    pub fn next_gaussian(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-10);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

// =========================================================================
// CsiAugmentor — GPU tensor augmentation (requires torch-backend)
// =========================================================================

#[cfg(feature = "torch-backend")]
use torch::{Device, Kind, Tensor};

/// Physics-based CSI augmentation pipeline for WiFi pose estimation training.
///
/// All transforms run on the same device as the input tensors (GPU or CPU)
/// and operate on the full batch simultaneously.
#[cfg(feature = "torch-backend")]
pub struct CsiAugmentor {
    /// Number of antenna streams (tx × rx). Default: 3.
    n_ant: i64,
    /// Number of time frames per window. Default: 100.
    n_time: i64,
    /// Number of subcarriers. Default: 56.
    n_sub: i64,
    /// Device for random tensor generation (must match input tensors).
    device: Device,
}

#[cfg(feature = "torch-backend")]
impl CsiAugmentor {
    /// Create a new augmentor matching the dataset geometry.
    pub fn new(n_ant: i64, n_time: i64, n_sub: i64, device: Device) -> Self {
        Self { n_ant, n_time, n_sub, device }
    }

    /// Apply the full augmentation pipeline to a training batch.
    ///
    /// Each transform is applied independently with its own probability.
    /// The pipeline order is chosen so later transforms don't undo earlier ones:
    ///
    /// 1. Amplitude jitter (global scale per sample)
    /// 2. Phase drift (per-antenna constant offset)
    /// 3. Gaussian noise (additive)
    /// 4. Subcarrier band masking (frequency dropout)
    /// 5. Temporal masking (time-frame dropout)
    /// 6. Antenna dropout (full antenna channel zero-out)
    pub fn augment(
        &self,
        amp: &Tensor,   // [B, T*ant, sub]
        phase: &Tensor, // [B, T*ant, sub]
    ) -> (Tensor, Tensor) {
        let mut a = amp.shallow_clone();
        let mut p = phase.shallow_clone();

        // 1. Amplitude jitter: ×[0.7, 1.3] per sample  (p=0.5)
        //    Simulates varying transmitter–receiver distance / path-loss.
        a = self.amplitude_jitter(&a, 0.5, 0.7, 1.3);

        // 2. Phase drift: per-antenna random constant offset  (p=0.5)
        //    Simulates local-oscillator phase offset between antennas.
        p = self.phase_drift(&p, 0.5);

        // 3. Gaussian noise  (p=0.6, σ_amp=0.03, σ_phase=0.02)
        //    Simulates receiver thermal noise / quantisation noise.
        let (a2, p2) = self.gaussian_noise(&a, &p, 0.6, 0.03, 0.02);
        a = a2;
        p = p2;

        // 4. Subcarrier band masking  (p=0.5, width 1–8)
        //    Simulates frequency-selective deep fading.
        a = self.subcarrier_mask(&a, 0.5, 1, 8);
        p = self.subcarrier_mask(&p, 0.5, 1, 8);

        // 5. Temporal masking  (p=0.5, width 1–15 frames)
        //    Simulates transient occlusion / signal dropout.
        let (a3, p3) = self.temporal_mask(&a, &p, 0.5, 1, 15);
        a = a3;
        p = p3;

        // 6. Antenna dropout  (p=0.15)
        //    Simulates antenna hardware failure / severe antenna-level fading.
        let (a4, p4) = self.antenna_dropout(&a, &p, 0.15);
        a = a4;
        p = p4;

        (a, p)
    }

    // -----------------------------------------------------------------------
    // Individual transforms
    // -----------------------------------------------------------------------

    /// Per-sample amplitude scaling by a random factor in [lo, hi].
    fn amplitude_jitter(&self, amp: &Tensor, prob: f64, lo: f64, hi: f64) -> Tensor {
        let b = amp.size()[0];
        // Coin flip per sample: [B]
        let coin = Tensor::rand([b], (Kind::Float, self.device));
        // Random scale in [lo, hi], shape [B, 1, 1] for broadcast
        let scale = Tensor::rand([b, 1, 1], (Kind::Float, self.device)) * (hi - lo) + lo;
        // Where coin < prob, apply scale; otherwise keep original
        let mask = coin.lt(prob).unsqueeze(-1).unsqueeze(-1).to_kind(Kind::Float);
        let scaled = amp * &scale;
        // mask * scaled + (1 - mask) * amp
        &scaled * &mask + amp * (Tensor::ones_like(&mask) - &mask)
    }

    /// Per-antenna random phase offset (LO drift simulation).
    ///
    /// Adds a constant offset ∈ [-0.3, 0.3] (in normalised phase units, ≈ [-54°, 54°])
    /// to each antenna stream independently.
    fn phase_drift(&self, phase: &Tensor, prob: f64) -> Tensor {
        let b = phase.size()[0];
        let coin: f64 = Tensor::rand([], (Kind::Float, Device::Cpu)).double_value(&[]);
        if coin >= prob {
            return phase.shallow_clone();
        }
        // Generate per-antenna offsets: [B, ant]
        let offsets = Tensor::rand([b, self.n_ant], (Kind::Float, self.device)) * 0.6 - 0.3;
        // Expand to [B, T*ant, 1]: repeat each antenna offset T times
        // offsets [B, ant] → [B, 1, ant] → repeat [B, T, ant] → reshape [B, T*ant] → unsqueeze
        let offsets = offsets
            .unsqueeze(1)
            .expand([b, self.n_time, self.n_ant], false)
            .contiguous()
            .reshape([b, self.n_time * self.n_ant])
            .unsqueeze(-1); // [B, T*ant, 1]
        phase + offsets
    }

    /// Additive Gaussian noise on both amplitude and phase.
    fn gaussian_noise(
        &self,
        amp: &Tensor,
        phase: &Tensor,
        prob: f64,
        sigma_amp: f64,
        sigma_phase: f64,
    ) -> (Tensor, Tensor) {
        let coin: f64 = Tensor::rand([], (Kind::Float, Device::Cpu)).double_value(&[]);
        if coin >= prob {
            return (amp.shallow_clone(), phase.shallow_clone());
        }
        let noise_a = Tensor::randn(amp.size().as_slice(), (Kind::Float, self.device)) * sigma_amp;
        let noise_p = Tensor::randn(phase.size().as_slice(), (Kind::Float, self.device)) * sigma_phase;
        (amp + noise_a, phase + noise_p)
    }

    /// Zero out a random contiguous band of subcarriers (SpecAugment-style).
    ///
    /// Width is uniform in [min_w, max_w]. The same band is masked for every
    /// sample in the batch (simulates environment-level frequency fading).
    fn subcarrier_mask(&self, x: &Tensor, prob: f64, min_w: i64, max_w: i64) -> Tensor {
        let coin: f64 = Tensor::rand([], (Kind::Float, Device::Cpu)).double_value(&[]);
        if coin >= prob {
            return x.shallow_clone();
        }
        let w = Tensor::randint_low(min_w, max_w + 1, [], (Kind::Int64, Device::Cpu))
            .int64_value(&[]);
        let max_start = (self.n_sub - w).max(0);
        let start = if max_start > 0 {
            Tensor::randint(max_start, [], (Kind::Int64, Device::Cpu)).int64_value(&[])
        } else {
            0
        };

        // Build mask: ones everywhere except the masked band
        let mut mask = Tensor::ones([1, 1, self.n_sub], (Kind::Float, self.device));
        if w > 0 {
            let _ = mask.narrow(2, start, w).fill_(0.0);
        }
        x * mask // broadcast [1,1,sub] over [B, T*ant, sub]
    }

    /// Zero out contiguous time frames (all antennas for those frames).
    ///
    /// Width is uniform in [min_w, max_w] frames. Affects all antennas
    /// for the selected frames, preserving inter-antenna consistency.
    fn temporal_mask(
        &self,
        amp: &Tensor,
        phase: &Tensor,
        prob: f64,
        min_w: i64,
        max_w: i64,
    ) -> (Tensor, Tensor) {
        let coin: f64 = Tensor::rand([], (Kind::Float, Device::Cpu)).double_value(&[]);
        if coin >= prob {
            return (amp.shallow_clone(), phase.shallow_clone());
        }
        let b = amp.size()[0];
        let flat = self.n_time * self.n_ant;

        let w_frames = Tensor::randint_low(min_w, max_w + 1, [], (Kind::Int64, Device::Cpu))
            .int64_value(&[]);
        let max_start = (self.n_time - w_frames).max(0);
        let start_frame = if max_start > 0 {
            Tensor::randint(max_start, [], (Kind::Int64, Device::Cpu)).int64_value(&[])
        } else {
            0
        };

        // Build mask in [1, T, ant, 1] then reshape to [1, T*ant, 1]
        let mut mask_4d = Tensor::ones(
            [1, self.n_time, self.n_ant, 1],
            (Kind::Float, self.device),
        );
        if w_frames > 0 {
            let _ = mask_4d.narrow(1, start_frame, w_frames).fill_(0.0);
        }
        let mask = mask_4d.reshape([1, flat, 1]); // broadcast over [B, T*ant, sub]

        (amp * &mask, phase * &mask)
    }

    /// Zero out one entire antenna stream across all time frames.
    ///
    /// Simulates antenna hardware failure or severe per-antenna fading.
    /// At most one antenna is dropped per batch to avoid information loss.
    fn antenna_dropout(
        &self,
        amp: &Tensor,
        phase: &Tensor,
        prob: f64,
    ) -> (Tensor, Tensor) {
        let coin: f64 = Tensor::rand([], (Kind::Float, Device::Cpu)).double_value(&[]);
        if coin >= prob {
            return (amp.shallow_clone(), phase.shallow_clone());
        }
        let flat = self.n_time * self.n_ant;
        // Pick which antenna to drop: 0, 1, or 2
        let drop_ant = Tensor::randint(self.n_ant, [], (Kind::Int64, Device::Cpu))
            .int64_value(&[]);

        // Build mask in [1, T, ant, 1]
        let mut mask_4d = Tensor::ones(
            [1, self.n_time, self.n_ant, 1],
            (Kind::Float, self.device),
        );
        // Zero out the selected antenna across all time steps
        // mask_4d[:, :, drop_ant, :] = 0
        let _ = mask_4d
            .narrow(2, drop_ant, 1)
            .fill_(0.0);
        let mask = mask_4d.reshape([1, flat, 1]);

        (amp * &mask, phase * &mask)
    }
}

/// Legacy type alias for backward compatibility.
#[cfg(feature = "torch-backend")]
pub type VirtualDomainAugmentor = CsiAugmentor;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[cfg(feature = "torch-backend")]
mod tests {
    use super::*;

    fn device() -> Device { Device::Cpu }

    fn make_augmentor() -> CsiAugmentor {
        CsiAugmentor::new(3, 100, 56, device())
    }

    fn dummy_batch(b: i64) -> (Tensor, Tensor) {
        let amp = Tensor::rand([b, 300, 56], (Kind::Float, device()));
        let phase = Tensor::rand([b, 300, 56], (Kind::Float, device())) * 2.0 - 1.0;
        (amp, phase)
    }

    #[test]
    fn augment_preserves_shape() {
        let aug = make_augmentor();
        let (amp, phase) = dummy_batch(4);
        let (a_out, p_out) = aug.augment(&amp, &phase);
        assert_eq!(a_out.size(), vec![4, 300, 56]);
        assert_eq!(p_out.size(), vec![4, 300, 56]);
    }

    #[test]
    fn augment_produces_finite_values() {
        let aug = make_augmentor();
        let (amp, phase) = dummy_batch(8);
        let (a_out, p_out) = aug.augment(&amp, &phase);
        assert!(a_out.isfinite().all().int64_value(&[]) == 1, "amp must be finite");
        assert!(p_out.isfinite().all().int64_value(&[]) == 1, "phase must be finite");
    }

    #[test]
    fn subcarrier_mask_zeroes_band() {
        let aug = make_augmentor();
        let x = Tensor::ones([2, 300, 56], (Kind::Float, device()));
        // Run many times; at least one should have zeros
        let mut found_zeros = false;
        for _ in 0..20 {
            let out = aug.subcarrier_mask(&x, 1.0, 4, 8);
            let min_val = out.min().double_value(&[]);
            if min_val < 0.5 {
                found_zeros = true;
                break;
            }
        }
        assert!(found_zeros, "subcarrier mask should zero out some subcarriers");
    }

    #[test]
    fn temporal_mask_zeroes_frames() {
        let aug = make_augmentor();
        let amp = Tensor::ones([2, 300, 56], (Kind::Float, device()));
        let phase = Tensor::ones([2, 300, 56], (Kind::Float, device()));
        let mut found_zeros = false;
        for _ in 0..20 {
            let (a_out, _) = aug.temporal_mask(&amp, &phase, 1.0, 5, 15);
            let min_val = a_out.min().double_value(&[]);
            if min_val < 0.5 {
                found_zeros = true;
                break;
            }
        }
        assert!(found_zeros, "temporal mask should zero out some frames");
    }

    #[test]
    fn antenna_dropout_zeroes_one_antenna() {
        let aug = make_augmentor();
        let amp = Tensor::ones([2, 300, 56], (Kind::Float, device()));
        let phase = Tensor::ones([2, 300, 56], (Kind::Float, device()));
        let (a_out, _) = aug.antenna_dropout(&amp, &phase, 1.0); // force apply
        // Reshape to [2, 100, 3, 56] and check exactly one antenna is zeroed
        let reshaped = a_out.reshape([2, 100, 3, 56]);
        let per_ant_sum = reshaped.sum_dim_intlist([1_i64, 3].as_slice(), false, Kind::Float);
        // per_ant_sum: [2, 3] — one of the 3 should be 0
        let min_per_sample = per_ant_sum.min_dim(1, false).0;
        for i in 0..2 {
            assert!(
                min_per_sample.double_value(&[i]) < 1e-6,
                "one antenna should be fully zeroed"
            );
        }
    }

    #[test]
    fn amplitude_jitter_changes_values() {
        let aug = make_augmentor();
        let amp = Tensor::ones([4, 300, 56], (Kind::Float, device()));
        let mut different = false;
        for _ in 0..10 {
            let out = aug.amplitude_jitter(&amp, 1.0, 0.7, 1.3);
            let diff = (&out - &amp).abs().sum(Kind::Float).double_value(&[]);
            if diff > 1e-3 {
                different = true;
                break;
            }
        }
        assert!(different, "amplitude jitter should modify values");
    }

    #[test]
    fn phase_drift_adds_per_antenna_offset() {
        let aug = make_augmentor();
        let phase = Tensor::zeros([2, 300, 56], (Kind::Float, device()));
        let out = aug.phase_drift(&phase, 1.0); // force apply
        // Reshape to [2, 100, 3, 56]
        let reshaped = out.reshape([2, 100, 3, 56]);
        // Each antenna should have a constant offset across time and subcarriers
        // Check that within one antenna, the std across time is very low
        let ant0 = reshaped.select(2, 0); // [2, 100, 56]
        let ant0_std = ant0.std(true).double_value(&[]);
        // Since input was 0 and we add a constant per antenna, std should be ~0
        // (or very small from the per-sample variation)
        assert!(ant0_std < 0.4, "per-antenna offset should be roughly constant, got std={ant0_std}");
    }

    #[test]
    fn gaussian_noise_increases_variance() {
        let aug = make_augmentor();
        let amp = Tensor::ones([4, 300, 56], (Kind::Float, device()));
        let phase = Tensor::zeros([4, 300, 56], (Kind::Float, device()));
        let (a_out, p_out) = aug.gaussian_noise(&amp, &phase, 1.0, 0.03, 0.02);
        let amp_std = a_out.std(true).double_value(&[]);
        let ph_std = p_out.std(true).double_value(&[]);
        assert!(amp_std > 0.01, "noise should increase amp variance, got {amp_std}");
        assert!(ph_std > 0.005, "noise should increase phase variance, got {ph_std}");
    }

}

#[cfg(test)]
mod tests_rng {
    use super::*;

    #[test]
    fn xorshift64_deterministic() {
        let mut a = Xorshift64::new(999);
        let mut b = Xorshift64::new(999);
        for _ in 0..100 { assert_eq!(a.next_u64(), b.next_u64()); }
    }

    #[test]
    fn xorshift64_f32_in_unit_interval() {
        let mut rng = Xorshift64::new(42);
        for _ in 0..1000 {
            let v = rng.next_f32();
            assert!(v >= 0.0 && v < 1.0, "f32 sample {v} not in [0, 1)");
        }
    }
}
