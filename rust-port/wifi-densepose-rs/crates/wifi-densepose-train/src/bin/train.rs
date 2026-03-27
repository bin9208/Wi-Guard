//! `train` binary — entry point for the WiFi-DensePose training pipeline.
//!
//! # Usage
//!
//! ```bash
//! # Full training with default config (requires torch-backend feature)
//! cargo run --features torch-backend --bin train
//!
//! # Custom config and data directory
//! cargo run --features torch-backend --bin train -- \
//!     --config config.json --data-dir /data/mm-fi
//!
//! # GPU training
//! cargo run --features torch-backend --bin train -- --cuda
//!
//! # Smoke-test with synthetic data (no real dataset required)
//! cargo run --features torch-backend --bin train -- --dry-run
//! ```
//!
//! Exit code 0 on success, non-zero on configuration or dataset errors.
//!
//! **Note**: This binary requires the `torch-backend` Cargo feature to be
//! enabled. When the feature is disabled a stub `main` is compiled that
//! immediately exits with a helpful error message.

use clap::Parser;
use std::path::PathBuf;
use tracing::{error, info, warn};

use wifi_densepose_train::{
    config::TrainingConfig,
    dataset::{CsiDataset, MmFiDataset, SyntheticCsiDataset, SyntheticConfig},
};

// ---------------------------------------------------------------------------
// CLI arguments
// ---------------------------------------------------------------------------

/// Command-line arguments for the WiFi-DensePose training binary.
#[derive(Parser, Debug)]
#[command(
    name = "train",
    version,
    about = "Train WiFi-DensePose on the MM-Fi dataset",
    long_about = None
)]
struct Args {
    /// Path to a JSON training-configuration file.
    ///
    /// If not provided, [`TrainingConfig::default`] is used.
    #[arg(short, long, value_name = "FILE")]
    config: Option<PathBuf>,

    /// Root directory containing MM-Fi recordings (alias for --data-dir).
    #[arg(long, value_name = "DIR")]
    dataset: Option<PathBuf>,

    /// Root directory containing MM-Fi recordings.
    #[arg(long, value_name = "DIR")]
    data_dir: Option<PathBuf>,

    /// Dataset format (currently only "mmfi" is supported).
    #[arg(long, default_value = "mmfi")]
    dataset_type: String,

    /// Train/Val split strategy: "random" or "subject" (Leave-Subjects-Out).
    #[arg(long, default_value = "random")]
    split_mode: String,

    /// Override the number of training epochs.
    #[arg(long, value_name = "N")]
    epochs: Option<usize>,

    /// Override the number of warmup epochs.
    #[arg(long, value_name = "N")]
    warmup_epochs: Option<usize>,

    /// Override the learning rate milestones (e.g. "30,45").
    #[arg(long, value_name = "CSV", value_delimiter = ',')]
    lr_milestones: Option<Vec<usize>>,

    /// Override the mini-batch size.
    #[arg(long, value_name = "N")]
    batch_size: Option<usize>,

    /// Override the initial learning rate.
    #[arg(long, value_name = "LR")]
    lr: Option<f64>,

    /// Override the checkpoint output directory from the config.
    #[arg(long, value_name = "DIR")]
    checkpoint_dir: Option<PathBuf>,

    /// Enable CUDA training (sets `use_gpu = true` in the config).
    #[arg(long, default_value_t = false)]
    cuda: bool,

    /// Override the GPU device ID (default: 0).
    #[arg(long, default_value_t = 0)]
    gpu_device_id: i64,

    /// Run a smoke-test with a synthetic dataset instead of real MM-Fi data.
    ///
    /// Useful for verifying the pipeline without downloading the dataset.
    #[arg(long, default_value_t = false)]
    dry_run: bool,

    /// Number of synthetic samples when `--dry-run` is active.
    #[arg(long, default_value_t = 64)]
    dry_run_samples: usize,

    /// Log level: trace, debug, info, warn, error.
    #[arg(long, default_value = "info")]
    log_level: String,

    /// Resume training from a saved checkpoint (.pt file).
    #[arg(long, value_name = "FILE")]
    resume: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let args = Args::parse();

    // Initialise structured logging.
    tracing_subscriber::fmt()
        .with_max_level(
            args.log_level
                .parse::<tracing_subscriber::filter::LevelFilter>()
                .unwrap_or(tracing_subscriber::filter::LevelFilter::INFO),
        )
        .with_target(false)
        .with_thread_ids(false)
        .init();

    info!(
        "WiFi-DensePose Training Pipeline v{}",
        wifi_densepose_train::VERSION
    );

    // ------------------------------------------------------------------
    // Build TrainingConfig
    // ------------------------------------------------------------------

    let mut config = if let Some(ref cfg_path) = args.config {
        info!("Loading configuration from {}", cfg_path.display());
        match TrainingConfig::from_json(cfg_path) {
            Ok(c) => c,
            Err(e) => {
                error!("Failed to load config: {e}");
                std::process::exit(1);
            }
        }
    } else {
        info!("No config file provided — using TrainingConfig::default()");
        TrainingConfig::default()
    };

    // Apply CLI overrides.
    if let Some(dir) = args.checkpoint_dir {
        info!("Overriding checkpoint_dir → {}", dir.display());
        config.checkpoint_dir = dir;
    }
    if args.cuda {
        #[cfg(feature = "torch-backend")]
        {
            if torch::Cuda::is_available() {
                info!(
                    "CUDA override: use_gpu = true (using GPU device {})",
                    args.gpu_device_id
                );
                config.use_gpu = true;
                config.gpu_device_id = args.gpu_device_id;
            } else {
                warn!("CUDA override requested (--cuda), but backend does not support it.");
                warn!("Ensure Libtorch with CUDA is installed and torch-rs can find it.");
                config.use_gpu = false;
            }
        }
        #[cfg(not(feature = "torch-backend"))]
        {
            warn!("CUDA override requested (--cuda), but torch-backend feature is disabled.");
            config.use_gpu = false;
        }
    }

    if let Some(e) = args.epochs {
        info!("Overriding num_epochs → {}", e);
        config.num_epochs = e;
    }
    if let Some(w) = args.warmup_epochs {
        info!("Overriding warmup_epochs → {}", w);
        config.warmup_epochs = w;
    }
    if let Some(m) = args.lr_milestones {
        info!("Overriding lr_milestones → {:?}", m);
        config.lr_milestones = m;
    }
    if let Some(bs) = args.batch_size {
        info!("Overriding batch_size → {}", bs);
        config.batch_size = bs;
    }
    if let Some(lr) = args.lr {
        info!("Overriding learning_rate → {:.2e}", lr);
        config.learning_rate = lr;
    }

    // Validate the final configuration.
    if let Err(e) = config.validate() {
        error!("Config validation failed: {e}");
        std::process::exit(1);
    }

    log_config_summary(&config);

    // ------------------------------------------------------------------
    // Build datasets
    // ------------------------------------------------------------------

    let data_dir = args
        .data_dir
        .clone()
        .or(args.dataset.clone())
        .unwrap_or_else(|| PathBuf::from("data/mm-fi"));

    if args.dry_run {
        info!(
            "DRY RUN: using SyntheticCsiDataset ({} samples)",
            args.dry_run_samples
        );
        let syn_cfg = SyntheticConfig {
            num_subcarriers: config.num_subcarriers,
            num_antennas_tx: config.num_antennas_tx,
            num_antennas_rx: config.num_antennas_rx,
            window_frames: config.window_frames,
            num_keypoints: config.num_keypoints,
            signal_frequency_hz: 2.4e9,
        };
        let n_total = args.dry_run_samples;
        let n_val = (n_total / 5).max(1);
        let n_train = n_total - n_val;
        let train_ds = SyntheticCsiDataset::new(n_train, syn_cfg.clone());
        let val_ds = SyntheticCsiDataset::new(n_val, syn_cfg);

        info!(
            "Synthetic split: {} train / {} val",
            train_ds.len(),
            val_ds.len()
        );

        run_training(config, &train_ds, &val_ds, args.resume);
    } else {
        info!("Loading MM-Fi dataset from {}", data_dir.display());

        let train_ds = match MmFiDataset::discover(
            &data_dir,
            config.window_frames,
            config.num_subcarriers,
            config.num_keypoints,
        ) {
            Ok(ds) => ds,
            Err(e) => {
                error!("Failed to load dataset: {e}");
                error!(
                    "Ensure MM-Fi data exists at {}",
                    data_dir.display()
                );
                std::process::exit(1);
            }
        };

        if train_ds.is_empty() {
            error!(
                "Dataset is empty — no samples found in {}",
                data_dir.display()
            );
            std::process::exit(1);
        }

        info!("Dataset: {} samples", train_ds.len());

        let (train_ds_split, val_ds_split) = match args.split_mode.to_lowercase().as_str() {
            "subject" => {
                info!("Splitting dataset by Subject IDs (80/20 Environment-Balanced strategy)");
                wifi_densepose_train::dataset::DatasetSubset::split_by_subject_balanced(&train_ds, 0.8, config.seed)
            }
            _ => {
                info!("Splitting dataset randomly (80/20 strategy)");
                wifi_densepose_train::dataset::DatasetSubset::split(&train_ds, 0.8, config.seed)
            }
        };

        info!(
            "Dataset split ({:.1}/{:.1}): {} train / {} val samples",
            0.8 * 100.0,
            0.2 * 100.0,
            train_ds_split.len(),
            val_ds_split.len()
        );

        run_training(config, &train_ds_split, &val_ds_split, args.resume);
    }
}

// ---------------------------------------------------------------------------
// run_training — conditionally compiled on torch-backend
// ---------------------------------------------------------------------------

#[cfg(feature = "torch-backend")]
fn run_training(
    config: TrainingConfig,
    train_ds: &dyn CsiDataset,
    val_ds: &dyn CsiDataset,
    resume_path: Option<PathBuf>,
) {
    use wifi_densepose_train::trainer::Trainer;

    println!("Checking environment...");
    println!("CUDA available: {}", torch::Cuda::is_available());
    println!("CUDA device count: {}", torch::Cuda::device_count());

    info!(
        "Starting training: {} train / {} val samples",
        train_ds.len(),
        val_ds.len()
    );

    let mut trainer = Trainer::new(config);
    let mut start_epoch = 0;
    
    if let Some(path) = resume_path {
        info!("Resuming from checkpoint: {}", path.display());
        match trainer.load_checkpoint(&path) {
            Ok(epoch) => {
                start_epoch = epoch;
                info!("Successfully loaded weights and detected epoch {start_epoch}.");
            }
            Err(e) => {
                error!("Failed to resume from {}: {e}", path.display());
                std::process::exit(1);
            }
        }
    }

    match trainer.train(train_ds, val_ds, start_epoch, 0.0) {
        Ok(result) => {
            info!("Training complete.");
            info!("  Best PCK@0.2 : {:.4}", result.best_pck);
            info!("  Best epoch   : {}", result.best_epoch);
            info!("  Final loss   : {:.6}", result.final_train_loss);
            if let Some(ref ckpt) = result.checkpoint_path {
                info!("  Best checkpoint: {}", ckpt.display());
            }
        }
        Err(e) => {
            error!("Training failed: {e}");
            std::process::exit(1);
        }
    }
}

#[cfg(not(feature = "torch-backend"))]
fn run_training(
    _config: TrainingConfig,
    train_ds: &dyn CsiDataset,
    val_ds: &dyn CsiDataset,
    _resume_path: Option<std::path::PathBuf>,
) {
    info!(
        "Pipeline verification complete: {} train / {} val samples loaded.",
        train_ds.len(),
        val_ds.len()
    );
    info!(
        "Full training requires the `torch-backend` feature: \
         cargo run --features torch-backend --bin train"
    );
    info!("Config and dataset infrastructure: OK");
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Log a human-readable summary of the active training configuration.
fn log_config_summary(config: &TrainingConfig) {
    info!("Training configuration:");
    info!("  subcarriers  : {} (native: {})", config.num_subcarriers, config.native_subcarriers);
    info!("  antennas     : {}×{}", config.num_antennas_tx, config.num_antennas_rx);
    info!("  window frames: {}", config.window_frames);
    info!("  batch size   : {}", config.batch_size);
    info!("  learning rate: {:.2e}", config.learning_rate);
    info!("  epochs       : {}", config.num_epochs);
    info!("  device       : {}", if config.use_gpu { "GPU" } else { "CPU" });
    info!("  checkpoint   : {}", config.checkpoint_dir.display());
}
