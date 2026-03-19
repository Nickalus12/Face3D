use std::sync::Arc;
use std::time::Instant;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

// ── Data types shared across commands ────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub name: String,
    pub has_gaussians: bool,
    pub has_mesh: bool,
    pub has_renders: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_size_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub name: String,
    pub memory_total: String,
    pub memory_used: String,
    pub utilization: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineLog {
    pub line: String,
    pub level: String,
    pub timestamp: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStageChange {
    pub stage: u32,
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineMetric {
    pub iter: u64,
    pub loss: f64,
    pub psnr: f64,
    pub gaussians: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineComplete {
    pub exit_code: i32,
    pub duration_secs: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreviewFile {
    pub name: String,
    pub preview_type: String,
    pub path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythonInfo {
    pub version: String,
    pub env_name: String,
    pub packages_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub os_version: String,
    pub cpu: String,
    pub ram_gb: f64,
    pub disk_free_gb: f64,
    pub disk_total_gb: f64,
    pub cuda_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorInfo {
    pub name: String,
    pub samples: u64,
    pub duration: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorSummary {
    pub device_name: String,
    pub recording_time: String,
    pub duration: f64,
    pub sensors: Vec<SensorInfo>,
}

// ── Pipeline process state ───────────────────────────────────────

/// Tracks a running pipeline child process.
pub struct PipelineState {
    /// PID of the child process (None when idle).
    pub pid: Option<u32>,
    /// When the pipeline was started.
    pub started_at: Option<Instant>,
    /// Current stage number parsed from stdout.
    pub current_stage: Option<u32>,
    /// Current stage name parsed from stdout.
    pub current_stage_name: Option<String>,
}

impl PipelineState {
    pub fn new() -> Self {
        Self {
            pid: None,
            started_at: None,
            current_stage: None,
            current_stage_name: None,
        }
    }

    pub fn is_running(&self) -> bool {
        self.pid.is_some()
    }

    #[allow(dead_code)]
    pub fn elapsed_secs(&self) -> f64 {
        self.started_at
            .map(|t| t.elapsed().as_secs_f64())
            .unwrap_or(0.0)
    }

    pub fn reset(&mut self) {
        self.pid = None;
        self.started_at = None;
        self.current_stage = None;
        self.current_stage_name = None;
    }
}

/// Thread-safe wrapper around pipeline state.
pub struct PipelineProcess(pub Arc<Mutex<PipelineState>>);

impl PipelineProcess {
    pub fn new() -> Self {
        Self(Arc::new(Mutex::new(PipelineState::new())))
    }
}

// ── GPU cache state ──────────────────────────────────────────────

/// Caches GPU info for a configurable TTL to avoid hammering nvidia-smi.
pub struct GpuCache {
    pub info: Option<GpuInfo>,
    pub last_fetch: Option<Instant>,
}

impl GpuCache {
    pub fn new() -> Self {
        Self {
            info: None,
            last_fetch: None,
        }
    }

    /// Returns cached info if still valid (within `ttl_secs`).
    pub fn get(&self, ttl_secs: f64) -> Option<&GpuInfo> {
        match (&self.info, self.last_fetch) {
            (Some(info), Some(fetched)) if fetched.elapsed().as_secs_f64() < ttl_secs => {
                Some(info)
            }
            _ => None,
        }
    }

    /// Store new GPU info with current timestamp.
    pub fn set(&mut self, info: GpuInfo) {
        self.info = Some(info);
        self.last_fetch = Some(Instant::now());
    }
}

/// Thread-safe wrapper around GPU cache.
pub struct GpuCacheState(pub Arc<Mutex<GpuCache>>);

impl GpuCacheState {
    pub fn new() -> Self {
        Self(Arc::new(Mutex::new(GpuCache::new())))
    }
}

// ── Constants ────────────────────────────────────────────────────

pub const PROJECT_ROOT: &str = "D:/Projects/3D";
pub const PYTHON_EXE: &str = "C:/Users/Nicka/miniconda3/envs/face3d/python.exe";
pub const CUDA_HOME: &str = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8";
