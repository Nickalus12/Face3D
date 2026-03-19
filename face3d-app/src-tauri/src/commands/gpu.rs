use std::process::Command;

use tauri::{command, State};

use crate::error::{AppError, AppResult};
use crate::state::{GpuCacheState, GpuInfo};

/// Cache TTL in seconds — avoid hammering nvidia-smi.
const GPU_CACHE_TTL_SECS: f64 = 2.0;

#[command]
pub async fn get_gpu_info(cache: State<'_, GpuCacheState>) -> AppResult<GpuInfo> {
    // Check cache first
    {
        let guard = cache.0.lock();
        if let Some(info) = guard.get(GPU_CACHE_TTL_SECS) {
            return Ok(info.clone());
        }
    }

    // Cache miss — query nvidia-smi
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader",
        ])
        .output()
        .map_err(|e| AppError::Gpu(format!("Failed to run nvidia-smi: {}", e)))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(AppError::Gpu(format!("nvidia-smi failed: {}", stderr)));
    }

    let text = String::from_utf8_lossy(&output.stdout);
    let parts: Vec<&str> = text.trim().split(", ").collect();

    if parts.len() >= 4 {
        let info = GpuInfo {
            name: parts[0].trim().to_string(),
            memory_total: parts[1].trim().to_string(),
            memory_used: parts[2].trim().to_string(),
            utilization: parts[3].trim().to_string(),
        };

        // Update cache
        {
            let mut guard = cache.0.lock();
            guard.set(info.clone());
        }

        Ok(info)
    } else {
        Err(AppError::Gpu(format!(
            "Unexpected nvidia-smi output: {}",
            text.trim()
        )))
    }
}
