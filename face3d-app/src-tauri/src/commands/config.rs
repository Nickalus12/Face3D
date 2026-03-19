use std::fs;
use std::path::Path;
use std::process::Command;

use tauri::command;

use crate::error::{AppError, AppResult};
use crate::state::{PythonInfo, SystemInfo, CUDA_HOME, PROJECT_ROOT, PYTHON_EXE};

#[command]
pub async fn get_pipeline_config() -> AppResult<String> {
    let config_path = Path::new(PROJECT_ROOT).join("config/pipeline.yaml");
    let content = fs::read_to_string(&config_path)
        .map_err(|e| AppError::Config(format!("Failed to read pipeline.yaml: {}", e)))?;
    Ok(content)
}

#[command]
pub async fn save_pipeline_config(config: String) -> AppResult<()> {
    // Validate YAML syntax before saving
    serde_yaml::from_str::<serde_yaml::Value>(&config)
        .map_err(|e| AppError::Yaml(format!("Invalid YAML syntax: {}", e)))?;

    let config_path = Path::new(PROJECT_ROOT).join("config/pipeline.yaml");

    // Create a backup before overwriting
    if config_path.exists() {
        let backup_path = Path::new(PROJECT_ROOT).join("config/pipeline.yaml.bak");
        fs::copy(&config_path, &backup_path)
            .map_err(|e| AppError::Config(format!("Failed to create backup: {}", e)))?;
    }

    fs::write(&config_path, &config)
        .map_err(|e| AppError::Config(format!("Failed to write pipeline.yaml: {}", e)))?;

    Ok(())
}

#[command]
pub async fn get_python_info() -> AppResult<PythonInfo> {
    // Get Python version
    let version_output = Command::new(PYTHON_EXE)
        .arg("--version")
        .output()
        .map_err(|e| AppError::System(format!("Failed to run python --version: {}", e)))?;
    let version = String::from_utf8_lossy(&version_output.stdout)
        .trim()
        .to_string();

    // Extract conda env name from PYTHON_EXE path
    let env_name = Path::new(PYTHON_EXE)
        .parent()
        .and_then(|p| p.file_name())
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "unknown".to_string());

    // Count installed packages
    let pip_output = Command::new(PYTHON_EXE)
        .args(["-m", "pip", "list", "--format=columns"])
        .output()
        .map_err(|e| AppError::System(format!("Failed to run pip list: {}", e)))?;
    let pip_text = String::from_utf8_lossy(&pip_output.stdout);
    // Subtract 2 for header lines
    let packages_count = pip_text.lines().count().saturating_sub(2);

    Ok(PythonInfo {
        version,
        env_name,
        packages_count,
    })
}

#[command]
pub async fn get_system_info() -> AppResult<SystemInfo> {
    // OS version
    let os_output = Command::new("cmd")
        .args(["/C", "ver"])
        .output()
        .map_err(|e| AppError::System(e.to_string()))?;
    let os_version = String::from_utf8_lossy(&os_output.stdout)
        .trim()
        .to_string();

    // CPU info via wmic
    let cpu_output = Command::new("wmic")
        .args(["cpu", "get", "name", "/value"])
        .output()
        .map_err(|e| AppError::System(e.to_string()))?;
    let cpu_text = String::from_utf8_lossy(&cpu_output.stdout);
    let cpu = cpu_text
        .lines()
        .find(|l| l.starts_with("Name="))
        .map(|l| l.trim_start_matches("Name=").trim().to_string())
        .unwrap_or_else(|| "Unknown".to_string());

    // RAM via wmic
    let ram_output = Command::new("wmic")
        .args(["os", "get", "TotalVisibleMemorySize", "/value"])
        .output()
        .map_err(|e| AppError::System(e.to_string()))?;
    let ram_text = String::from_utf8_lossy(&ram_output.stdout);
    let ram_kb: f64 = ram_text
        .lines()
        .find(|l| l.starts_with("TotalVisibleMemorySize="))
        .and_then(|l| {
            l.trim_start_matches("TotalVisibleMemorySize=")
                .trim()
                .parse()
                .ok()
        })
        .unwrap_or(0.0);
    let ram_gb = ram_kb / 1_048_576.0;

    // Disk space for D: drive
    let disk_output = Command::new("wmic")
        .args([
            "logicaldisk",
            "where",
            "DeviceID='D:'",
            "get",
            "FreeSpace,Size",
            "/value",
        ])
        .output()
        .map_err(|e| AppError::System(e.to_string()))?;
    let disk_text = String::from_utf8_lossy(&disk_output.stdout);
    let mut disk_free_gb = 0.0_f64;
    let mut disk_total_gb = 0.0_f64;
    for line in disk_text.lines() {
        let line = line.trim();
        if let Some(val) = line.strip_prefix("FreeSpace=") {
            if let Ok(v) = val.parse::<f64>() {
                disk_free_gb = v / 1_073_741_824.0;
            }
        }
        if let Some(val) = line.strip_prefix("Size=") {
            if let Ok(v) = val.parse::<f64>() {
                disk_total_gb = v / 1_073_741_824.0;
            }
        }
    }

    // CUDA version from nvcc
    let cuda_path = Path::new(CUDA_HOME).join("bin/nvcc.exe");
    let cuda_version = if cuda_path.exists() {
        match Command::new(cuda_path.to_str().unwrap_or_default())
            .arg("--version")
            .output()
        {
            Ok(o) => {
                let text = String::from_utf8_lossy(&o.stdout).to_string();
                text.lines()
                    .find(|l| l.contains("release"))
                    .map(|l| l.to_string())
                    .unwrap_or_else(|| "Unknown".to_string())
            }
            Err(_) => "Unknown".to_string(),
        }
    } else {
        format!("CUDA_HOME: {}", CUDA_HOME)
    };

    Ok(SystemInfo {
        os_version,
        cpu,
        ram_gb,
        disk_free_gb,
        disk_total_gb,
        cuda_version,
    })
}

#[command]
pub async fn open_folder(path: String) -> AppResult<()> {
    Command::new("explorer")
        .arg(&path)
        .spawn()
        .map_err(|e| AppError::System(format!("Failed to open folder: {}", e)))?;
    Ok(())
}
