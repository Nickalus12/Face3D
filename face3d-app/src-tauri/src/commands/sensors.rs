use std::path::Path;
use std::process::Command;

use tauri::command;

use crate::error::{AppError, AppResult};
use crate::state::{SensorSummary, PROJECT_ROOT, PYTHON_EXE};

/// Run the sensor NPZ reader script and return its stdout.
fn run_sensor_script(args: &[&str]) -> AppResult<String> {
    let script = format!("{}/scripts/read_sensor_npz.py", PROJECT_ROOT);
    let mut cmd_args = vec!["-u", &script];
    cmd_args.extend_from_slice(args);

    let output = Command::new(PYTHON_EXE)
        .args(&cmd_args)
        .current_dir(PROJECT_ROOT)
        .output()
        .map_err(|e| AppError::Sensor(format!("Failed to run sensor script: {}", e)))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(AppError::Sensor(format!("Sensor script failed: {}", stderr)));
    }

    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    Ok(stdout)
}

#[command]
pub async fn get_sensor_summary(session_id: String) -> AppResult<SensorSummary> {
    let npz_path = format!(
        "{}/data/processed/{}/sensors/sensor_logger_data.npz",
        PROJECT_ROOT, session_id
    );

    if !Path::new(&npz_path).exists() {
        return Ok(SensorSummary {
            device_name: String::new(),
            recording_time: String::new(),
            duration: 0.0,
            sensors: Vec::new(),
        });
    }

    let json_str = run_sensor_script(&["summary", &npz_path])?;
    let summary: SensorSummary =
        serde_json::from_str(&json_str).map_err(|e| AppError::Sensor(format!("Failed to parse summary: {}", e)))?;
    Ok(summary)
}

#[command]
pub async fn get_sensor_data(
    session_id: String,
    sensor_type: String,
) -> AppResult<Vec<Vec<f64>>> {
    let npz_path = format!(
        "{}/data/processed/{}/sensors/sensor_logger_data.npz",
        PROJECT_ROOT, session_id
    );

    if !Path::new(&npz_path).exists() {
        return Ok(Vec::new());
    }

    let json_str = run_sensor_script(&["data", &npz_path, &sensor_type])?;
    let data: Vec<Vec<f64>> =
        serde_json::from_str(&json_str).map_err(|e| AppError::Sensor(format!("Failed to parse sensor data: {}", e)))?;
    Ok(data)
}
