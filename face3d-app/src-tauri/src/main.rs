#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

use std::{
    fs,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
};

use serde::{Deserialize, Serialize};
use tauri::{command, AppHandle, Emitter, Manager, State};

const PROJECT_ROOT: &str = "D:/Projects/3D";
const PYTHON_EXE: &str = "C:/Users/Nicka/miniconda3/envs/face3d/python.exe";
const CUDA_HOME: &str = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8";

// ── Data types ───────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Session {
    id: String,
    name: String,
    has_gaussians: bool,
    has_mesh: bool,
    has_renders: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GpuInfo {
    name: String,
    memory_total: String,
    memory_used: String,
    utilization: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PipelineLog {
    line: String,
    level: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PreviewFile {
    name: String,
    preview_type: String,
    path: String,
}

// ── State ────────────────────────────────────────────────────────

struct PipelineProcess(Arc<Mutex<Option<u32>>>); // Stores PID

// ── Commands ─────────────────────────────────────────────────────

#[command]
async fn list_sessions() -> Result<Vec<Session>, String> {
    let output_dir = Path::new(PROJECT_ROOT).join("data/output");
    let mut sessions = Vec::new();

    if output_dir.exists() && output_dir.is_dir() {
        for entry in fs::read_dir(&output_dir).map_err(|e| e.to_string())? {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().into_owned();
            sessions.push(Session {
                id: name.clone(),
                name,
                has_gaussians: path.join("gaussians.ply").exists(),
                has_mesh: path.join("mesh.ply").exists() || path.join("mesh.obj").exists(),
                has_renders: path.join("renders").exists(),
            });
        }
    }

    sessions.sort_by(|a, b| b.id.cmp(&a.id)); // newest first
    Ok(sessions)
}

#[command]
async fn start_pipeline(
    content_dir: String,
    session: String,
    app: AppHandle,
    state: State<'_, PipelineProcess>,
) -> Result<(), String> {
    // Check if already running
    {
        let guard = state.0.lock().unwrap();
        if guard.is_some() {
            return Err("Pipeline already running".into());
        }
    }

    let script = format!("{}/scripts/run_pipeline.py", PROJECT_ROOT);

    let mut child = Command::new(PYTHON_EXE)
        .arg("-u") // unbuffered output
        .arg(&script)
        .arg("--content-dir")
        .arg(&content_dir)
        .arg("--session")
        .arg(&session)
        .env("CUDA_HOME", CUDA_HOME)
        .env("TORCH_CUDA_ARCH_LIST", "8.6")
        .current_dir(PROJECT_ROOT)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to start pipeline: {}", e))?;

    let pid = child.id();
    {
        let mut guard = state.0.lock().unwrap();
        *guard = Some(pid);
    }

    // Stream stdout in a background thread
    let stdout = child.stdout.take().unwrap();
    let app_stdout = app.clone();
    std::thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            match line {
                Ok(text) => {
                    let level = if text.contains("[ERROR]") {
                        "error"
                    } else if text.contains("[WARN") {
                        "warn"
                    } else {
                        "info"
                    };
                    let _ = app_stdout.emit("pipeline-log", PipelineLog {
                        line: text,
                        level: level.to_string(),
                    });
                }
                Err(_) => break,
            }
        }
    });

    // Stream stderr in a background thread
    let stderr = child.stderr.take().unwrap();
    let app_stderr = app.clone();
    std::thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line in reader.lines() {
            match line {
                Ok(text) => {
                    let _ = app_stderr.emit("pipeline-log", PipelineLog {
                        line: text,
                        level: "stderr".to_string(),
                    });
                }
                Err(_) => break,
            }
        }
    });

    // Wait for child to finish in background
    let state_clone = state.0.clone();
    let app_done = app.clone();
    std::thread::spawn(move || {
        let status = child.wait();
        let mut guard = state_clone.lock().unwrap();
        *guard = None;
        let exit_code = status.map(|s| s.code().unwrap_or(-1)).unwrap_or(-1);
        let _ = app_done.emit("pipeline-complete", exit_code);
    });

    Ok(())
}

#[command]
async fn stop_pipeline(state: State<'_, PipelineProcess>) -> Result<(), String> {
    let mut guard = state.0.lock().unwrap();
    if let Some(pid) = guard.take() {
        // Kill the process tree on Windows
        let _ = Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/F", "/T"])
            .output();
        Ok(())
    } else {
        Err("No pipeline running".into())
    }
}

#[command]
async fn get_gpu_info() -> Result<GpuInfo, String> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader"])
        .output()
        .map_err(|e| e.to_string())?;

    let text = String::from_utf8_lossy(&output.stdout);
    let parts: Vec<&str> = text.trim().split(", ").collect();

    if parts.len() >= 4 {
        Ok(GpuInfo {
            name: parts[0].trim().to_string(),
            memory_total: parts[1].trim().to_string(),
            memory_used: parts[2].trim().to_string(),
            utilization: parts[3].trim().to_string(),
        })
    } else {
        Err("Failed to parse nvidia-smi output".into())
    }
}

#[command]
async fn is_pipeline_running(state: State<'_, PipelineProcess>) -> Result<bool, String> {
    let guard = state.0.lock().unwrap();
    Ok(guard.is_some())
}

#[command]
async fn open_folder(path: String) -> Result<(), String> {
    Command::new("explorer")
        .arg(&path)
        .spawn()
        .map_err(|e| e.to_string())?;
    Ok(())
}

#[command]
async fn get_model_path(session_id: String) -> Result<String, String> {
    let output_dir = Path::new(PROJECT_ROOT)
        .join("data/output")
        .join(&session_id);

    // Prefer gaussians.ply, fall back to mesh.ply, then mesh.obj
    let candidates = ["gaussians.ply", "mesh.ply", "mesh.obj"];
    for name in &candidates {
        let path = output_dir.join(name);
        if path.exists() && path.is_file() {
            return Ok(path.to_string_lossy().into_owned());
        }
    }

    Err(format!("No model file found for session '{}'", session_id))
}

#[command]
async fn get_session_files(session_id: String) -> Result<Vec<(String, u64)>, String> {
    let output_dir = Path::new(PROJECT_ROOT)
        .join("data/output")
        .join(&session_id);

    let mut files = Vec::new();
    if output_dir.exists() {
        for entry in fs::read_dir(&output_dir).map_err(|e| e.to_string())? {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };
            let path = entry.path();
            if path.is_file() {
                let name = entry.file_name().to_string_lossy().into_owned();
                let size = fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                files.push((name, size));
            }
        }
    }

    files.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(files)
}

// ── Gallery commands ─────────────────────────────────────────────

fn list_png_files(dir: &Path) -> Vec<String> {
    let mut files = Vec::new();
    if dir.exists() && dir.is_dir() {
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() {
                    if let Some(ext) = path.extension() {
                        let ext_lower = ext.to_string_lossy().to_lowercase();
                        if ext_lower == "png" || ext_lower == "jpg" || ext_lower == "jpeg" {
                            files.push(path.to_string_lossy().into_owned());
                        }
                    }
                }
            }
        }
    }
    files.sort();
    files
}

#[command]
async fn list_renders(session_id: String) -> Result<Vec<String>, String> {
    let renders_dir = Path::new(PROJECT_ROOT)
        .join("data/output")
        .join(&session_id)
        .join("renders");
    Ok(list_png_files(&renders_dir))
}

#[command]
async fn list_previews(session_id: String) -> Result<Vec<PreviewFile>, String> {
    let previews_dir = Path::new(PROJECT_ROOT)
        .join("data/output")
        .join(&session_id)
        .join("previews");

    let mut result = Vec::new();
    if previews_dir.exists() && previews_dir.is_dir() {
        if let Ok(entries) = fs::read_dir(&previews_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if !path.is_file() {
                    continue;
                }
                if let Some(ext) = path.extension() {
                    let ext_lower = ext.to_string_lossy().to_lowercase();
                    if ext_lower != "png" && ext_lower != "jpg" && ext_lower != "jpeg" {
                        continue;
                    }
                } else {
                    continue;
                }

                let name = entry.file_name().to_string_lossy().into_owned();
                let name_lower = name.to_lowercase();

                let preview_type = if name_lower.contains("depth") {
                    "depth"
                } else if name_lower.contains("landmark") {
                    "landmark"
                } else if name_lower.contains("mask") {
                    "mask"
                } else if name_lower.contains("comparison") || name_lower.contains("compare") {
                    "comparison"
                } else {
                    "other"
                };

                result.push(PreviewFile {
                    name,
                    preview_type: preview_type.to_string(),
                    path: path.to_string_lossy().into_owned(),
                });
            }
        }
    }

    result.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(result)
}

#[command]
async fn list_input_frames(session_id: String) -> Result<Vec<String>, String> {
    // Try frames_srgb first, fall back to frames
    let srgb_dir = Path::new(PROJECT_ROOT)
        .join("data/processed")
        .join(&session_id)
        .join("frames_srgb");

    if srgb_dir.exists() && srgb_dir.is_dir() {
        let files = list_png_files(&srgb_dir);
        if !files.is_empty() {
            return Ok(files);
        }
    }

    let frames_dir = Path::new(PROJECT_ROOT)
        .join("data/processed")
        .join(&session_id)
        .join("frames");
    Ok(list_png_files(&frames_dir))
}

#[command]
async fn get_report_path(session_id: String) -> Result<Option<String>, String> {
    let report = Path::new(PROJECT_ROOT)
        .join("data/output")
        .join(&session_id)
        .join("report.html");

    if report.exists() && report.is_file() {
        Ok(Some(report.to_string_lossy().into_owned()))
    } else {
        Ok(None)
    }
}

#[command]
async fn get_image_counts(session_id: String) -> Result<(usize, usize, usize), String> {
    let renders_dir = Path::new(PROJECT_ROOT)
        .join("data/output")
        .join(&session_id)
        .join("renders");
    let previews_dir = Path::new(PROJECT_ROOT)
        .join("data/output")
        .join(&session_id)
        .join("previews");
    let srgb_dir = Path::new(PROJECT_ROOT)
        .join("data/processed")
        .join(&session_id)
        .join("frames_srgb");
    let frames_dir = Path::new(PROJECT_ROOT)
        .join("data/processed")
        .join(&session_id)
        .join("frames");

    let render_count = list_png_files(&renders_dir).len();
    let preview_count = list_png_files(&previews_dir).len();
    let frames = list_png_files(&srgb_dir);
    let frame_count = if frames.is_empty() {
        list_png_files(&frames_dir).len()
    } else {
        frames.len()
    };

    Ok((render_count, preview_count, frame_count))
}

// ── Settings commands ────────────────────────────────────────────

#[command]
async fn get_pipeline_config() -> Result<String, String> {
    let config_path = Path::new(PROJECT_ROOT).join("config/pipeline.yaml");
    fs::read_to_string(&config_path).map_err(|e| format!("Failed to read pipeline.yaml: {}", e))
}

#[command]
async fn save_pipeline_config(config: String) -> Result<(), String> {
    let config_path = Path::new(PROJECT_ROOT).join("config/pipeline.yaml");
    fs::write(&config_path, &config)
        .map_err(|e| format!("Failed to write pipeline.yaml: {}", e))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PythonInfo {
    version: String,
    env_name: String,
    packages_count: usize,
}

#[command]
async fn get_python_info() -> Result<PythonInfo, String> {
    // Get Python version
    let version_output = Command::new(PYTHON_EXE)
        .arg("--version")
        .output()
        .map_err(|e| format!("Failed to run python --version: {}", e))?;
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
        .map_err(|e| format!("Failed to run pip list: {}", e))?;
    let pip_text = String::from_utf8_lossy(&pip_output.stdout);
    // Subtract 2 for header lines
    let packages_count = pip_text.lines().count().saturating_sub(2);

    Ok(PythonInfo {
        version,
        env_name,
        packages_count,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SystemInfo {
    os_version: String,
    cpu: String,
    ram_gb: f64,
    disk_free_gb: f64,
    disk_total_gb: f64,
    cuda_version: String,
}

#[command]
async fn get_system_info() -> Result<SystemInfo, String> {
    // OS version
    let os_output = Command::new("cmd")
        .args(["/C", "ver"])
        .output()
        .map_err(|e| e.to_string())?;
    let os_version = String::from_utf8_lossy(&os_output.stdout).trim().to_string();

    // CPU info via wmic
    let cpu_output = Command::new("wmic")
        .args(["cpu", "get", "name", "/value"])
        .output()
        .map_err(|e| e.to_string())?;
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
        .map_err(|e| e.to_string())?;
    let ram_text = String::from_utf8_lossy(&ram_output.stdout);
    let ram_kb: f64 = ram_text
        .lines()
        .find(|l| l.starts_with("TotalVisibleMemorySize="))
        .and_then(|l| l.trim_start_matches("TotalVisibleMemorySize=").trim().parse().ok())
        .unwrap_or(0.0);
    let ram_gb = ram_kb / 1_048_576.0;

    // Disk space for D: drive
    let disk_output = Command::new("wmic")
        .args(["logicaldisk", "where", "DeviceID='D:'", "get", "FreeSpace,Size", "/value"])
        .output()
        .map_err(|e| e.to_string())?;
    let disk_text = String::from_utf8_lossy(&disk_output.stdout);
    let mut disk_free_gb = 0.0_f64;
    let mut disk_total_gb = 0.0_f64;
    for line in disk_text.lines() {
        let line = line.trim();
        if line.starts_with("FreeSpace=") {
            if let Ok(v) = line.trim_start_matches("FreeSpace=").parse::<f64>() {
                disk_free_gb = v / 1_073_741_824.0;
            }
        }
        if line.starts_with("Size=") {
            if let Ok(v) = line.trim_start_matches("Size=").parse::<f64>() {
                disk_total_gb = v / 1_073_741_824.0;
            }
        }
    }

    // CUDA version from nvcc
    let cuda_path = Path::new(CUDA_HOME).join("bin/nvcc.exe");
    let cuda_version = if cuda_path.exists() {
        let nvcc_output = Command::new(cuda_path.to_str().unwrap())
            .arg("--version")
            .output()
            .ok();
        nvcc_output
            .map(|o| {
                let text = String::from_utf8_lossy(&o.stdout).to_string();
                text.lines()
                    .find(|l| l.contains("release"))
                    .map(|l| l.to_string())
                    .unwrap_or_else(|| "Unknown".to_string())
            })
            .unwrap_or_else(|| "Unknown".to_string())
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

// ── Sensor data commands ─────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SensorInfo {
    name: String,
    samples: u64,
    duration: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SensorSummary {
    device_name: String,
    recording_time: String,
    duration: f64,
    sensors: Vec<SensorInfo>,
}

fn run_sensor_script(args: &[&str]) -> Result<String, String> {
    let script = format!("{}/scripts/read_sensor_npz.py", PROJECT_ROOT);
    let mut cmd_args = vec!["-u", &script];
    cmd_args.extend_from_slice(args);

    let output = Command::new(PYTHON_EXE)
        .args(&cmd_args)
        .current_dir(PROJECT_ROOT)
        .output()
        .map_err(|e| format!("Failed to run sensor script: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Sensor script failed: {}", stderr));
    }

    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    Ok(stdout)
}

#[command]
async fn get_sensor_summary(session_id: String) -> Result<SensorSummary, String> {
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
        serde_json::from_str(&json_str).map_err(|e| format!("Failed to parse summary: {}", e))?;
    Ok(summary)
}

#[command]
async fn get_sensor_data(
    session_id: String,
    sensor_type: String,
) -> Result<Vec<Vec<f64>>, String> {
    let npz_path = format!(
        "{}/data/processed/{}/sensors/sensor_logger_data.npz",
        PROJECT_ROOT, session_id
    );

    if !Path::new(&npz_path).exists() {
        return Ok(Vec::new());
    }

    let json_str = run_sensor_script(&["data", &npz_path, &sensor_type])?;
    let data: Vec<Vec<f64>> =
        serde_json::from_str(&json_str).map_err(|e| format!("Failed to parse sensor data: {}", e))?;
    Ok(data)
}

// ── Session metrics command ──────────────────────────────────────

#[command]
async fn get_session_metrics(session_id: String) -> Result<String, String> {
    let metrics_path = Path::new(PROJECT_ROOT)
        .join("data/output")
        .join(&session_id)
        .join("metrics.jsonl");

    if metrics_path.exists() && metrics_path.is_file() {
        let content = fs::read_to_string(&metrics_path).map_err(|e| e.to_string())?;
        // Convert JSONL to a JSON array string
        let lines: Vec<&str> = content.lines().filter(|l| !l.trim().is_empty()).collect();
        let json_array = format!("[{}]", lines.join(","));
        Ok(json_array)
    } else {
        Ok(String::new())
    }
}

// ── Main ─────────────────────────────────────────────────────────

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(PipelineProcess(Arc::new(Mutex::new(None))))
        .invoke_handler(tauri::generate_handler![
            list_sessions,
            start_pipeline,
            stop_pipeline,
            get_gpu_info,
            is_pipeline_running,
            open_folder,
            get_session_files,
            get_model_path,
            list_renders,
            list_previews,
            list_input_frames,
            get_report_path,
            get_image_counts,
            get_sensor_summary,
            get_sensor_data,
            get_pipeline_config,
            save_pipeline_config,
            get_python_info,
            get_system_info,
            get_session_metrics,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
