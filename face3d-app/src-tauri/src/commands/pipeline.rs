use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::time::Instant;

use tauri::{command, AppHandle, Emitter, State};

use crate::error::{AppError, AppResult};
use crate::state::{
    PipelineComplete, PipelineLog, PipelineMetric, PipelineProcess, PipelineStageChange,
    CUDA_HOME, PROJECT_ROOT, PYTHON_EXE,
};

// ── Stdout line parser ───────────────────────────────────────────

/// Determine log level from a raw pipeline output line.
fn classify_log_level(text: &str) -> &'static str {
    if text.contains("[ERROR]") || text.contains("Error:") || text.contains("Traceback") {
        "error"
    } else if text.contains("[WARN") || text.contains("Warning:") {
        "warn"
    } else if text.contains("[DEBUG]") {
        "debug"
    } else {
        "info"
    }
}

/// Try to extract a stage change from a log line.
/// Expected format: "=== Stage N: Stage Name ==="  or  "[Stage N/14]"
fn parse_stage_change(text: &str) -> Option<(u32, String)> {
    // Pattern: === Stage 7: Depth Estimation ===
    if let Some(rest) = text.strip_prefix("=== Stage ") {
        if let Some(colon_idx) = rest.find(':') {
            if let Ok(num) = rest[..colon_idx].trim().parse::<u32>() {
                let name = rest[colon_idx + 1..]
                    .trim_end_matches('=')
                    .trim()
                    .to_string();
                return Some((num, name));
            }
        }
    }
    // Pattern: [Stage 7/14]
    if let Some(start) = text.find("[Stage ") {
        let after = &text[start + 7..];
        if let Some(slash) = after.find('/') {
            if let Ok(num) = after[..slash].trim().parse::<u32>() {
                // Try to get name after the bracket
                if let Some(end) = after.find(']') {
                    let name = after[end + 1..].trim().to_string();
                    let name = if name.is_empty() {
                        format!("Stage {}", num)
                    } else {
                        name
                    };
                    return Some((num, name));
                }
            }
        }
    }
    None
}

/// Try to extract training metrics from a log line.
/// Expected format contains: "iter=1000 loss=0.0123 psnr=25.4 gaussians=50000"
fn parse_training_metric(text: &str) -> Option<PipelineMetric> {
    // Look for lines that contain iteration info from Gaussian Splatting training
    let has_iter = text.contains("iter=") || text.contains("Iter ");
    if !has_iter {
        return None;
    }

    let mut iter: Option<u64> = None;
    let mut loss: Option<f64> = None;
    let mut psnr: Option<f64> = None;
    let mut gaussians: Option<u64> = None;

    for part in text.split_whitespace() {
        if let Some(val) = part.strip_prefix("iter=") {
            iter = val.parse().ok();
        } else if let Some(val) = part.strip_prefix("loss=") {
            loss = val.parse().ok();
        } else if let Some(val) = part.strip_prefix("psnr=") {
            psnr = val.parse().ok();
        } else if let Some(val) = part.strip_prefix("gaussians=") {
            gaussians = val.parse().ok();
        } else if let Some(val) = part.strip_prefix("n_gaussians=") {
            gaussians = val.parse().ok();
        }
    }

    // Also handle "Iter 1000: loss=0.0123" format
    if iter.is_none() {
        if let Some(idx) = text.find("Iter ") {
            let after = &text[idx + 5..];
            if let Some(end) = after.find(|c: char| !c.is_ascii_digit()) {
                iter = after[..end].parse().ok();
            }
        }
    }

    if let Some(it) = iter {
        Some(PipelineMetric {
            iter: it,
            loss: loss.unwrap_or(0.0),
            psnr: psnr.unwrap_or(0.0),
            gaussians: gaussians.unwrap_or(0),
        })
    } else {
        None
    }
}

/// Get a timestamp string for log entries.
fn now_timestamp() -> String {
    chrono::Local::now().format("%H:%M:%S%.3f").to_string()
}

// ── Commands ─────────────────────────────────────────────────────

#[command]
pub async fn start_pipeline(
    content_dir: String,
    session: String,
    app: AppHandle,
    state: State<'_, PipelineProcess>,
) -> AppResult<()> {
    // Check if already running
    {
        let guard = state.0.lock();
        if guard.is_running() {
            return Err(AppError::Pipeline("Pipeline already running".into()));
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
        .map_err(|e| AppError::Pipeline(format!("Failed to start pipeline: {}", e)))?;

    let pid = child.id();
    {
        let mut guard = state.0.lock();
        guard.pid = Some(pid);
        guard.started_at = Some(Instant::now());
        guard.current_stage = None;
        guard.current_stage_name = None;
    }

    // Stream stdout in a background thread, parsing structured events
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| AppError::Pipeline("Failed to capture stdout".into()))?;
    let app_stdout = app.clone();
    let state_arc = state.0.clone();
    std::thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            match line {
                Ok(text) => {
                    // Check for stage changes
                    if let Some((stage_num, stage_name)) = parse_stage_change(&text) {
                        {
                            let mut guard = state_arc.lock();
                            guard.current_stage = Some(stage_num);
                            guard.current_stage_name = Some(stage_name.clone());
                        }
                        let _ = app_stdout.emit(
                            "pipeline-stage-change",
                            PipelineStageChange {
                                stage: stage_num,
                                name: stage_name,
                            },
                        );
                    }

                    // Check for training metrics
                    if let Some(metric) = parse_training_metric(&text) {
                        let _ = app_stdout.emit("pipeline-metric", metric);
                    }

                    // Always emit the raw log line
                    let level = classify_log_level(&text);
                    let _ = app_stdout.emit(
                        "pipeline-log",
                        PipelineLog {
                            line: text,
                            level: level.to_string(),
                            timestamp: now_timestamp(),
                        },
                    );
                }
                Err(_) => break,
            }
        }
    });

    // Stream stderr in a background thread
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| AppError::Pipeline("Failed to capture stderr".into()))?;
    let app_stderr = app.clone();
    std::thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line in reader.lines() {
            match line {
                Ok(text) => {
                    let level = classify_log_level(&text);
                    // Stderr from Python often contains warnings/progress that are info-level
                    let effective_level = if level == "info" { "stderr" } else { level };
                    let _ = app_stderr.emit(
                        "pipeline-log",
                        PipelineLog {
                            line: text,
                            level: effective_level.to_string(),
                            timestamp: now_timestamp(),
                        },
                    );
                }
                Err(_) => break,
            }
        }
    });

    // Wait for child to finish in background
    let state_clone = state.0.clone();
    let app_done = app.clone();
    std::thread::spawn(move || {
        let started = {
            let guard = state_clone.lock();
            guard.started_at
        };
        let status = child.wait();
        let duration_secs = started
            .map(|t| t.elapsed().as_secs_f64())
            .unwrap_or(0.0);
        {
            let mut guard = state_clone.lock();
            guard.reset();
        }
        let exit_code = status.map(|s| s.code().unwrap_or(-1)).unwrap_or(-1);
        let _ = app_done.emit(
            "pipeline-complete",
            PipelineComplete {
                exit_code,
                duration_secs,
            },
        );
    });

    Ok(())
}

#[command]
pub async fn stop_pipeline(state: State<'_, PipelineProcess>) -> AppResult<()> {
    let pid = {
        let mut guard = state.0.lock();
        guard.pid.take()
    };

    if let Some(pid) = pid {
        // First try graceful termination via taskkill without /F
        let graceful = Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/T"])
            .output();

        match graceful {
            Ok(output) if output.status.success() => {
                // Give the process 5 seconds to exit gracefully
                let start = Instant::now();
                loop {
                    std::thread::sleep(std::time::Duration::from_millis(250));
                    // Check if process is still alive
                    let check = Command::new("tasklist")
                        .args(["/FI", &format!("PID eq {}", pid), "/NH"])
                        .output();
                    match check {
                        Ok(out) => {
                            let text = String::from_utf8_lossy(&out.stdout);
                            if !text.contains(&pid.to_string()) {
                                // Process exited cleanly
                                let mut guard = state.0.lock();
                                guard.reset();
                                return Ok(());
                            }
                        }
                        Err(_) => break,
                    }
                    if start.elapsed().as_secs() >= 5 {
                        break;
                    }
                }
            }
            _ => {}
        }

        // Force kill after timeout
        let _ = Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/F", "/T"])
            .output();

        let mut guard = state.0.lock();
        guard.reset();
        Ok(())
    } else {
        Err(AppError::Pipeline("No pipeline running".into()))
    }
}

#[command]
pub async fn is_pipeline_running(state: State<'_, PipelineProcess>) -> AppResult<bool> {
    let guard = state.0.lock();
    Ok(guard.is_running())
}
