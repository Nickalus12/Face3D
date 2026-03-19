/**
 * Intelligent Pipeline Auto-Preset
 *
 * Analyzes hardware capabilities + input data to determine the optimal
 * pipeline configuration. No user selection needed.
 *
 * Scoring:
 *   hardware_score (1-5) — based on VRAM, GPU tier
 *   input_score    (1-5) — based on data richness (videos, photos, sensors, DNGs)
 *   final_score    = min(hardware_score, input_score)
 *
 * The minimum ensures we never exceed hardware limits (no point training
 * at maximum quality on a 4GB GPU) and never waste time over-processing
 * minimal input (no point running 7K iterations on a single low-res video).
 */

// ── Types ──────────────────────────────────────────────────────

export interface ScanResult {
  videos: number;
  photos: number;
  photos_jpg: number;
  photos_dng: number;
  sensor_logs: number;
  total_bytes: number;
  total_size: string;
  largest_video_mb: number;
}

export interface GpuCapabilities {
  name: string;
  vram_total_gb: number;
  vram_available_gb: number;
}

export interface PipelineConfig {
  iterations: number;
  maxGaussians: number;
  processRes: number;
  maxFrames: number;
  textureRes: number;
}

export interface AutoPresetResult {
  score: number;           // 1-5
  label: string;           // "Quick", "Standard", "High", "Ultra", "Maximum"
  config: PipelineConfig;
  estimatedMinutes: number;
  estimatedVramGb: number;
  reasons: string[];       // Human-readable explanations
  hardwareScore: number;
  inputScore: number;
}

// ── Score Tables ───────────────────────────────────────────────

const CONFIGS: Record<number, PipelineConfig> = {
  1: { iterations: 500,  maxGaussians: 100_000, processRes: 336, maxFrames: 40,  textureRes: 1024 },
  2: { iterations: 1000, maxGaussians: 150_000, processRes: 504, maxFrames: 60,  textureRes: 1024 },
  3: { iterations: 3000, maxGaussians: 300_000, processRes: 504, maxFrames: 80,  textureRes: 2048 },
  4: { iterations: 5000, maxGaussians: 400_000, processRes: 768, maxFrames: 120, textureRes: 2048 },
  5: { iterations: 7000, maxGaussians: 500_000, processRes: 1024, maxFrames: 200, textureRes: 4096 },
};

const LABELS: Record<number, string> = {
  1: "Quick Preview",
  2: "Standard",
  3: "High Quality",
  4: "Ultra",
  5: "Maximum",
};

// Rough estimates based on RTX 3080 benchmarks
const TIME_ESTIMATES: Record<number, number> = {
  1: 3,    // minutes
  2: 5,
  3: 10,
  4: 18,
  5: 30,
};

const VRAM_ESTIMATES: Record<number, number> = {
  1: 2.5,  // GB
  2: 3.5,
  3: 5,
  4: 8,
  5: 12,
};

// ── Hardware Scoring ───────────────────────────────────────────

function scoreHardware(gpu: GpuCapabilities | null): { score: number; reasons: string[] } {
  if (!gpu) {
    return { score: 2, reasons: ["No GPU detected — using conservative defaults"] };
  }

  const vram = gpu.vram_total_gb;
  const reasons: string[] = [];
  let score: number;

  if (vram >= 16) {
    score = 5;
    reasons.push(`${vram.toFixed(0)}GB VRAM — full quality supported`);
  } else if (vram >= 12) {
    score = 4;
    reasons.push(`${vram.toFixed(0)}GB VRAM — high quality supported`);
  } else if (vram >= 8) {
    score = 3;
    reasons.push(`${vram.toFixed(0)}GB VRAM — standard quality`);
  } else if (vram >= 6) {
    score = 2;
    reasons.push(`${vram.toFixed(0)}GB VRAM — limited, reduced settings`);
  } else {
    score = 1;
    reasons.push(`${vram.toFixed(0)}GB VRAM — minimal, preview only`);
  }

  // Bonus for high-end GPUs (they're faster per iteration too)
  const name = gpu.name.toLowerCase();
  if (name.includes("4090") || name.includes("a100") || name.includes("h100")) {
    reasons.push(`${gpu.name} — high-end GPU, faster training`);
  } else if (name.includes("4080") || name.includes("3090")) {
    reasons.push(`${gpu.name} — powerful GPU`);
  }

  return { score, reasons };
}

// ── Input Scoring ──────────────────────────────────────────────

function scoreInput(scan: ScanResult): { score: number; reasons: string[] } {
  const reasons: string[] = [];
  let score = 1;

  // Video presence (baseline)
  if (scan.videos === 0) {
    reasons.push("No video — limited reconstruction possible");
    return { score: 1, reasons };
  }

  score = 2; // At least one video
  reasons.push(`${scan.videos} video${scan.videos > 1 ? "s" : ""}`);

  // Video size as quality proxy
  // < 50MB = very short / low-res, 50-200MB = normal, 200MB+ = long / high-res, 500MB+ = 8K
  if (scan.largest_video_mb > 500) {
    score = Math.max(score, 3);
    reasons.push("8K/long video — higher resolution input");
  } else if (scan.largest_video_mb > 200) {
    score = Math.max(score, 3);
    reasons.push("Good quality video source");
  }

  // Photos boost quality significantly
  if (scan.photos_dng > 0) {
    score = Math.max(score, 4);
    reasons.push(`${scan.photos_dng} Expert RAW photos — maximum texture quality`);
    if (scan.photos_dng >= 10) {
      score = 5;
      reasons.push("Dense RAW coverage — full quality warranted");
    }
  } else if (scan.photos_jpg > 0) {
    score = Math.max(score, 3);
    reasons.push(`${scan.photos_jpg} JPG photos — enhanced texture detail`);
    if (scan.photos_jpg >= 15) {
      score = Math.max(score, 4);
    }
  }

  // Sensor data enables better poses
  if (scan.sensor_logs > 0) {
    score = Math.min(score + 1, 5);
    reasons.push(`${scan.sensor_logs} sensor log${scan.sensor_logs > 1 ? "s" : ""} — improved camera tracking`);
  }

  // Multi-source = highest potential
  if (scan.videos > 0 && scan.photos > 0 && scan.sensor_logs > 0) {
    score = Math.max(score, 4);
    reasons.push("Full multi-source capture — video + photos + sensors");
  }

  return { score, reasons };
}

// ── Main Function ──────────────────────────────────────────────

export function computeAutoPreset(
  scan: ScanResult,
  gpu: GpuCapabilities | null,
): AutoPresetResult {
  const hw = scoreHardware(gpu);
  const inp = scoreInput(scan);

  // Final score is the minimum — can't exceed either limit
  const finalScore = Math.min(hw.score, inp.score);

  // If hardware is the bottleneck, explain why
  const reasons = [...inp.reasons];
  if (hw.score < inp.score) {
    reasons.push(`⚠ Limited by GPU — ${hw.reasons[0]}`);
  } else {
    reasons.push(...hw.reasons);
  }

  return {
    score: finalScore,
    label: LABELS[finalScore],
    config: CONFIGS[finalScore],
    estimatedMinutes: TIME_ESTIMATES[finalScore],
    estimatedVramGb: VRAM_ESTIMATES[finalScore],
    reasons,
    hardwareScore: hw.score,
    inputScore: inp.score,
  };
}
