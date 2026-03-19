import React, { useState, useCallback } from "react";
import {
  X,
  Upload,
  Video,
  Image as ImageIcon,
  FileText,
  ChevronRight,
  ChevronLeft,
  Zap,
  Scale,
  Crown,
  Settings2,
  Rocket,
  HardDrive,
  Clock,
  Cpu,
  CheckCircle2,
} from "lucide-react";
import { selectContentDir } from "../lib/tauri";
import { invoke } from "@tauri-apps/api/core";
import usePipelineStore from "../store/pipelineStore";
import useSessionStore from "../store/sessionStore";
import useToastStore from "../store/toastStore";

// ── Types ─────────────────────────────────────────────────────

interface DetectedFiles {
  videos: number;
  photos: number;
  sensorLogs: number;
  totalSize: string;
}

type QualityPreset = "quick" | "balanced" | "maximum";

interface AdvancedConfig {
  iterations: number;
  maxGaussians: number;
  resolution: number;
}

const PRESET_DEFAULTS: Record<QualityPreset, AdvancedConfig> = {
  quick: { iterations: 1000, maxGaussians: 150_000, resolution: 1024 },
  balanced: { iterations: 3000, maxGaussians: 300_000, resolution: 2048 },
  maximum: { iterations: 7000, maxGaussians: 500_000, resolution: 4096 },
};

const PRESET_META: Record<
  QualityPreset,
  { label: string; icon: React.FC<{ size?: number; className?: string }>; description: string; time: string }
> = {
  quick: {
    label: "Quick",
    icon: Zap,
    description: "Fast preview — fewer Gaussians, lower res",
    time: "~5 min",
  },
  balanced: {
    label: "Balanced",
    icon: Scale,
    description: "Best quality/speed tradeoff",
    time: "~10 min",
  },
  maximum: {
    label: "Maximum",
    icon: Crown,
    description: "Highest fidelity — more iterations, full resolution",
    time: "~25 min",
  },
};

interface NewScanWizardProps {
  isOpen: boolean;
  onClose: () => void;
}

// ── Component ─────────────────────────────────────────────────

const NewScanWizard: React.FC<NewScanWizardProps> = ({ isOpen, onClose }) => {
  const [step, setStep] = useState(1);
  const [contentDir, setContentDir] = useState<string | null>(null);
  const [detectedFiles, setDetectedFiles] = useState<DetectedFiles | null>(null);
  const [sessionName, setSessionName] = useState(
    () => `scan_${new Date().toISOString().slice(0, 10).replace(/-/g, "")}`
  );
  const [preset, setPreset] = useState<QualityPreset>("balanced");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [advanced, setAdvanced] = useState<AdvancedConfig>(PRESET_DEFAULTS.balanced);
  const [isDragOver, setIsDragOver] = useState(false);
  const [slideDir, setSlideDir] = useState<"left" | "right">("right");

  const { startPipeline } = usePipelineStore();
  const { createSession } = useSessionStore();
  const { addToast } = useToastStore();

  // ── Folder selection ────────────────────────────────────────

  const handleBrowse = useCallback(async () => {
    const dir = await selectContentDir();
    if (dir) {
      setContentDir(dir);
      // Actually scan the directory for content files
      try {
        const result = await invoke<{
          videos: number;
          photos: number;
          photos_dng: number;
          sensor_logs: number;
          total_size: string;
        }>("scan_content_dir", { path: dir });
        setDetectedFiles({
          videos: result.videos,
          photos: result.photos,
          sensorLogs: result.sensor_logs,
          totalSize: result.total_size,
        });
        // Auto-generate a friendlier session name from folder name
        const folderName = dir.split(/[/\\]/).filter(Boolean).pop() ?? "";
        if (folderName && folderName !== "New") {
          setSessionName(folderName.toLowerCase().replace(/\s+/g, "_"));
        }
      } catch {
        // Fallback if scan fails
        setDetectedFiles({ videos: 0, photos: 0, sensorLogs: 0, totalSize: "Unknown" });
      }
    }
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const items = e.dataTransfer.files;
    if (items.length > 0) {
      const path = (items[0] as any).path ?? items[0].name;
      setContentDir(path);
      try {
        const result = await invoke<{
          videos: number; photos: number; sensor_logs: number; total_size: string;
        }>("scan_content_dir", { path });
        setDetectedFiles({
          videos: result.videos, photos: result.photos,
          sensorLogs: result.sensor_logs, totalSize: result.total_size,
        });
      } catch {
        setDetectedFiles({ videos: 0, photos: 0, sensorLogs: 0, totalSize: "Unknown" });
      }
    }
  }, []);

  // ── Navigation ──────────────────────────────────────────────

  const goNext = () => {
    setSlideDir("right");
    setStep((s) => Math.min(s + 1, 3));
  };

  const goBack = () => {
    setSlideDir("left");
    setStep((s) => Math.max(s - 1, 1));
  };

  const handlePresetChange = (p: QualityPreset) => {
    setPreset(p);
    setAdvanced(PRESET_DEFAULTS[p]);
  };

  // ── Start pipeline ─────────────────────────────────────────

  const handleStart = async () => {
    if (!contentDir) return;
    createSession(sessionName, contentDir);
    try {
      await startPipeline(contentDir, sessionName);
      addToast("Pipeline started successfully", "success");
      handleClose();
    } catch {
      addToast("Failed to start pipeline", "error");
    }
  };

  const handleClose = () => {
    setStep(1);
    setContentDir(null);
    setDetectedFiles(null);
    setSessionName(
      `scan_${new Date().toISOString().slice(0, 10).replace(/-/g, "")}`
    );
    setPreset("balanced");
    setAdvanced(PRESET_DEFAULTS.balanced);
    setShowAdvanced(false);
    onClose();
  };

  if (!isOpen) return null;

  // ── Slide animation class ──────────────────────────────────

  const slideClass =
    slideDir === "right"
      ? "animate-[slideInRight_250ms_ease-out]"
      : "animate-[slideInLeft_250ms_ease-out]";

  return (
    <div className="fixed inset-0 z-[90] flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/70 backdrop-blur-sm animate-[fadeIn_200ms_ease-out]"
        onClick={handleClose}
      />

      {/* Modal */}
      <div className="relative w-full max-w-[560px] bg-[#111113] border border-zinc-800 rounded-2xl shadow-2xl animate-[scaleIn_250ms_ease-out] flex flex-col max-h-[85vh]">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-zinc-800/60 shrink-0">
          <div>
            <h2 className="text-lg font-semibold text-zinc-100">New Scan</h2>
            <p className="text-xs text-zinc-500 mt-0.5">
              Step {step} of 3 —{" "}
              {step === 1
                ? "Select Content"
                : step === 2
                ? "Configure"
                : "Review"}
            </p>
          </div>
          <button
            onClick={handleClose}
            className="p-2 rounded-lg text-zinc-500 hover:text-zinc-200 hover:bg-white/5 transition-colors"
          >
            <X size={18} />
          </button>
        </div>

        {/* Step Indicators */}
        <div className="flex gap-2 px-6 pt-4 shrink-0">
          {[1, 2, 3].map((s) => (
            <div
              key={s}
              className={`h-1 flex-1 rounded-full transition-all duration-300 ${
                s <= step ? "bg-indigo-500" : "bg-zinc-800"
              }`}
            />
          ))}
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto px-6 py-5">
          {/* ── STEP 1: Select Content ───────────────── */}
          {step === 1 && (
            <div key="step1" className={slideClass}>
              <div
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={handleBrowse}
                className={`border-2 border-dashed rounded-xl p-8 flex flex-col items-center gap-4 cursor-pointer transition-all duration-200 ${
                  isDragOver
                    ? "border-indigo-500 bg-indigo-500/5"
                    : contentDir
                    ? "border-emerald-500/40 bg-emerald-500/5"
                    : "border-zinc-700 hover:border-zinc-500 bg-zinc-900/30"
                }`}
              >
                {contentDir ? (
                  <CheckCircle2 size={36} className="text-emerald-400" />
                ) : (
                  <Upload size={36} className="text-zinc-500" />
                )}
                <div className="text-center">
                  <p className="text-sm font-medium text-zinc-200">
                    {contentDir
                      ? "Content folder selected"
                      : "Drop content folder here"}
                  </p>
                  <p className="text-xs text-zinc-500 mt-1">
                    {contentDir ?? "or click to browse"}
                  </p>
                </div>
              </div>

              {/* Detected files */}
              {detectedFiles && (
                <div className="mt-5 grid grid-cols-3 gap-3">
                  <div className="flex items-center gap-2.5 p-3 rounded-lg bg-zinc-900/50 border border-zinc-800/60">
                    <Video size={16} className="text-indigo-400 shrink-0" />
                    <div>
                      <p className="text-sm font-medium text-zinc-200">
                        {detectedFiles.videos}
                      </p>
                      <p className="text-[11px] text-zinc-500">Videos</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2.5 p-3 rounded-lg bg-zinc-900/50 border border-zinc-800/60">
                    <ImageIcon
                      size={16}
                      className="text-amber-400 shrink-0"
                    />
                    <div>
                      <p className="text-sm font-medium text-zinc-200">
                        {detectedFiles.photos}
                      </p>
                      <p className="text-[11px] text-zinc-500">Photos</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2.5 p-3 rounded-lg bg-zinc-900/50 border border-zinc-800/60">
                    <FileText
                      size={16}
                      className="text-emerald-400 shrink-0"
                    />
                    <div>
                      <p className="text-sm font-medium text-zinc-200">
                        {detectedFiles.sensorLogs}
                      </p>
                      <p className="text-[11px] text-zinc-500">Sensor Logs</p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ── STEP 2: Configure ────────────────────── */}
          {step === 2 && (
            <div key="step2" className={`space-y-5 ${slideClass}`}>
              {/* Session Name */}
              <div>
                <label className="block text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-2">
                  Session Name
                </label>
                <input
                  type="text"
                  value={sessionName}
                  onChange={(e) => setSessionName(e.target.value)}
                  className="w-full px-3 py-2.5 rounded-lg bg-zinc-900/60 border border-zinc-700/60 text-sm text-zinc-200 placeholder-zinc-600 focus:border-indigo-500/50 focus:outline-none focus:ring-1 focus:ring-indigo-500/20 transition-colors"
                  placeholder="my_scan_001"
                />
              </div>

              {/* Quality Preset */}
              <div>
                <label className="block text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-2">
                  Quality Preset
                </label>
                <div className="grid grid-cols-3 gap-2">
                  {(
                    Object.entries(PRESET_META) as [
                      QualityPreset,
                      (typeof PRESET_META)[QualityPreset]
                    ][]
                  ).map(([key, meta]) => {
                    const Icon = meta.icon;
                    const isSelected = preset === key;
                    return (
                      <button
                        key={key}
                        onClick={() => handlePresetChange(key)}
                        className={`p-3 rounded-xl border text-left transition-all duration-200 ${
                          isSelected
                            ? "border-indigo-500/50 bg-indigo-500/10 ring-1 ring-indigo-500/20"
                            : "border-zinc-800 bg-zinc-900/30 hover:border-zinc-600"
                        }`}
                      >
                        <Icon
                          size={18}
                          className={
                            isSelected ? "text-indigo-400" : "text-zinc-500"
                          }
                        />
                        <p
                          className={`text-sm font-medium mt-2 ${
                            isSelected ? "text-indigo-300" : "text-zinc-300"
                          }`}
                        >
                          {meta.label}
                        </p>
                        <p className="text-[11px] text-zinc-500 mt-0.5">
                          {meta.description}
                        </p>
                        <p className="text-[11px] text-zinc-600 mt-1">
                          {meta.time}
                        </p>
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Advanced Toggle */}
              <div>
                <button
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  className="flex items-center gap-2 text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
                >
                  <Settings2 size={14} />
                  <span>Advanced Parameters</span>
                  <ChevronRight
                    size={14}
                    className={`transition-transform duration-200 ${
                      showAdvanced ? "rotate-90" : ""
                    }`}
                  />
                </button>

                <div
                  className={`overflow-hidden transition-all duration-300 ease-out ${
                    showAdvanced
                      ? "max-h-60 opacity-100 mt-3"
                      : "max-h-0 opacity-0"
                  }`}
                >
                  <div className="space-y-3 p-4 rounded-lg bg-zinc-900/40 border border-zinc-800/50">
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-zinc-400">Iterations</span>
                      <input
                        type="number"
                        value={advanced.iterations}
                        onChange={(e) =>
                          setAdvanced((a) => ({
                            ...a,
                            iterations: parseInt(e.target.value, 10) || 0,
                          }))
                        }
                        className="w-24 px-2 py-1 text-right text-xs rounded bg-zinc-800 border border-zinc-700 text-zinc-200 focus:border-indigo-500/50 focus:outline-none"
                      />
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-zinc-400">
                        Max Gaussians
                      </span>
                      <input
                        type="number"
                        value={advanced.maxGaussians}
                        onChange={(e) =>
                          setAdvanced((a) => ({
                            ...a,
                            maxGaussians: parseInt(e.target.value, 10) || 0,
                          }))
                        }
                        className="w-24 px-2 py-1 text-right text-xs rounded bg-zinc-800 border border-zinc-700 text-zinc-200 focus:border-indigo-500/50 focus:outline-none"
                      />
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-zinc-400">Resolution</span>
                      <input
                        type="number"
                        value={advanced.resolution}
                        onChange={(e) =>
                          setAdvanced((a) => ({
                            ...a,
                            resolution: parseInt(e.target.value, 10) || 0,
                          }))
                        }
                        className="w-24 px-2 py-1 text-right text-xs rounded bg-zinc-800 border border-zinc-700 text-zinc-200 focus:border-indigo-500/50 focus:outline-none"
                      />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* ── STEP 3: Review ───────────────────────── */}
          {step === 3 && (
            <div key="step3" className={`space-y-4 ${slideClass}`}>
              <div className="p-4 rounded-xl bg-zinc-900/40 border border-zinc-800/50 space-y-3">
                <div className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-2">
                  Summary
                </div>

                <div className="flex items-center justify-between text-sm">
                  <span className="text-zinc-500">Session</span>
                  <span className="text-zinc-200 font-medium">
                    {sessionName}
                  </span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-zinc-500">Content</span>
                  <span
                    className="text-zinc-200 font-mono text-xs max-w-[260px] truncate"
                    title={contentDir ?? ""}
                  >
                    {contentDir}
                  </span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-zinc-500">Quality</span>
                  <span className="text-indigo-300 font-medium capitalize">
                    {preset}
                  </span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-zinc-500">Iterations</span>
                  <span className="text-zinc-200">
                    {advanced.iterations.toLocaleString()}
                  </span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-zinc-500">Max Gaussians</span>
                  <span className="text-zinc-200">
                    {advanced.maxGaussians.toLocaleString()}
                  </span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-zinc-500">Resolution</span>
                  <span className="text-zinc-200">{advanced.resolution}px</span>
                </div>
              </div>

              {/* Estimated resources */}
              <div className="grid grid-cols-3 gap-3">
                <div className="p-3 rounded-lg bg-zinc-900/30 border border-zinc-800/40 flex flex-col items-center gap-1.5">
                  <Clock size={16} className="text-amber-400" />
                  <span className="text-sm font-medium text-zinc-200">
                    {PRESET_META[preset].time}
                  </span>
                  <span className="text-[11px] text-zinc-500">Est. Time</span>
                </div>
                <div className="p-3 rounded-lg bg-zinc-900/30 border border-zinc-800/40 flex flex-col items-center gap-1.5">
                  <Cpu size={16} className="text-indigo-400" />
                  <span className="text-sm font-medium text-zinc-200">
                    {preset === "maximum"
                      ? "12+ GB"
                      : preset === "balanced"
                      ? "8+ GB"
                      : "4+ GB"}
                  </span>
                  <span className="text-[11px] text-zinc-500">VRAM Req.</span>
                </div>
                <div className="p-3 rounded-lg bg-zinc-900/30 border border-zinc-800/40 flex flex-col items-center gap-1.5">
                  <HardDrive size={16} className="text-emerald-400" />
                  <span className="text-sm font-medium text-zinc-200">
                    {preset === "maximum"
                      ? "~2 GB"
                      : preset === "balanced"
                      ? "~800 MB"
                      : "~300 MB"}
                  </span>
                  <span className="text-[11px] text-zinc-500">Output Size</span>
                </div>
              </div>

              {/* 14 stages preview */}
              <div className="p-4 rounded-xl bg-zinc-900/40 border border-zinc-800/50">
                <div className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-3">
                  Pipeline Stages (14)
                </div>
                <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs text-zinc-500">
                  {[
                    "Frame Extraction",
                    "Color Correction",
                    "Quality Filtering",
                    "IMU Parsing",
                    "Rotation Priors",
                    "COLMAP SfM",
                    "Depth Estimation",
                    "Depth Alignment",
                    "Face Landmarks",
                    "FLAME Fitting",
                    "Face Segmentation",
                    "Gaussian Init",
                    "Gaussian Training",
                    "Export & Render",
                  ].map((name, i) => (
                    <span key={i} className="truncate">
                      {i + 1}. {name}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-6 py-4 border-t border-zinc-800/60 shrink-0">
          <button
            onClick={step === 1 ? handleClose : goBack}
            className="px-4 py-2 rounded-lg text-sm text-zinc-400 hover:text-zinc-200 hover:bg-white/5 transition-colors flex items-center gap-1.5"
          >
            {step === 1 ? (
              "Cancel"
            ) : (
              <>
                <ChevronLeft size={14} /> Back
              </>
            )}
          </button>

          {step < 3 ? (
            <button
              onClick={goNext}
              disabled={step === 1 && !contentDir}
              className={`px-5 py-2 rounded-lg text-sm font-medium flex items-center gap-1.5 transition-all ${
                step === 1 && !contentDir
                  ? "bg-zinc-800 text-zinc-600 cursor-not-allowed"
                  : "bg-indigo-600 text-white hover:bg-indigo-500 shadow-lg shadow-indigo-500/20"
              }`}
            >
              Next <ChevronRight size={14} />
            </button>
          ) : (
            <button
              onClick={handleStart}
              disabled={!contentDir}
              className="px-5 py-2.5 rounded-lg text-sm font-semibold bg-gradient-to-r from-indigo-600 to-purple-600 text-white hover:from-indigo-500 hover:to-purple-500 shadow-lg shadow-indigo-500/25 transition-all flex items-center gap-2"
            >
              <Rocket size={16} />
              Start Reconstruction
            </button>
          )}
        </div>
      </div>

      {/* Keyframe animations injected via style tag */}
      <style>{`
        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        @keyframes scaleIn {
          from { opacity: 0; transform: scale(0.95); }
          to { opacity: 1; transform: scale(1); }
        }
        @keyframes slideInRight {
          from { opacity: 0; transform: translateX(16px); }
          to { opacity: 1; transform: translateX(0); }
        }
        @keyframes slideInLeft {
          from { opacity: 0; transform: translateX(-16px); }
          to { opacity: 1; transform: translateX(0); }
        }
      `}</style>
    </div>
  );
};

export default NewScanWizard;
