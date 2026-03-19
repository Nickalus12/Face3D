import React, { useState, useCallback } from "react";
import {
  X,
  Upload,
  Video,
  Image as ImageIcon,
  FileText,
  Zap,
  Rocket,
  HardDrive,
  Clock,
  Cpu,
  CheckCircle2,
  ChevronDown,
  Camera,
  Radio,
  Sparkles,
} from "lucide-react";
import { selectContentDir } from "../lib/tauri";
import { invoke } from "@tauri-apps/api/core";
import usePipelineStore from "../store/pipelineStore";
import useSessionStore from "../store/sessionStore";
import useToastStore from "../store/toastStore";
import {
  computeAutoPreset,
  type ScanResult,
  type AutoPresetResult,
} from "../lib/autoPreset";

// ── Types ─────────────────────────────────────────────────────

interface NewScanWizardProps {
  isOpen: boolean;
  onClose: () => void;
}

function isTauri(): boolean {
  return typeof window !== "undefined" && !!(window as any).__TAURI_INTERNALS__;
}

// ── Component ─────────────────────────────────────────────────

const NewScanWizard: React.FC<NewScanWizardProps> = ({ isOpen, onClose }) => {
  const [contentDir, setContentDir] = useState<string | null>(null);
  const [scanResult, setScanResult] = useState<ScanResult | null>(null);
  const [autoPreset, setAutoPreset] = useState<AutoPresetResult | null>(null);
  const [sessionName, setSessionName] = useState("");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);
  const [isScanning, setIsScanning] = useState(false);

  const { startPipeline } = usePipelineStore();
  const { createSession } = useSessionStore();
  const { addToast } = useToastStore();

  // Get GPU info from pipeline store for hardware scoring
  const gpuInfo = usePipelineStore((s) => s.gpuInfo);

  // ── Scan directory ────────────────────────────────────────

  const scanDirectory = useCallback(async (dir: string) => {
    setIsScanning(true);
    try {
      let result: ScanResult;
      if (isTauri()) {
        result = await invoke<ScanResult>("scan_content_dir", { path: dir });
      } else {
        // Dev mock
        result = { videos: 2, photos: 39, photos_jpg: 20, photos_dng: 19, sensor_logs: 2, total_bytes: 1_800_000_000, total_size: "1.8 GB", largest_video_mb: 735 };
      }
      setScanResult(result);

      // Compute auto preset based on scan + GPU
      const gpu = gpuInfo ? {
        name: gpuInfo.name,
        vram_total_gb: parseFloat(gpuInfo.memory_total) / 1024,
        vram_available_gb: (parseFloat(gpuInfo.memory_total) - parseFloat(gpuInfo.memory_used)) / 1024,
      } : null;
      const preset = computeAutoPreset(result, gpu);
      setAutoPreset(preset);

      // Auto-generate session name from folder
      const folderName = dir.split(/[/\\]/).filter(Boolean).pop() ?? "";
      const cleanName = folderName.toLowerCase().replace(/[^a-z0-9_-]/g, "_").replace(/_+/g, "_");
      setSessionName(cleanName || `scan_${Date.now().toString(36)}`);
    } catch (e) {
      console.error("Scan failed:", e);
      setScanResult(null);
      setAutoPreset(null);
    }
    setIsScanning(false);
  }, [gpuInfo]);

  const handleBrowse = useCallback(async () => {
    const dir = await selectContentDir();
    if (dir) {
      setContentDir(dir);
      await scanDirectory(dir);
    }
  }, [scanDirectory]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback(() => setIsDragOver(false), []);

  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const items = e.dataTransfer.files;
    if (items.length > 0) {
      const path = (items[0] as any).path ?? items[0].name;
      setContentDir(path);
      await scanDirectory(path);
    }
  }, [scanDirectory]);

  // ── Start pipeline ─────────────────────────────────────────

  const handleStart = async () => {
    if (!contentDir || !autoPreset) return;

    const id = sessionName.trim().replace(/\s+/g, "_").toLowerCase();
    createSession(id, contentDir);

    // Write to DB
    try {
      const { upsertSession } = await import("../lib/database");
      await upsertSession({
        id,
        name: sessionName,
        contentDir,
        photoCount: scanResult?.photos ?? 0,
        sensorLogCount: scanResult?.sensor_logs ?? 0,
      });
    } catch { /* non-fatal */ }

    try {
      await startPipeline(contentDir, id);
      addToast(`Pipeline started — ${autoPreset.label} (~${autoPreset.estimatedMinutes} min)`, "success");
      handleClose();
    } catch {
      addToast("Failed to start pipeline", "error");
    }
  };

  const handleClose = () => {
    setContentDir(null);
    setScanResult(null);
    setAutoPreset(null);
    setSessionName("");
    setShowAdvanced(false);
    onClose();
  };

  if (!isOpen) return null;

  const hasData = scanResult && autoPreset;

  return (
    <div className="fixed inset-0 z-[90] flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/70 backdrop-blur-sm animate-[fadeIn_200ms_ease-out]"
        onClick={handleClose}
      />

      {/* Modal */}
      <div className="relative w-full max-w-[520px] bg-[#111113] border border-zinc-800 rounded-2xl shadow-2xl animate-[scaleIn_250ms_ease-out] flex flex-col max-h-[85vh]">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-zinc-800/60 shrink-0">
          <div>
            <h2 className="text-lg font-semibold text-zinc-100">New Scan</h2>
            <p className="text-xs text-zinc-500 mt-0.5">
              {hasData ? "Ready to start" : "Select your capture data"}
            </p>
          </div>
          <button
            onClick={handleClose}
            className="p-2 rounded-lg text-zinc-500 hover:text-zinc-200 hover:bg-white/5 transition-colors"
          >
            <X size={18} />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto px-6 py-5 space-y-4">

          {/* ── Drop Zone ──────────────────────────────────── */}
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={handleBrowse}
            className={`border-2 border-dashed rounded-xl p-6 flex flex-col items-center gap-3 cursor-pointer transition-all duration-200 ${
              isDragOver
                ? "border-indigo-500 bg-indigo-500/5"
                : contentDir
                ? "border-emerald-500/40 bg-emerald-500/[0.03]"
                : "border-zinc-700 hover:border-zinc-500 bg-zinc-900/30"
            }`}
          >
            {isScanning ? (
              <>
                <div className="w-8 h-8 border-2 border-zinc-700 border-t-indigo-500 rounded-full animate-spin" />
                <p className="text-sm text-zinc-400">Scanning directory...</p>
              </>
            ) : contentDir ? (
              <>
                <CheckCircle2 size={28} className="text-emerald-400" />
                <div className="text-center">
                  <p className="text-sm font-medium text-zinc-200">
                    {contentDir.split(/[/\\]/).pop()}
                  </p>
                  <p className="text-xs text-zinc-500 mt-1">Click to change</p>
                </div>
              </>
            ) : (
              <>
                <Upload size={28} className="text-zinc-500" />
                <div className="text-center">
                  <p className="text-sm font-medium text-zinc-300">
                    Drop folder here or click to browse
                  </p>
                  <p className="text-xs text-zinc-600 mt-1">
                    Select a folder with video, photos, and sensor data
                  </p>
                </div>
              </>
            )}
          </div>

          {/* ── Everything below appears after scan ─────────── */}
          {hasData && (
            <div className="space-y-4 animate-[fadeIn_300ms_ease-out]">

              {/* Detected Content */}
              <div className="grid grid-cols-4 gap-2">
                <ContentStat
                  icon={Video} label="Videos"
                  count={scanResult.videos}
                  color={scanResult.videos > 0 ? "emerald" : "zinc"}
                />
                <ContentStat
                  icon={Camera} label="RAW"
                  count={scanResult.photos_dng}
                  color={scanResult.photos_dng > 0 ? "indigo" : "zinc"}
                />
                <ContentStat
                  icon={ImageIcon} label="Photos"
                  count={scanResult.photos_jpg}
                  color={scanResult.photos_jpg > 0 ? "blue" : "zinc"}
                />
                <ContentStat
                  icon={Radio} label="Sensors"
                  count={scanResult.sensor_logs}
                  color={scanResult.sensor_logs > 0 ? "amber" : "zinc"}
                />
              </div>

              {/* Auto Preset — the main insight */}
              <div className="p-4 rounded-xl bg-indigo-500/[0.04] border border-indigo-500/20">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <Sparkles size={16} className="text-indigo-400" />
                    <span className="text-sm font-semibold text-indigo-300">
                      {autoPreset.label}
                    </span>
                  </div>
                  <div className="flex items-center gap-1">
                    {[1, 2, 3, 4, 5].map((i) => (
                      <div
                        key={i}
                        className={`w-2 h-2 rounded-full ${
                          i <= autoPreset.score ? "bg-indigo-400" : "bg-zinc-700"
                        }`}
                      />
                    ))}
                  </div>
                </div>

                {/* Estimates */}
                <div className="grid grid-cols-3 gap-3 mb-3">
                  <div className="text-center">
                    <Clock size={14} className="text-amber-400 mx-auto mb-1" />
                    <div className="text-sm font-bold text-zinc-200">
                      ~{autoPreset.estimatedMinutes} min
                    </div>
                    <div className="text-[10px] text-zinc-500">Est. Time</div>
                  </div>
                  <div className="text-center">
                    <Cpu size={14} className="text-indigo-400 mx-auto mb-1" />
                    <div className="text-sm font-bold text-zinc-200">
                      {autoPreset.estimatedVramGb} GB
                    </div>
                    <div className="text-[10px] text-zinc-500">VRAM</div>
                  </div>
                  <div className="text-center">
                    <HardDrive size={14} className="text-emerald-400 mx-auto mb-1" />
                    <div className="text-sm font-bold text-zinc-200">
                      {scanResult.total_size}
                    </div>
                    <div className="text-[10px] text-zinc-500">Input</div>
                  </div>
                </div>

                {/* Reasons — why this preset was chosen */}
                <div className="space-y-1">
                  {autoPreset.reasons.slice(0, 4).map((reason, i) => (
                    <p key={i} className={`text-[11px] leading-relaxed ${
                      reason.startsWith("⚠") ? "text-amber-400/80" : "text-zinc-500"
                    }`}>
                      {reason.startsWith("⚠") ? reason : `• ${reason}`}
                    </p>
                  ))}
                </div>
              </div>

              {/* Session Name */}
              <div>
                <label className="text-xs font-medium text-zinc-400 block mb-1.5">
                  Session Name
                </label>
                <input
                  type="text"
                  value={sessionName}
                  onChange={(e) => setSessionName(e.target.value)}
                  className="w-full bg-zinc-800/60 border border-zinc-700/50 rounded-lg px-3 py-2.5 text-sm text-zinc-200 placeholder:text-zinc-500 focus:outline-none focus:border-indigo-500/50 focus:ring-1 focus:ring-indigo-500/20 transition-colors"
                  placeholder="my_scan"
                />
              </div>

              {/* Advanced toggle */}
              <button
                onClick={() => setShowAdvanced(!showAdvanced)}
                className="flex items-center gap-1.5 text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
              >
                <ChevronDown
                  size={12}
                  className={`transition-transform ${showAdvanced ? "rotate-180" : ""}`}
                />
                Advanced settings
              </button>

              {showAdvanced && (
                <div className="p-4 rounded-xl bg-zinc-900/40 border border-zinc-800/50 space-y-3 animate-[fadeIn_200ms_ease-out]">
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="text-[11px] text-zinc-500 block mb-1">Iterations</label>
                      <input
                        type="number"
                        value={autoPreset.config.iterations}
                        readOnly
                        className="w-full bg-zinc-800/40 border border-zinc-700/30 rounded-lg px-3 py-2 text-sm text-zinc-300 font-mono"
                      />
                    </div>
                    <div>
                      <label className="text-[11px] text-zinc-500 block mb-1">Max Gaussians</label>
                      <input
                        type="text"
                        value={autoPreset.config.maxGaussians.toLocaleString()}
                        readOnly
                        className="w-full bg-zinc-800/40 border border-zinc-700/30 rounded-lg px-3 py-2 text-sm text-zinc-300 font-mono"
                      />
                    </div>
                    <div>
                      <label className="text-[11px] text-zinc-500 block mb-1">DA3 Resolution</label>
                      <input
                        type="text"
                        value={`${autoPreset.config.processRes}px`}
                        readOnly
                        className="w-full bg-zinc-800/40 border border-zinc-700/30 rounded-lg px-3 py-2 text-sm text-zinc-300 font-mono"
                      />
                    </div>
                    <div>
                      <label className="text-[11px] text-zinc-500 block mb-1">Max Frames</label>
                      <input
                        type="text"
                        value={autoPreset.config.maxFrames}
                        readOnly
                        className="w-full bg-zinc-800/40 border border-zinc-700/30 rounded-lg px-3 py-2 text-sm text-zinc-300 font-mono"
                      />
                    </div>
                  </div>
                  <p className="text-[10px] text-zinc-600">
                    Settings determined automatically from your hardware + input data.
                  </p>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-zinc-800/60 shrink-0 flex items-center justify-between">
          <button
            onClick={handleClose}
            className="px-4 py-2 text-sm text-zinc-400 hover:text-zinc-200 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleStart}
            disabled={!hasData || !sessionName.trim()}
            className={`flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-semibold transition-all duration-200 active:scale-95 ${
              hasData && sessionName.trim()
                ? "bg-indigo-600 hover:bg-indigo-500 text-white shadow-lg shadow-indigo-500/20"
                : "bg-zinc-800 text-zinc-600 cursor-not-allowed"
            }`}
          >
            <Rocket size={16} />
            Start Pipeline
          </button>
        </div>
      </div>
    </div>
  );
};

// ── Small Components ──────────────────────────────────────────

const ContentStat: React.FC<{
  icon: React.FC<{ size?: number; className?: string }>;
  label: string;
  count: number;
  color: string;
}> = ({ icon: Icon, label, count, color }) => (
  <div className={`p-2.5 rounded-lg border text-center transition-all ${
    count > 0
      ? `bg-${color}-500/[0.05] border-${color}-500/20`
      : "bg-zinc-900/30 border-zinc-800/40"
  }`}>
    <Icon size={14} className={`mx-auto mb-1 ${count > 0 ? `text-${color}-400` : "text-zinc-600"}`} />
    <div className={`text-lg font-bold ${count > 0 ? "text-zinc-200" : "text-zinc-700"}`}>
      {count}
    </div>
    <div className="text-[10px] text-zinc-500">{label}</div>
  </div>
);

export default NewScanWizard;
