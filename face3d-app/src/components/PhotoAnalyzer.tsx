import { useState, useCallback, useRef } from 'react';
import {
  Upload,
  ScanSearch,
  User,
  Ruler,
  Sparkles,
  Camera,
  AlertCircle,
  CheckCircle2,
  Loader2,
  X,
} from 'lucide-react';
import { analyzePhoto, type PhotoAnalysisResult } from '../lib/api';
import { isTauri } from '../lib/api';
import { convertFileSrc } from '@tauri-apps/api/core';

function qualityLabel(score: number): { label: string; color: string } {
  if (score >= 90) return { label: 'A+', color: 'text-emerald-400' };
  if (score >= 80) return { label: 'B+', color: 'text-emerald-400' };
  if (score >= 70) return { label: 'B', color: 'text-yellow-400' };
  if (score >= 60) return { label: 'C+', color: 'text-yellow-400' };
  if (score >= 50) return { label: 'C', color: 'text-orange-400' };
  return { label: 'D', color: 'text-red-400' };
}

function metricStatus(value: number, good: [number, number]): 'good' | 'ok' | 'bad' {
  if (value >= good[0] && value <= good[1]) return 'good';
  const margin = (good[1] - good[0]) * 0.3;
  if (value >= good[0] - margin && value <= good[1] + margin) return 'ok';
  return 'bad';
}

function StatusDot({ status }: { status: 'good' | 'ok' | 'bad' }) {
  const colors = {
    good: 'bg-emerald-400',
    ok: 'bg-yellow-400',
    bad: 'bg-red-400',
  };
  return <div className={`w-1.5 h-1.5 rounded-full ${colors[status]}`} />;
}

export default function PhotoAnalyzer() {
  const [analysis, setAnalysis] = useState<PhotoAnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [imagePath, setImagePath] = useState<string | null>(null);
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleAnalyze = useCallback(async (filePath: string) => {
    setLoading(true);
    setError(null);
    setAnalysis(null);
    setImagePath(filePath);

    if (isTauri()) {
      try {
        setImageSrc(convertFileSrc(filePath));
      } catch {
        setImageSrc(null);
      }
    }

    const result = await analyzePhoto(filePath);
    setLoading(false);

    if (!result) {
      setError('Analysis failed -- check that the file is a valid image.');
      return;
    }
    if (!result.detected) {
      setError('No face detected in this image.');
    }
    setAnalysis(result);
  }, []);

  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);

    // In Tauri, dropped files come through the dataTransfer
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      const file = files[0];
      // Tauri gives us the full path via webkitRelativePath or the name
      // For Tauri 2.0, file paths from drag & drop need special handling
      const path = (file as any).path || file.name;
      if (path) {
        handleAnalyze(path);
      }
    }
  }, [handleAnalyze]);

  const handleFileSelect = useCallback(() => {
    if (isTauri()) {
      // Use Tauri dialog for file selection
      import('@tauri-apps/plugin-dialog').then(({ open }) => {
        open({
          multiple: false,
          filters: [{ name: 'Images', extensions: ['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff', 'dng'] }],
        }).then((selected) => {
          if (selected && typeof selected === 'string') {
            handleAnalyze(selected);
          } else if (selected && Array.isArray(selected) && selected.length > 0) {
            handleAnalyze(selected[0]);
          }
        });
      });
    } else {
      fileInputRef.current?.click();
    }
  }, [handleAnalyze]);

  const handleInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const url = URL.createObjectURL(file);
      setImageSrc(url);
      // In dev mode we can't actually analyze -- mock will be used
      handleAnalyze(file.name);
    }
  }, [handleAnalyze]);

  const a = analysis;
  const q = a ? qualityLabel(a.quality_score) : null;

  return (
    <div className="h-full flex flex-col bg-[#0a0a0b] overflow-hidden">
      {/* Header */}
      <div className="shrink-0 px-6 py-4 border-b border-zinc-800/40 flex items-center gap-3">
        <ScanSearch size={20} className="text-indigo-400" />
        <div>
          <h1 className="text-sm font-semibold text-zinc-100">Photo Analyzer</h1>
          <p className="text-xs text-zinc-500">Extract face data, quality metrics, and identity from a single photo</p>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto scrollbar-hide">
        {!analysis && !loading ? (
          /* Drop zone */
          <div className="flex items-center justify-center h-full p-8">
            <div
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
              onClick={handleFileSelect}
              className={`w-full max-w-lg aspect-[4/3] rounded-2xl border-2 border-dashed cursor-pointer transition-all duration-200 flex flex-col items-center justify-center gap-4 ${
                dragOver
                  ? 'border-indigo-500 bg-indigo-500/5 scale-[1.01]'
                  : 'border-zinc-700/50 bg-zinc-900/30 hover:border-zinc-600 hover:bg-zinc-900/50'
              }`}
            >
              <div className={`p-4 rounded-2xl transition-colors ${dragOver ? 'bg-indigo-500/10' : 'bg-zinc-800/50'}`}>
                <Upload size={32} className={dragOver ? 'text-indigo-400' : 'text-zinc-500'} />
              </div>
              <div className="text-center">
                <p className="text-sm text-zinc-300 font-medium">Drop a photo here or click to browse</p>
                <p className="text-xs text-zinc-600 mt-1">JPG, PNG, DNG, TIFF supported</p>
              </div>
              {error && (
                <div className="flex items-center gap-2 px-3 py-2 bg-red-500/10 border border-red-500/20 rounded-lg">
                  <AlertCircle size={14} className="text-red-400" />
                  <span className="text-xs text-red-300">{error}</span>
                </div>
              )}
            </div>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={handleInputChange}
            />
          </div>
        ) : loading ? (
          /* Loading state */
          <div className="flex flex-col items-center justify-center h-full gap-4">
            <div className="relative">
              <Loader2 size={40} className="text-indigo-400 animate-spin" />
            </div>
            <div className="text-center">
              <p className="text-sm text-zinc-300">Analyzing photo...</p>
              <p className="text-xs text-zinc-600 mt-1">Running InsightFace detection + quality metrics</p>
            </div>
          </div>
        ) : a ? (
          /* Results */
          <div className="p-6">
            {/* Back button */}
            <button
              onClick={() => { setAnalysis(null); setError(null); setImagePath(null); setImageSrc(null); }}
              className="mb-4 flex items-center gap-1.5 text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
            >
              <X size={14} /> Analyze another photo
            </button>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Left column -- photo preview */}
              <div className="space-y-4">
                {/* Photo with overlay */}
                <div className="relative rounded-xl overflow-hidden bg-zinc-900 border border-zinc-800/40">
                  {imageSrc ? (
                    <img src={imageSrc} alt="Analyzed photo" className="w-full object-contain max-h-[500px]" />
                  ) : (
                    <div className="w-full aspect-[4/3] flex items-center justify-center text-zinc-600">
                      <Camera size={48} />
                    </div>
                  )}
                  {/* Detection overlay */}
                  {a.detected && (
                    <div className="absolute top-3 left-3 flex items-center gap-1.5 px-2 py-1 bg-emerald-500/20 border border-emerald-500/30 rounded-lg backdrop-blur-sm">
                      <CheckCircle2 size={12} className="text-emerald-400" />
                      <span className="text-[11px] text-emerald-300 font-medium">
                        Face detected ({(a.detection_score * 100).toFixed(0)}%)
                      </span>
                    </div>
                  )}
                  {!a.detected && (
                    <div className="absolute top-3 left-3 flex items-center gap-1.5 px-2 py-1 bg-red-500/20 border border-red-500/30 rounded-lg backdrop-blur-sm">
                      <AlertCircle size={12} className="text-red-400" />
                      <span className="text-[11px] text-red-300 font-medium">No face detected</span>
                    </div>
                  )}
                </div>

                {/* File info */}
                <div className="px-3 py-2 bg-zinc-900/50 rounded-lg border border-zinc-800/30">
                  <p className="text-xs text-zinc-500 truncate">{imagePath}</p>
                  <p className="text-[11px] text-zinc-600 mt-0.5">
                    Analysis time: {a.analysis_time_ms.toFixed(0)}ms
                  </p>
                </div>
              </div>

              {/* Right column -- analysis results */}
              <div className="space-y-4">
                {a.detected && (
                  <>
                    {/* Identity */}
                    <Section icon={User} title="Identity">
                      <div className="grid grid-cols-2 gap-x-4 gap-y-2">
                        {a.identity_match ? (
                          <MetricRow label="Match" value={`${a.identity_match} (${(a.identity_confidence * 100).toFixed(0)}%)`} />
                        ) : (
                          <MetricRow label="Match" value="New face" valueClass="text-amber-400" />
                        )}
                        <MetricRow label="Age" value={String(a.age)} />
                        <MetricRow label="Gender" value={a.gender === 'F' ? 'Female' : 'Male'} />
                        <MetricRow
                          label="Angle"
                          value={`${a.estimated_angle} (${a.head_pose.yaw > 0 ? '+' : ''}${a.head_pose.yaw.toFixed(1)} yaw)`}
                        />
                      </div>
                    </Section>

                    {/* Head Pose */}
                    <Section icon={Ruler} title="Head Pose">
                      <div className="grid grid-cols-3 gap-3">
                        <PoseCard label="Yaw" value={a.head_pose.yaw} />
                        <PoseCard label="Pitch" value={a.head_pose.pitch} />
                        <PoseCard label="Roll" value={a.head_pose.roll} />
                      </div>
                    </Section>

                    {/* Quality Metrics */}
                    <Section icon={Sparkles} title="Quality">
                      <div className="flex items-center gap-3 mb-3">
                        <div className={`text-3xl font-bold tabular-nums ${q!.color}`}>
                          {a.quality_score.toFixed(0)}
                        </div>
                        <div>
                          <div className={`text-sm font-semibold ${q!.color}`}>{q!.label}</div>
                          <div className="text-[11px] text-zinc-500">Overall score</div>
                        </div>
                        {/* Score bar */}
                        <div className="flex-1 h-2 bg-zinc-800 rounded-full overflow-hidden ml-2">
                          <div
                            className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-emerald-400 transition-all duration-500"
                            style={{ width: `${a.quality_score}%` }}
                          />
                        </div>
                      </div>

                      <div className="grid grid-cols-2 gap-x-4 gap-y-1.5">
                        <MetricRowDot label="Sharpness" value={a.sharpness.toFixed(0)} status={metricStatus(a.sharpness, [80, 300])} />
                        <MetricRowDot label="Brightness" value={a.brightness.toFixed(0)} status={metricStatus(a.brightness, [80, 180])} />
                        <MetricRowDot label="Contrast" value={a.contrast.toFixed(0)} status={metricStatus(a.contrast, [30, 80])} />
                        <MetricRowDot label="Noise" value={a.noise_level.toFixed(0)} status={metricStatus(a.noise_level, [0, 25])} />
                        <MetricRow label="Color" value={a.color_temperature} />
                        <MetricRow label="Face area" value={`${(a.face_area_ratio * 100).toFixed(1)}%`} />
                      </div>
                    </Section>

                    {/* Recommendations */}
                    {a.recommendations.length > 0 && (
                      <Section icon={AlertCircle} title="Recommendations">
                        <ul className="space-y-1.5">
                          {a.recommendations.map((rec, i) => (
                            <li key={i} className="flex items-start gap-2 text-xs text-zinc-400">
                              <span className="text-indigo-400 mt-0.5 shrink-0">--</span>
                              {rec}
                            </li>
                          ))}
                        </ul>
                      </Section>
                    )}
                  </>
                )}

                {!a.detected && (
                  <div className="flex flex-col items-center justify-center py-12 text-center">
                    <AlertCircle size={32} className="text-red-400/50 mb-3" />
                    <p className="text-sm text-zinc-400">No face detected in this image</p>
                    <p className="text-xs text-zinc-600 mt-1">Ensure the face is visible, well-lit, and not too small in the frame</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}

function Section({ icon: Icon, title, children }: { icon: any; title: string; children: React.ReactNode }) {
  return (
    <div className="bg-zinc-900/40 border border-zinc-800/30 rounded-xl p-4">
      <div className="flex items-center gap-2 mb-3">
        <Icon size={14} className="text-indigo-400" />
        <span className="text-xs font-semibold text-zinc-300 uppercase tracking-wider">{title}</span>
      </div>
      {children}
    </div>
  );
}

function MetricRow({ label, value, valueClass }: { label: string; value: string; valueClass?: string }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-xs text-zinc-500">{label}</span>
      <span className={`text-xs font-medium tabular-nums ${valueClass || 'text-zinc-300'}`}>{value}</span>
    </div>
  );
}

function MetricRowDot({ label, value, status }: { label: string; value: string; status: 'good' | 'ok' | 'bad' }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-xs text-zinc-500">{label}</span>
      <div className="flex items-center gap-1.5">
        <StatusDot status={status} />
        <span className="text-xs font-medium tabular-nums text-zinc-300">{value}</span>
      </div>
    </div>
  );
}

function PoseCard({ label, value }: { label: string; value: number }) {
  const abs = Math.abs(value);
  const color = abs < 10 ? 'text-emerald-400' : abs < 30 ? 'text-yellow-400' : 'text-orange-400';
  return (
    <div className="bg-zinc-800/40 rounded-lg p-2.5 text-center">
      <div className={`text-lg font-bold tabular-nums ${color}`}>
        {value > 0 ? '+' : ''}{value.toFixed(1)}
      </div>
      <div className="text-[10px] text-zinc-500 uppercase tracking-wider mt-0.5">{label}</div>
    </div>
  );
}
