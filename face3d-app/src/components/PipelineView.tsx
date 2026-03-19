import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import {
  FolderOpen,
  Film,
  Palette,
  Filter,
  Smartphone,
  Compass,
  Layers,
  Map,
  Maximize,
  ScanFace,
  Box,
  Scissors,
  Sparkles,
  Flame,
  Package,
  Check,
  X,
  AlertTriangle,
  SkipForward,
  Clock,
  Zap,
  Activity,
  ChevronDown,
  Terminal,
  Play,
  type LucideIcon,
} from 'lucide-react';
import usePipelineStore, { type StageStatus } from '../store/pipelineStore';
import PipelineVisualizer from './PipelineVisualizer';

// ── Stage metadata ──────────────────────────────────────────────

interface StageInfo {
  name: string;
  icon: LucideIcon;
  description: string;
  color: string;
}

const STAGE_INFO: Record<number, StageInfo> = {
  0:  { name: 'Organize',    icon: FolderOpen,  description: 'Detecting and organizing video, photos, and sensor data',           color: 'violet' },
  1:  { name: 'Extract',     icon: Film,        description: 'Pulling frames from video — 8K auto-converts to 1080p',             color: 'blue' },
  2:  { name: 'Color',       icon: Palette,     description: 'Converting S-Log3 footage to natural sRGB colors',                  color: 'rose' },
  3:  { name: 'Filter',      icon: Filter,      description: 'Removing blurry and poorly exposed frames',                         color: 'amber' },
  4:  { name: 'Sensors',     icon: Smartphone,   description: 'Parsing gyroscope, accelerometer, and barometer data',              color: 'cyan' },
  5:  { name: 'Orient',      icon: Compass,     description: 'Computing camera orientation from sensor fusion',                   color: 'teal' },
  6:  { name: 'Depth',       icon: Layers,      description: 'Estimating depth and camera poses with Depth Anything 3',           color: 'indigo' },
  7:  { name: 'SfM',         icon: Map,         description: 'Structure from Motion (skipped when DA3 is active)',                 color: 'slate' },
  8:  { name: 'Align',       icon: Maximize,    description: 'Aligning depth to sparse reconstruction',                           color: 'slate' },
  9:  { name: 'Landmarks',   icon: ScanFace,    description: 'Detecting 478 facial landmark points with MediaPipe',               color: 'pink' },
  10: { name: 'FLAME',       icon: Box,         description: 'Fitting parametric face model (5,023 vertices)',                     color: 'orange' },
  11: { name: 'Segment',     icon: Scissors,    description: 'Creating face masks with GrabCut refinement',                        color: 'lime' },
  12: { name: 'Initialize',  icon: Sparkles,    description: 'Placing 30,000 Gaussians on the FLAME mesh',                        color: 'purple' },
  13: { name: 'Train',       icon: Flame,       description: 'Optimizing 2D Gaussian Splatting (3,000 iterations)',                color: 'red' },
  14: { name: 'Export',      icon: Package,     description: 'Generating PLY, mesh, textures, and turntable renders',             color: 'emerald' },
};

// ── Helpers ─────────────────────────────────────────────────────

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  if (m < 60) return `${m}m ${s.toString().padStart(2, '0')}s`;
  const h = Math.floor(m / 60);
  const rm = m % 60;
  return `${h}h ${rm}m`;
}

function formatTime(date: Date): string {
  return date.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

const STAGE_DURATION_ESTIMATES: Record<number, number> = {
  0: 5, 1: 30, 2: 45, 3: 20, 4: 10, 5: 15, 6: 300,
  7: 120, 8: 30, 9: 25, 10: 60, 11: 40, 12: 15, 13: 900, 14: 60,
};

// ── Stage Card ──────────────────────────────────────────────────

const StageCard: React.FC<{
  stage: StageStatus;
  info: StageInfo;
  isActive: boolean;
  logs: string[];
  stageElapsed: number;
  animDelay: number;
}> = ({ stage, info, isActive, logs, stageElapsed, animDelay }) => {
  const Icon = info.icon;
  const logEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (isActive && logEndRef.current) {
      logEndRef.current.scrollTop = logEndRef.current.scrollHeight;
    }
  }, [logs.length, isActive]);

  const statusClasses = (() => {
    switch (stage.status) {
      case 'running':
        return 'border-indigo-500/40 bg-indigo-500/[0.04] shadow-lg shadow-indigo-500/[0.08] animate-pulse-glow-stage';
      case 'complete':
        return 'border-emerald-500/30 bg-emerald-500/[0.03]';
      case 'error':
        return 'border-red-500/40 bg-red-500/[0.04] shadow-lg shadow-red-500/[0.08]';
      case 'skipped':
        return 'border-zinc-800/40 bg-zinc-900/20 opacity-50';
      default:
        return 'border-zinc-800/40 bg-zinc-900/20 opacity-60';
    }
  })();

  const iconColor = (() => {
    switch (stage.status) {
      case 'running': return 'text-indigo-400';
      case 'complete': return 'text-emerald-400';
      case 'error': return 'text-red-400';
      case 'skipped': return 'text-zinc-600 line-through';
      default: return 'text-zinc-600';
    }
  })();

  const StatusIcon = () => {
    switch (stage.status) {
      case 'complete':
        return (
          <div className="absolute -top-1.5 -right-1.5 w-5 h-5 rounded-full bg-emerald-500 flex items-center justify-center animate-check-pop shadow-lg shadow-emerald-500/30">
            <Check size={11} className="text-white" strokeWidth={3} />
          </div>
        );
      case 'error':
        return (
          <div className="absolute -top-1.5 -right-1.5 w-5 h-5 rounded-full bg-red-500 flex items-center justify-center shadow-lg shadow-red-500/30">
            <X size={11} className="text-white" strokeWidth={3} />
          </div>
        );
      case 'skipped':
        return (
          <div className="absolute -top-1.5 -right-1.5 w-5 h-5 rounded-full bg-zinc-700 flex items-center justify-center">
            <SkipForward size={10} className="text-zinc-400" />
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <div
      className={`pipeline-stage-card relative rounded-xl border p-4 transition-all duration-500 ${statusClasses}`}
      style={{ animationDelay: `${animDelay}ms` }}
    >
      <StatusIcon />

      {/* Header */}
      <div className="flex items-center gap-3 mb-2">
        <div className={`p-2 rounded-lg bg-white/[0.03] border border-white/[0.05] ${iconColor} transition-colors duration-300`}>
          <Icon size={18} />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className={`text-sm font-semibold ${
              stage.status === 'running' ? 'text-white' :
              stage.status === 'complete' ? 'text-zinc-200' :
              stage.status === 'error' ? 'text-red-300' :
              'text-zinc-500'
            }`}>
              {info.name}
            </span>
            <span className="text-[11px] text-zinc-600 font-mono">#{stage.id}</span>
          </div>
          <p className="text-xs text-zinc-500 leading-tight mt-0.5 truncate">{info.description}</p>
        </div>

        {/* Duration / ETA */}
        <div className="shrink-0 text-right">
          {stage.status === 'running' && (
            <div className="text-xs text-indigo-400 font-mono tabular-nums">{formatDuration(stageElapsed)}</div>
          )}
          {stage.status === 'complete' && stage.elapsed != null && (
            <div className="text-xs text-emerald-400/70 font-mono tabular-nums">{formatDuration(stage.elapsed)}</div>
          )}
          {stage.status === 'pending' && (
            <div className="text-[11px] text-zinc-700 font-mono">~{formatDuration(STAGE_DURATION_ESTIMATES[stage.id] ?? 60)}</div>
          )}
        </div>
      </div>

      {/* Active stage: progress bar + live logs */}
      {isActive && (
        <div className="mt-3 space-y-3 animate-slideUp">
          {/* Progress shimmer bar */}
          <div className="w-full h-1 bg-zinc-800 rounded-full overflow-hidden">
            <div className="h-full rounded-full bg-gradient-to-r from-indigo-600 via-indigo-400 to-indigo-600 progress-shine" style={{ width: '100%' }} />
          </div>

          {/* Live log stream */}
          {logs.length > 0 && (
            <div
              ref={logEndRef}
              className="max-h-28 overflow-y-auto scrollbar-hide rounded-lg bg-black/30 border border-white/[0.04] p-2.5 space-y-0.5"
            >
              {logs.map((line, i) => (
                <div key={i} className="text-[11px] font-mono text-zinc-500 leading-relaxed truncate hover:text-zinc-300 transition-colors">
                  <span className="text-zinc-700 mr-1.5 select-none">&gt;</span>
                  {line}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// ── Connector Line ──────────────────────────────────────────────

const StageConnector: React.FC<{ status: StageStatus['status'] }> = ({ status }) => {
  const color = status === 'complete'
    ? 'from-emerald-500/40 to-emerald-500/10'
    : status === 'running'
    ? 'from-indigo-500/40 to-indigo-500/10'
    : 'from-zinc-800/60 to-zinc-800/20';

  return (
    <div className="flex justify-center py-1">
      <div className={`w-px h-6 bg-gradient-to-b ${color} transition-colors duration-500`} />
    </div>
  );
};

// ── Overall Progress Ring ───────────────────────────────────────

const ProgressRing: React.FC<{ completed: number; total: number; elapsed: number }> = ({ completed, total, elapsed }) => {
  const progress = total > 0 ? completed / total : 0;
  const circumference = 2 * Math.PI * 42;
  const strokeDashoffset = circumference * (1 - progress);

  return (
    <div className="relative w-24 h-24">
      <svg className="w-24 h-24 -rotate-90" viewBox="0 0 96 96">
        {/* Track */}
        <circle cx="48" cy="48" r="42" fill="none" stroke="rgba(39,39,42,0.5)" strokeWidth="3" />
        {/* Progress */}
        <circle
          cx="48" cy="48" r="42" fill="none"
          stroke="url(#progressGradient)"
          strokeWidth="3"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          className="transition-all duration-700 ease-out"
        />
        <defs>
          <linearGradient id="progressGradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#818cf8" />
            <stop offset="50%" stopColor="#6366f1" />
            <stop offset="100%" stopColor="#10b981" />
          </linearGradient>
        </defs>
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-lg font-bold text-white tabular-nums">{Math.round(progress * 100)}%</span>
        <span className="text-[9px] text-zinc-500 font-mono">{formatDuration(elapsed)}</span>
      </div>
    </div>
  );
};

// ── Training Live Panel ─────────────────────────────────────────

const TrainingPanel: React.FC = () => {
  const { metrics, gpuInfo } = usePipelineStore();
  const latestMetric = metrics.length > 0 ? metrics[metrics.length - 1] : null;

  if (!latestMetric) return null;

  const lossPoints = metrics.filter(m => m.loss != null).slice(-50);
  const maxLoss = Math.max(...lossPoints.map(m => m.loss!), 0.001);
  const minLoss = Math.min(...lossPoints.map(m => m.loss!), 0);

  return (
    <div className="mt-4 p-4 rounded-xl bg-zinc-900/40 border border-zinc-800/50 space-y-3">
      <div className="flex items-center gap-2 mb-1">
        <Activity size={14} className="text-red-400" />
        <span className="text-xs font-semibold text-zinc-300 uppercase tracking-wider">Training Live</span>
      </div>

      {/* Metrics grid */}
      <div className="grid grid-cols-4 gap-2">
        <MetricBadge label="Iteration" value={latestMetric.iter?.toLocaleString() ?? '--'} color="indigo" />
        <MetricBadge label="Loss" value={latestMetric.loss?.toFixed(4) ?? '--'} color="amber" />
        <MetricBadge label="PSNR" value={latestMetric.psnr ? `${latestMetric.psnr.toFixed(1)} dB` : '--'} color="emerald" />
        <MetricBadge label="Gaussians" value={
          latestMetric.gaussians
            ? latestMetric.gaussians >= 1000
              ? `${Math.round(latestMetric.gaussians / 1000)}K`
              : latestMetric.gaussians.toString()
            : '--'
        } color="purple" />
      </div>

      {/* Mini loss chart */}
      {lossPoints.length > 1 && (
        <div className="h-16 flex items-end gap-px">
          {lossPoints.map((m, i) => {
            const normalizedHeight = ((m.loss! - minLoss) / (maxLoss - minLoss)) * 100;
            const height = Math.max(2, 100 - normalizedHeight); // invert: lower loss = taller bar
            return (
              <div
                key={i}
                className="flex-1 bg-gradient-to-t from-indigo-500/60 to-indigo-400/20 rounded-t-sm transition-all duration-300"
                style={{ height: `${height}%` }}
              />
            );
          })}
        </div>
      )}

      {/* GPU info */}
      {gpuInfo && (
        <div className="flex items-center gap-4 text-[11px] text-zinc-500 pt-1 border-t border-zinc-800/30">
          <span className="flex items-center gap-1">
            <Zap size={10} className="text-amber-400" />
            GPU {gpuInfo.utilization ?? '--'}%
          </span>
          <span>VRAM {gpuInfo.memory_used ? `${(parseFloat(gpuInfo.memory_used) / 1024).toFixed(1)}` : '--'}/{gpuInfo.memory_total ? `${(parseFloat(gpuInfo.memory_total) / 1024).toFixed(0)}` : '--'} GB</span>
        </div>
      )}
    </div>
  );
};

const MetricBadge: React.FC<{ label: string; value: string; color: string }> = ({ label, value, color }) => (
  <div className={`p-2 rounded-lg bg-${color}-500/[0.05] border border-${color}-500/10`}>
    <div className="text-[9px] text-zinc-500 uppercase tracking-wider mb-0.5">{label}</div>
    <div className={`text-sm font-semibold text-${color}-400 tabular-nums`}>{value}</div>
  </div>
);

// ── Idle State ──────────────────────────────────────────────────

const IdleState: React.FC = () => {
  const stageEntries = Object.entries(STAGE_INFO).map(([k, v]) => ({ num: Number(k), ...v }));

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="max-w-2xl mx-auto px-8 py-10">
        {/* Header */}
        <div className="mb-8">
          <h2 className="text-xl font-bold text-zinc-100 mb-2">Pipeline</h2>
          <p className="text-sm text-zinc-500">
            Select a session and start a scan, or re-run the pipeline. 15 stages from video to 3D Gaussian Splat.
          </p>
        </div>

        {/* Stage list — clean vertical layout with generous spacing */}
        <div className="space-y-1">
          {stageEntries.map((stage) => {
            const Icon = stage.icon;
            return (
              <div key={stage.num} className="flex items-center gap-4 py-3 px-4 rounded-xl hover:bg-zinc-800/20 transition-colors group">
                <div className="w-8 h-8 rounded-lg bg-zinc-800/50 border border-zinc-700/30 flex items-center justify-center shrink-0 group-hover:border-zinc-600/50 transition-colors">
                  <Icon size={15} className="text-zinc-500 group-hover:text-zinc-400 transition-colors" />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-mono text-zinc-600 w-5">{stage.num}</span>
                    <span className="text-sm font-medium text-zinc-300">{stage.name}</span>
                  </div>
                </div>
                <span className="text-xs text-zinc-700 hidden sm:block max-w-[200px] truncate">{stage.description}</span>
              </div>
            );
          })}
        </div>

        {/* Legend */}
        <div className="mt-8 pt-6 border-t border-zinc-800/40 flex items-center gap-5 text-xs text-zinc-600">
          <div className="flex items-center gap-1.5">
            <div className="w-2 h-2 rounded-full bg-zinc-700" />
            Pending
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-2 h-2 rounded-full bg-indigo-500" />
            Active
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-2 h-2 rounded-full bg-emerald-500" />
            Complete
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-2 h-2 rounded-full bg-red-500" />
            Error
          </div>
        </div>
      </div>
    </div>
  );
};

// ── Complete State ──────────────────────────────────────────────

const CompleteState: React.FC<{ elapsed: number; stageCount: number }> = ({ elapsed, stageCount }) => (
  <div className="mb-6 p-5 rounded-xl bg-emerald-500/[0.04] border border-emerald-500/20 animate-scaleIn relative overflow-hidden">
    {/* Confetti dots */}
    <div className="absolute inset-0 pointer-events-none overflow-hidden">
      {[...Array(16)].map((_, i) => (
        <div
          key={i}
          className="absolute w-1 h-1 rounded-full"
          style={{
            left: `${5 + i * 6}%`,
            top: `${15 + (i % 4) * 20}%`,
            background: ['#10b981', '#6366f1', '#f59e0b', '#3b82f6', '#a78bfa', '#ec4899', '#14b8a6', '#f43f5e'][i % 8],
            opacity: 0.7,
            animation: `confetti-dot 2.5s ease-out ${i * 0.08}s infinite`,
          }}
        />
      ))}
    </div>
    <div className="relative text-center">
      <div className="inline-flex items-center gap-2 mb-2">
        <div className="w-8 h-8 rounded-full bg-emerald-500/20 flex items-center justify-center animate-check-pop">
          <Check size={18} className="text-emerald-400" strokeWidth={3} />
        </div>
      </div>
      <h3 className="text-lg font-bold text-emerald-300">Pipeline Complete</h3>
      <p className="text-xs text-zinc-500 mt-1">
        {stageCount} stages finished in {formatDuration(elapsed)}
      </p>
    </div>
  </div>
);

// ── Error State ─────────────────────────────────────────────────

const ErrorBanner: React.FC = () => (
  <div className="mb-4 p-4 rounded-xl bg-red-500/[0.04] border border-red-500/20 flex items-center gap-3">
    <AlertTriangle size={20} className="text-red-400 shrink-0" />
    <div>
      <h3 className="text-sm font-semibold text-red-300">Pipeline Error</h3>
      <p className="text-xs text-zinc-500 mt-0.5">Check the console logs for details.</p>
    </div>
  </div>
);

// ── Main PipelineView ───────────────────────────────────────────

const PipelineView: React.FC = () => {
  const { status, stages, logs, metrics, startedAt, currentStage, sessionName } = usePipelineStore();
  const [elapsed, setElapsed] = useState(0);
  const [stageElapsed, setStageElapsed] = useState(0);
  const [stageStartTime, setStageStartTime] = useState<number | null>(null);
  const [showTimeline, setShowTimeline] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const prevStageRef = useRef(currentStage);

  const isRunning = status === 'running' || status === 'stopping';
  const isComplete = status === 'complete';
  const isError = status === 'error';
  const isIdle = status === 'idle';

  // Track elapsed time
  useEffect(() => {
    if (!startedAt || !isRunning) return;
    const tick = setInterval(() => setElapsed(Math.floor((Date.now() - startedAt) / 1000)), 1000);
    return () => clearInterval(tick);
  }, [startedAt, isRunning]);

  // Track stage elapsed
  useEffect(() => {
    if (isRunning && currentStage > 0) {
      if (prevStageRef.current !== currentStage) {
        setStageStartTime(Date.now());
        prevStageRef.current = currentStage;
      }
    }
  }, [currentStage, isRunning]);

  useEffect(() => {
    if (!stageStartTime || !isRunning) return;
    const tick = setInterval(() => setStageElapsed(Math.floor((Date.now() - stageStartTime) / 1000)), 1000);
    return () => clearInterval(tick);
  }, [stageStartTime, isRunning]);

  // Staggered reveal on start
  useEffect(() => {
    if (isRunning && !showTimeline) {
      const timer = setTimeout(() => setShowTimeline(true), 300);
      return () => clearTimeout(timer);
    }
    if (isIdle) {
      setShowTimeline(false);
    }
  }, [isRunning, isIdle, showTimeline]);

  // Initialize stage start time when pipeline starts
  useEffect(() => {
    if (isRunning && !stageStartTime) {
      setStageStartTime(Date.now());
    }
    if (isIdle) {
      setStageStartTime(null);
      setStageElapsed(0);
      setElapsed(0);
    }
  }, [isRunning, isIdle]);

  // Set elapsed for complete state
  useEffect(() => {
    if (isComplete && startedAt) {
      setElapsed(Math.floor((Date.now() - startedAt) / 1000));
    }
  }, [isComplete, startedAt]);

  // Auto-scroll to active stage
  useEffect(() => {
    if (!scrollRef.current || !isRunning) return;
    const activeEl = scrollRef.current.querySelector('[data-stage-active="true"]');
    if (activeEl) {
      activeEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, [currentStage, isRunning]);

  // Get stage-specific logs (last N lines matching current stage context)
  const getStageLogs = useCallback((stageId: number): string[] => {
    // Find the index of the log that starts this stage
    let startIdx = 0;
    let endIdx = logs.length;
    for (let i = logs.length - 1; i >= 0; i--) {
      const msg = logs[i].message;
      const stageMatch = msg.match(/(?:===\s*Stage|Running\s+stage)\s+(\d+)/i);
      if (stageMatch) {
        const sn = parseInt(stageMatch[1], 10);
        if (sn === stageId) {
          startIdx = i;
          break;
        }
        if (sn > stageId) {
          endIdx = i;
        }
      }
    }
    return logs.slice(startIdx, endIdx).map(l => l.message).slice(-8);
  }, [logs]);

  // ETA calculation
  const estimatedRemaining = useMemo(() => {
    if (!isRunning) return null;
    let remaining = 0;
    for (const stage of stages) {
      if (stage.status === 'pending') {
        remaining += STAGE_DURATION_ESTIMATES[stage.id] ?? 60;
      }
      if (stage.status === 'running') {
        const est = STAGE_DURATION_ESTIMATES[stage.id] ?? 60;
        remaining += Math.max(0, est - stageElapsed);
      }
    }
    return remaining;
  }, [stages, isRunning, stageElapsed]);

  const completedCount = stages.filter(s => s.status === 'complete').length;

  if (isIdle && !isComplete && !isError) {
    return (
      <div className="h-full flex flex-col bg-[#0a0a0b]">
        <IdleState />
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col bg-[#0a0a0b] overflow-hidden">
      {/* Header */}
      <div className="shrink-0 px-6 pt-5 pb-4 border-b border-zinc-800/40">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-lg font-bold text-zinc-100 flex items-center gap-2.5">
              Pipeline
              {isRunning && (
                <span className="relative flex h-2.5 w-2.5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-60" />
                  <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-indigo-500" />
                </span>
              )}
              {isComplete && <Check size={18} className="text-emerald-400" />}
              {isError && <AlertTriangle size={18} className="text-red-400" />}
            </h1>
            <p className="text-xs text-zinc-500 mt-0.5">
              {sessionName && <span className="text-zinc-400">{sessionName}</span>}
              {sessionName && ' — '}
              {isRunning ? `Stage ${currentStage} of ${stages.length}` :
               isComplete ? 'All stages finished' :
               isError ? 'Stopped with errors' : 'Ready'}
            </p>
          </div>

          {/* Progress Ring */}
          {(isRunning || isComplete) && (
            <ProgressRing completed={completedCount} total={stages.length} elapsed={elapsed} />
          )}
        </div>

        {/* ETA bar */}
        {isRunning && estimatedRemaining != null && (
          <div className="flex items-center gap-3 mt-3 text-[11px] text-zinc-500">
            <Clock size={10} />
            <span>ETA ~{formatDuration(estimatedRemaining)}</span>
            <div className="flex-1 h-px bg-zinc-800/60" />
            <span className="text-zinc-600">{formatTime(new Date())}</span>
          </div>
        )}
      </div>

      {/* Timeline */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-6 py-5 scrollbar-hide">
        {/* Live 3D visualizer during pipeline run */}
        {isRunning && (
          <div className="mb-4">
            <PipelineVisualizer completedStages={completedCount} compact className="max-w-[200px] mx-auto" />
          </div>
        )}
        {isComplete && <CompleteState elapsed={elapsed} stageCount={completedCount} />}
        {isError && <ErrorBanner />}

        <div className={`space-y-0 ${showTimeline || isComplete || isError ? 'pipeline-timeline-enter' : ''}`}>
          {stages.map((stage, index) => {
            const info = STAGE_INFO[stage.id] ?? { name: stage.name, icon: Box, description: '', color: 'zinc' };
            const isActive = stage.status === 'running';
            const stageLogs = isActive ? getStageLogs(stage.id) : [];

            return (
              <div key={stage.id} data-stage-active={isActive ? 'true' : 'false'}>
                <div
                  className="pipeline-stage-item"
                  style={{ animationDelay: `${index * 60}ms` }}
                >
                  <StageCard
                    stage={stage}
                    info={info}
                    isActive={isActive}
                    logs={stageLogs}
                    stageElapsed={isActive ? stageElapsed : (stage.elapsed ?? 0)}
                    animDelay={index * 60}
                  />
                </div>

                {/* Training live panel */}
                {isActive && stage.id === 13 && metrics.length > 0 && <TrainingPanel />}

                {/* Connector */}
                {index < stages.length - 1 && (
                  <StageConnector status={stage.status} />
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Bottom mini console peek */}
      {isRunning && logs.length > 0 && (
        <div className="shrink-0 border-t border-zinc-800/40 bg-zinc-900/30 px-5 py-2.5">
          <div className="flex items-center gap-2 mb-1.5">
            <Terminal size={10} className="text-zinc-600" />
            <span className="text-[9px] text-zinc-600 uppercase tracking-wider font-semibold">Latest</span>
          </div>
          <div className="text-[11px] font-mono text-zinc-500 truncate">
            {logs[logs.length - 1]?.message ?? ''}
          </div>
        </div>
      )}
    </div>
  );
};

export default PipelineView;
