import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import {
  Check,
  AlertTriangle,
  Clock,
  Zap,
  Terminal,
  FolderOpen,
  Eye,
} from 'lucide-react';
import usePipelineStore, { type StageStatus } from '../store/pipelineStore';
import useSessionStore from '../store/sessionStore';
import { getSessionSummaryFromDb } from '../lib/dbBridge';

// ── Stage metadata ──────────────────────────────────────────────

const STAGE_NAMES: Record<number, string> = {
  0: 'Organize',
  1: 'Extract',
  2: 'Color',
  3: 'Filter',
  4: 'Sensors',
  5: 'Orient',
  6: 'Depth',
  7: 'SfM',
  8: 'Align',
  9: 'Landmarks',
  10: 'FLAME',
  11: 'Segment',
  12: 'Initialize',
  13: 'Train',
  14: 'Export',
};

const STAGE_DESCRIPTIONS: Record<number, string> = {
  0: 'Detecting and organizing video, photos, and sensor data',
  1: 'Pulling frames from video — 8K auto-converts to 1080p',
  2: 'Converting S-Log3 footage to natural sRGB colors',
  3: 'Removing blurry and poorly exposed frames',
  4: 'Parsing gyroscope, accelerometer, and barometer data',
  5: 'Computing camera orientation from sensor fusion',
  6: 'Estimating depth and camera poses with Depth Anything 3',
  7: 'Structure from Motion (COLMAP)',
  8: 'Aligning depth to sparse reconstruction',
  9: 'Detecting 478 facial landmark points with MediaPipe',
  10: 'Fitting parametric face model (5,023 vertices)',
  11: 'Creating face masks with GrabCut refinement',
  12: 'Placing 30,000 Gaussians on the FLAME mesh',
  13: 'Optimizing 2D Gaussian Splatting (3,000 iterations)',
  14: 'Generating PLY, mesh, textures, and turntable renders',
};

const STAGE_DURATION_ESTIMATES: Record<number, number> = {
  0: 5, 1: 30, 2: 45, 3: 20, 4: 10, 5: 15, 6: 300,
  7: 120, 8: 30, 9: 25, 10: 60, 11: 40, 12: 15, 13: 900, 14: 60,
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

// ── Progress Ring ───────────────────────────────────────────────

const ProgressRing: React.FC<{ completed: number; total: number; elapsed: number }> = ({ completed, total, elapsed }) => {
  const progress = total > 0 ? completed / total : 0;
  const circumference = 2 * Math.PI * 42;
  const strokeDashoffset = circumference * (1 - progress);

  return (
    <div className="relative w-20 h-20 shrink-0">
      <svg className="w-20 h-20 -rotate-90" viewBox="0 0 96 96">
        <circle cx="48" cy="48" r="42" fill="none" stroke="rgba(39,39,42,0.5)" strokeWidth="3" />
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
            <stop offset="100%" stopColor="#10b981" />
          </linearGradient>
        </defs>
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-base font-bold text-white tabular-nums">{Math.round(progress * 100)}%</span>
        <span className="text-[10px] text-zinc-500 font-mono tabular-nums">{formatDuration(elapsed)}</span>
      </div>
    </div>
  );
};

// ── Training Metrics (inline in active stage card for stage 13) ─

const TrainingMetrics: React.FC = () => {
  const { metrics, gpuInfo } = usePipelineStore();
  const latest = metrics.length > 0 ? metrics[metrics.length - 1] : null;
  if (!latest) return null;

  return (
    <div className="mt-2 flex flex-wrap gap-3 text-[11px] font-mono">
      {latest.iter != null && (
        <span className="text-zinc-500">iter <span className="text-indigo-400">{latest.iter.toLocaleString()}</span></span>
      )}
      {latest.loss != null && (
        <span className="text-zinc-500">loss <span className="text-amber-400">{latest.loss.toFixed(4)}</span></span>
      )}
      {latest.psnr != null && (
        <span className="text-zinc-500">psnr <span className="text-emerald-400">{latest.psnr.toFixed(1)} dB</span></span>
      )}
      {latest.gaussians != null && (
        <span className="text-zinc-500">gs <span className="text-purple-400">
          {latest.gaussians >= 1000 ? `${Math.round(latest.gaussians / 1000)}K` : latest.gaussians}
        </span></span>
      )}
      {gpuInfo && gpuInfo.utilization != null && (
        <span className="text-zinc-500 flex items-center gap-1">
          <Zap size={9} className="text-amber-400" />
          <span className="text-zinc-400">{gpuInfo.utilization}%</span>
        </span>
      )}
    </div>
  );
};

// ── Idle State ──────────────────────────────────────────────────

interface LastRunSummary {
  duration: number;
  grade: string | null;
  gaussians: number | null;
}

const IdleState: React.FC = () => {
  const currentSession = useSessionStore(s => s.currentSession);
  const [lastRun, setLastRun] = useState<LastRunSummary | null>(null);

  useEffect(() => {
    if (!currentSession) {
      setLastRun(null);
      return;
    }
    let cancelled = false;
    getSessionSummaryFromDb(currentSession.id).then(summary => {
      if (cancelled || !summary) return;
      const stages = summary.stages as Array<{ duration_s?: number; status?: string }> | undefined;
      const totalDuration = stages?.reduce((sum: number, s: { duration_s?: number }) => sum + (s.duration_s ?? 0), 0) ?? 0;
      if (totalDuration === 0) return; // no previous run
      const grade = summary.session?.quality_grade ?? null;
      const gaussians = summary.training?.num_gaussians ?? summary.training?.gaussians ?? null;
      setLastRun({ duration: totalDuration, grade, gaussians });
    });
    return () => { cancelled = true; };
  }, [currentSession?.id]);

  return (
    <div className="h-full flex flex-col items-center justify-center px-8">
      <div className="max-w-md text-center">
        <h2 className="text-lg font-semibold text-zinc-100 mb-2">Pipeline</h2>
        <p className="text-sm text-zinc-500 leading-relaxed">
          No pipeline running. Select a session and start a new scan.
        </p>
        {lastRun && (
          <p className="text-xs text-zinc-600 mt-4 font-mono">
            Last run: {formatDuration(lastRun.duration)}
            {lastRun.grade && ` — Grade ${lastRun.grade}`}
            {lastRun.gaussians && ` — ${Math.round(lastRun.gaussians / 1000)}K Gaussians`}
          </p>
        )}
      </div>
    </div>
  );
};

// ── Complete State ──────────────────────────────────────────────

const CompleteState: React.FC<{ elapsed: number; stageCount: number }> = ({ elapsed, stageCount }) => {
  const { metrics } = usePipelineStore();
  const lastMetric = metrics.length > 0 ? metrics[metrics.length - 1] : null;
  const gaussianCount = lastMetric?.gaussians;

  return (
    <div className="h-full flex flex-col items-center justify-center px-8">
      <div className="max-w-lg w-full">
        <div className="p-8 rounded-2xl bg-emerald-500/[0.04] border border-emerald-500/20">
          <div className="flex items-center justify-center mb-4">
            <div className="w-12 h-12 rounded-full bg-emerald-500/15 flex items-center justify-center">
              <Check size={24} className="text-emerald-400" strokeWidth={2.5} />
            </div>
          </div>
          <h3 className="text-xl font-bold text-emerald-300 text-center mb-1">Pipeline Complete</h3>
          <p className="text-sm text-zinc-500 text-center mb-6">
            {stageCount} stages finished in {formatDuration(elapsed)}
            {gaussianCount ? ` — ${Math.round(gaussianCount / 1000)}K Gaussians` : ''}
          </p>

          <div className="flex items-center justify-center gap-3">
            <button className="flex items-center gap-2 px-4 py-2 rounded-lg bg-zinc-800 hover:bg-zinc-700 border border-zinc-700/50 text-sm text-zinc-300 transition-all active:scale-[0.97]">
              <Eye size={14} />
              View in Gallery
            </button>
            <button className="flex items-center gap-2 px-4 py-2 rounded-lg bg-zinc-800/60 hover:bg-zinc-700/60 border border-zinc-700/50 text-sm text-zinc-400 transition-all active:scale-[0.97]">
              <FolderOpen size={14} />
              Open Folder
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

// ── Running State ───────────────────────────────────────────────

const RunningTimeline: React.FC<{
  stages: StageStatus[];
  elapsed: number;
  stageElapsed: number;
  getStageLogs: (id: number) => string[];
  estimatedRemaining: number | null;
}> = ({ stages, elapsed, stageElapsed, getStageLogs, estimatedRemaining }) => {
  const { metrics } = usePipelineStore();
  const scrollRef = useRef<HTMLDivElement>(null);
  const completedCount = stages.filter(s => s.status === 'complete').length;
  const activeStage = stages.find(s => s.status === 'running');

  // Auto-scroll to active stage
  useEffect(() => {
    if (!scrollRef.current) return;
    const activeEl = scrollRef.current.querySelector('[data-active="true"]');
    if (activeEl) {
      activeEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, [activeStage?.id]);

  return (
    <div className="h-full flex flex-col bg-[#0a0a0b] overflow-hidden">
      {/* Header */}
      <div className="shrink-0 px-6 pt-5 pb-4 border-b border-zinc-800/40">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-lg font-semibold text-zinc-100 flex items-center gap-2.5">
              Pipeline
              <span className="relative flex h-2.5 w-2.5">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-60" />
                <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-indigo-500" />
              </span>
            </h1>
            <p className="text-xs text-zinc-500 mt-0.5">
              Stage {(activeStage?.id ?? 0) + 1} of {stages.length}
              {estimatedRemaining != null && (
                <span className="text-zinc-600 ml-2">
                  <Clock size={10} className="inline -mt-0.5 mr-0.5" />
                  ETA ~{formatDuration(estimatedRemaining)}
                </span>
              )}
            </p>
          </div>
          <ProgressRing completed={completedCount} total={stages.length} elapsed={elapsed} />
        </div>
      </div>

      {/* Timeline */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto scrollbar-hide">
        <div className="max-w-[600px] mx-auto px-6 py-5">
          {stages.map((stage, index) => {
            const name = STAGE_NAMES[stage.id] ?? stage.name;
            const isActive = stage.status === 'running';
            const isCompleted = stage.status === 'complete';
            const isSkipped = stage.status === 'skipped';
            const isPending = stage.status === 'pending';
            const isError = stage.status === 'error';

            return (
              <div key={stage.id} data-active={isActive ? 'true' : 'false'}>
                {/* Completed: single compact line */}
                {isCompleted && (
                  <div className="flex items-center gap-3 py-1.5">
                    <div className="w-5 flex justify-center">
                      <Check size={14} className="text-emerald-500" strokeWidth={2.5} />
                    </div>
                    <span className="text-sm text-zinc-400">{name}</span>
                    {stage.elapsed != null && (
                      <span className="text-xs text-zinc-600 font-mono tabular-nums ml-auto">{formatDuration(stage.elapsed)}</span>
                    )}
                  </div>
                )}

                {/* Active: expanded card */}
                {isActive && (
                  <div className="my-2 p-4 rounded-xl border border-indigo-500/30 bg-indigo-500/[0.04] shadow-lg shadow-indigo-500/[0.06]">
                    <div className="flex items-center gap-3 mb-2">
                      <div className="w-5 flex justify-center">
                        <div className="w-2.5 h-2.5 rounded-full bg-indigo-500 animate-pulse" />
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-semibold text-white">{name}</span>
                          <span className="text-[11px] text-zinc-600 font-mono">#{stage.id}</span>
                        </div>
                        <p className="text-xs text-zinc-500 mt-0.5">{STAGE_DESCRIPTIONS[stage.id] ?? ''}</p>
                      </div>
                      <span className="text-xs text-indigo-400 font-mono tabular-nums shrink-0">{formatDuration(stageElapsed)}</span>
                    </div>

                    {/* Shimmer progress bar */}
                    <div className="w-full h-1 bg-zinc-800 rounded-full overflow-hidden mb-3">
                      <div className="h-full rounded-full bg-gradient-to-r from-indigo-600 via-indigo-400 to-indigo-600 progress-shine" style={{ width: '100%' }} />
                    </div>

                    {/* Live logs */}
                    {(() => {
                      const logs = getStageLogs(stage.id);
                      if (logs.length === 0) return null;
                      return (
                        <div className="max-h-28 overflow-y-auto scrollbar-hide rounded-lg bg-black/40 border border-white/[0.04] p-2.5 space-y-0.5">
                          {logs.slice(-5).map((line, i) => (
                            <div key={i} className="text-[11px] font-mono text-zinc-500 leading-relaxed truncate">
                              <span className="text-zinc-700 mr-1.5 select-none">&gt;</span>
                              {line}
                            </div>
                          ))}
                        </div>
                      );
                    })()}

                    {/* Training metrics inline */}
                    {stage.id === 13 && metrics.length > 0 && <TrainingMetrics />}
                  </div>
                )}

                {/* Error: highlighted line */}
                {isError && (
                  <div className="flex items-center gap-3 py-1.5">
                    <div className="w-5 flex justify-center">
                      <AlertTriangle size={14} className="text-red-400" />
                    </div>
                    <span className="text-sm text-red-400">{name}</span>
                    <span className="text-xs text-red-500/60 ml-auto">failed</span>
                  </div>
                )}

                {/* Skipped: strikethrough dimmed line */}
                {isSkipped && (
                  <div className="flex items-center gap-3 py-1.5 opacity-30">
                    <div className="w-5 flex justify-center">
                      <span className="text-zinc-600 text-xs">—</span>
                    </div>
                    <span className="text-sm text-zinc-600 line-through">{name} (skipped)</span>
                  </div>
                )}

                {/* Pending: gray dot */}
                {isPending && (
                  <div className="flex items-center gap-3 py-1.5 opacity-40">
                    <div className="w-5 flex justify-center">
                      <div className="w-2 h-2 rounded-full border border-zinc-600" />
                    </div>
                    <span className="text-sm text-zinc-600">{name}</span>
                  </div>
                )}

                {/* Connector line between stages */}
                {index < stages.length - 1 && (
                  <div className="flex justify-start ml-[9px]">
                    <div className={`w-px h-1.5 ${
                      isCompleted ? 'bg-emerald-500/30' :
                      isActive ? 'bg-indigo-500/30' :
                      'bg-zinc-800'
                    }`} />
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Bottom log peek */}
      {(() => {
        const { logs } = usePipelineStore.getState();
        if (logs.length === 0) return null;
        return (
          <div className="shrink-0 border-t border-zinc-800/40 bg-zinc-900/30 px-5 py-2.5">
            <div className="flex items-center gap-2 mb-1">
              <Terminal size={10} className="text-zinc-600" />
              <span className="text-[10px] text-zinc-600 uppercase tracking-wider font-semibold">Latest</span>
            </div>
            <div className="text-[11px] font-mono text-zinc-500 truncate">
              {logs[logs.length - 1]?.message ?? ''}
            </div>
          </div>
        );
      })()}
    </div>
  );
};

// ── Main PipelineView ───────────────────────────────────────────

const PipelineView: React.FC = () => {
  const { status, stages, logs, startedAt, currentStage } = usePipelineStore();
  const [elapsed, setElapsed] = useState(0);
  const [stageElapsed, setStageElapsed] = useState(0);
  const [stageStartTime, setStageStartTime] = useState<number | null>(null);
  const prevStageRef = useRef(currentStage);

  const isRunning = status === 'running' || status === 'stopping';
  const isComplete = status === 'complete';
  const isError = status === 'error';

  // Track elapsed time
  useEffect(() => {
    if (!startedAt || !isRunning) return;
    const tick = setInterval(() => setElapsed(Math.floor((Date.now() - startedAt) / 1000)), 1000);
    return () => clearInterval(tick);
  }, [startedAt, isRunning]);

  // Track stage elapsed
  useEffect(() => {
    if (isRunning && currentStage >= 0) {
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

  // Initialize stage start time when pipeline starts
  useEffect(() => {
    if (isRunning && !stageStartTime) {
      setStageStartTime(Date.now());
    }
    if (!isRunning && status === 'idle') {
      setStageStartTime(null);
      setStageElapsed(0);
      setElapsed(0);
    }
  }, [isRunning, status]);

  // Set elapsed for complete state
  useEffect(() => {
    if (isComplete && startedAt) {
      setElapsed(Math.floor((Date.now() - startedAt) / 1000));
    }
  }, [isComplete, startedAt]);

  // Get stage-specific logs
  const getStageLogs = useCallback((stageId: number): string[] => {
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

  // ── IDLE ──
  if (status === 'idle') {
    return (
      <div className="h-full bg-[#0a0a0b]">
        <IdleState />
      </div>
    );
  }

  // ── COMPLETE ──
  if (isComplete) {
    return (
      <div className="h-full bg-[#0a0a0b]">
        <CompleteState elapsed={elapsed} stageCount={completedCount} />
      </div>
    );
  }

  // ── ERROR ──
  if (isError && !isRunning) {
    return (
      <div className="h-full flex flex-col items-center justify-center px-8 bg-[#0a0a0b]">
        <div className="max-w-lg w-full p-8 rounded-2xl bg-red-500/[0.04] border border-red-500/20">
          <div className="flex items-center justify-center mb-4">
            <div className="w-12 h-12 rounded-full bg-red-500/15 flex items-center justify-center">
              <AlertTriangle size={24} className="text-red-400" />
            </div>
          </div>
          <h3 className="text-xl font-bold text-red-300 text-center mb-1">Pipeline Error</h3>
          <p className="text-sm text-zinc-500 text-center">
            Failed at stage {currentStage}. Check the console for details.
          </p>
        </div>
      </div>
    );
  }

  // ── RUNNING ──
  return (
    <RunningTimeline
      stages={stages}
      elapsed={elapsed}
      stageElapsed={stageElapsed}
      getStageLogs={getStageLogs}
      estimatedRemaining={estimatedRemaining}
    />
  );
};

export default PipelineView;
