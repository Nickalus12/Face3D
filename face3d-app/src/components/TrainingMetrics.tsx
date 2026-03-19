import React, { useState, useEffect, useRef, useMemo } from 'react';
import {
  AreaChart, Area, LineChart, Line, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from 'recharts';
import {
  Activity, TrendingUp, TrendingDown, Minus, Zap,
  Clock, HardDrive, Layers, Box, FileText, BarChart2,
  CheckCircle2, Timer, Image as ImageIcon,
} from 'lucide-react';
import usePipelineStore, { PIPELINE_STAGES } from '../store/pipelineStore';
import useSessionStore from '../store/sessionStore';
import { useSessionData } from '../hooks/useSessionData';

// ── Animated Number Counter ──────────────────────────────────────

const AnimatedNumber: React.FC<{ value: number; decimals?: number; suffix?: string }> = ({
  value,
  decimals = 2,
  suffix = '',
}) => {
  const [display, setDisplay] = useState(value);
  const prevRef = useRef(value);
  const frameRef = useRef<number>(0);

  useEffect(() => {
    const from = prevRef.current;
    const to = value;
    const duration = 400;
    const startTime = performance.now();

    const animate = (now: number) => {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setDisplay(from + (to - from) * eased);
      if (progress < 1) {
        frameRef.current = requestAnimationFrame(animate);
      }
    };

    frameRef.current = requestAnimationFrame(animate);
    prevRef.current = value;
    return () => cancelAnimationFrame(frameRef.current);
  }, [value]);

  return (
    <span className="tabular-nums">
      {display.toFixed(decimals)}{suffix}
    </span>
  );
};

// ── Trend Arrow ──────────────────────────────────────────────────

const TrendArrow: React.FC<{ current: number | null; previous: number | null; invert?: boolean }> = ({
  current,
  previous,
  invert = false,
}) => {
  if (current == null || previous == null) return <Minus size={12} className="text-zinc-600" />;
  const diff = current - previous;
  const isPositive = invert ? diff < 0 : diff > 0;
  const isNegative = invert ? diff > 0 : diff < 0;
  if (Math.abs(diff) < 0.0001) return <Minus size={12} className="text-zinc-500" />;
  if (isPositive) return <TrendingUp size={12} className="text-emerald-400" />;
  if (isNegative) return <TrendingDown size={12} className="text-red-400" />;
  return <Minus size={12} className="text-zinc-600" />;
};

// ── Custom Tooltip ───────────────────────────────────────────────

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-zinc-900/95 backdrop-blur border border-zinc-800 p-3 rounded-lg shadow-xl shadow-black/30 text-xs animate-fadeIn">
        <p className="text-zinc-500 mb-1">Iteration {label}</p>
        {payload.map((p: any, i: number) => (
          <p key={i} className="font-mono font-bold" style={{ color: p.stroke || p.color }}>
            {p.name}: {typeof p.value === 'number' ? p.value.toFixed(4) : p.value}
          </p>
        ))}
      </div>
    );
  }
  return null;
};

// ── Empty State ──────────────────────────────────────────────────

const EmptyState = ({ label }: { label: string }) => (
  <div className="h-64 flex items-center justify-center">
    <div className="text-center animate-fadeIn">
      <div className="text-zinc-700 text-sm mb-1">No data yet</div>
      <div className="text-zinc-800 text-xs">
        {label} will appear once training begins
      </div>
    </div>
  </div>
);

// ── Stat Card ────────────────────────────────────────────────────

const StatCard: React.FC<{
  label: string;
  value: string;
  unit?: string;
  trend: React.ReactNode;
  accentColor: string;
  glowing?: boolean;
  icon?: React.ReactNode;
}> = ({ label, value, unit, trend, accentColor, glowing, icon }) => (
  <div
    className={`bg-zinc-900/30 border border-zinc-800/60 rounded-xl p-4 flex flex-col gap-1 transition-all duration-500 ${
      glowing ? 'animate-pulse-glow-emerald' : ''
    }`}
  >
    <div className="flex items-center justify-between">
      <div className="flex items-center gap-1.5">
        {icon}
        <span className="text-[11px] font-bold text-zinc-500 uppercase tracking-wider">{label}</span>
      </div>
      {trend}
    </div>
    <div className="flex items-baseline gap-1 mt-1">
      <span className={`text-2xl font-semibold ${accentColor}`}>{value}</span>
      {unit && <span className="text-xs text-zinc-600">{unit}</span>}
    </div>
  </div>
);

// ── Stage Timing Bar Chart ───────────────────────────────────────

const STAGE_COLORS: Record<string, string> = {
  complete: '#10b981',
  running: '#6366f1',
  error: '#ef4444',
  pending: '#3f3f46',
  skipped: '#52525b',
};

const StageTimingChart: React.FC = () => {
  const { stages, status } = usePipelineStore();
  const { currentSession } = useSessionStore();
  const [dbStages, setDbStages] = useState<any[]>([]);

  // Load historical stage data from DB when not running
  useEffect(() => {
    if (status === 'running' || !currentSession?.id) return;
    import('../lib/dbBridge').then(({ getSessionSummaryFromDb }) => {
      getSessionSummaryFromDb(currentSession.id).then((summary) => {
        if (summary?.stages && summary.stages.length > 0) {
          setDbStages(summary.stages.map((s: any) => ({
            name: (s.stage_name || `Stage ${s.stage_num}`).slice(0, 14),
            fullName: s.stage_name || `Stage ${s.stage_num}`,
            elapsed: s.duration_s ?? 0,
            status: s.status,
            id: s.stage_num,
          })));
        }
      }).catch(() => {});
    }).catch(() => {});
  }, [currentSession?.id, status]);

  // Use live data when running, DB data for history
  const liveStageData = stages
    .filter((s) => s.status !== 'pending')
    .map((s) => ({
      name: s.name.length > 14 ? s.name.slice(0, 12) + '..' : s.name,
      fullName: s.name,
      elapsed: s.elapsed ?? 0,
      status: s.status,
      id: s.id,
    }));

  const stageData = liveStageData.length > 0 ? liveStageData : dbStages;

  if (stageData.length === 0) {
    return (
      <div className="bg-zinc-900/30 border border-zinc-800/60 rounded-2xl p-6">
        <h3 className="text-sm font-semibold text-zinc-400 uppercase tracking-wider mb-4 flex items-center gap-2">
          <Timer size={16} className="text-zinc-500" />
          Stage Timing
        </h3>
        <div className="h-48 flex items-center justify-center text-zinc-700 text-sm">
          Run the pipeline to see stage timings
        </div>
      </div>
    );
  }

  const totalTime = stageData.reduce((sum, s) => sum + s.elapsed, 0);

  const StageTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const d = payload[0].payload;
      return (
        <div className="bg-zinc-900/95 backdrop-blur border border-zinc-800 p-3 rounded-lg shadow-xl text-xs">
          <p className="text-zinc-300 font-medium mb-1">{d.fullName}</p>
          <p className="text-zinc-400">
            Duration: <span className="text-zinc-200 font-mono">{formatDuration(d.elapsed)}</span>
          </p>
          <p className="text-zinc-500">
            {totalTime > 0 ? `${((d.elapsed / totalTime) * 100).toFixed(1)}% of total` : ''}
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="bg-zinc-900/30 border border-zinc-800/60 rounded-2xl p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-zinc-400 uppercase tracking-wider flex items-center gap-2">
          <Timer size={16} className="text-zinc-500" />
          Stage Timing
        </h3>
        <div className="flex items-center gap-2 text-xs text-zinc-500">
          <Clock size={12} />
          <span className="font-mono">{formatDuration(totalTime)}</span>
          <span>total</span>
        </div>
      </div>
      <div className="h-[260px]">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={stageData} layout="vertical" margin={{ left: 10, right: 20, top: 5, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#27272a" horizontal={false} />
            <XAxis
              type="number"
              stroke="#52525b"
              fontSize={10}
              tickLine={false}
              axisLine={false}
              tickFormatter={(v) => formatDuration(v)}
            />
            <YAxis
              type="category"
              dataKey="name"
              stroke="#52525b"
              fontSize={10}
              tickLine={false}
              axisLine={false}
              width={100}
            />
            <Tooltip content={<StageTooltip />} />
            <Bar dataKey="elapsed" radius={[0, 4, 4, 0]} maxBarSize={20}>
              {stageData.map((entry, idx) => (
                <Cell key={idx} fill={STAGE_COLORS[entry.status] || STAGE_COLORS.pending} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Legend */}
      <div className="flex items-center gap-4 mt-3 text-[11px] text-zinc-500">
        <div className="flex items-center gap-1.5">
          <div className="w-2 h-2 rounded-sm bg-emerald-500" />
          <span>Complete</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-2 h-2 rounded-sm bg-indigo-500" />
          <span>Running</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-2 h-2 rounded-sm bg-red-500" />
          <span>Error</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-2 h-2 rounded-sm bg-zinc-600" />
          <span>Skipped</span>
        </div>
      </div>
    </div>
  );
};

// ── Data Summary Cards ───────────────────────────────────────────

const DataSummaryCards: React.FC = () => {
  const { currentSession } = useSessionStore();
  const sessionData = useSessionData(currentSession?.id ?? null);
  const { stages, startedAt, status } = usePipelineStore();

  const isRunning = status === 'running';
  const completedStages = stages.filter((s) => s.status === 'complete').length;
  const totalElapsed = stages.reduce((sum, s) => sum + (s.elapsed ?? 0), 0);

  // Parse metrics from session data
  const parsedMetrics = useMemo(() => {
    if (!sessionData.metrics) return null;
    try {
      const parsed = JSON.parse(sessionData.metrics);
      if (Array.isArray(parsed) && parsed.length > 0) {
        const last = parsed[parsed.length - 1];
        return {
          psnr: last.psnr ?? null,
          loss: last.loss ?? null,
          gaussians: last.n_gs ?? last.gaussians ?? null,
          iterations: last.iter ?? last.iteration ?? parsed.length,
        };
      }
      if (typeof parsed === 'object' && !Array.isArray(parsed)) {
        return {
          psnr: parsed.psnr ?? null,
          loss: parsed.loss ?? null,
          gaussians: parsed.n_gs ?? parsed.gaussians ?? null,
          iterations: parsed.iter ?? parsed.iteration ?? null,
        };
      }
    } catch {
      // ignore
    }
    return null;
  }, [sessionData.metrics]);

  const totalSize = sessionData.files.reduce((sum, f) => sum + f.size, 0);
  const plyFile = sessionData.files.find((f) => f.name.endsWith('.ply'));
  const meshFile = sessionData.files.find((f) => f.name.includes('mesh') || f.name.endsWith('.obj'));
  const textureFile = sessionData.files.find((f) => f.name.includes('texture') || f.name.includes('.png'));

  const items: { label: string; value: string; icon: React.ReactNode; color: string }[] = [
    {
      label: 'Total Time',
      value: totalElapsed > 0 ? formatDuration(totalElapsed) : isRunning && startedAt ? formatDuration(Math.floor((Date.now() - startedAt) / 1000)) : '--',
      icon: <Clock size={16} />,
      color: 'text-indigo-400',
    },
    {
      label: 'Stages',
      value: `${completedStages} / ${PIPELINE_STAGES.length}`,
      icon: <CheckCircle2 size={16} />,
      color: completedStages === PIPELINE_STAGES.length ? 'text-emerald-400' : 'text-amber-400',
    },
    {
      label: 'Input Frames',
      value: sessionData.imageCounts[2] > 0
        ? `${sessionData.imageCounts[2]}`
        : '--',
      icon: <ImageIcon size={16} />,
      color: 'text-blue-400',
    },
    {
      label: 'Selected Frames',
      value: sessionData.imageCounts[1] > 0
        ? `${sessionData.imageCounts[1]}`
        : '--',
      icon: <ImageIcon size={16} />,
      color: 'text-cyan-400',
    },
    {
      label: 'Gaussians',
      value: parsedMetrics?.gaussians != null
        ? parsedMetrics.gaussians >= 1_000_000
          ? `${(parsedMetrics.gaussians / 1_000_000).toFixed(1)}M`
          : parsedMetrics.gaussians >= 1_000
          ? `${Math.round(parsedMetrics.gaussians / 1_000)}K`
          : String(parsedMetrics.gaussians)
        : '--',
      icon: <Layers size={16} />,
      color: 'text-purple-400',
    },
    {
      label: 'Renders',
      value: sessionData.imageCounts[0] > 0 ? String(sessionData.imageCounts[0]) : '--',
      icon: <Layers size={16} />,
      color: 'text-pink-400',
    },
  ];

  const fileItems: { name: string; size: number; color: string }[] = [];
  if (plyFile) fileItems.push({ name: 'Gaussian PLY', size: plyFile.size, color: 'text-indigo-400' });
  if (meshFile) fileItems.push({ name: 'Mesh', size: meshFile.size, color: 'text-blue-400' });
  if (textureFile) fileItems.push({ name: 'Texture', size: textureFile.size, color: 'text-emerald-400' });

  return (
    <div className="space-y-6">
      {/* Summary grid */}
      <div className="bg-zinc-900/30 border border-zinc-800/60 rounded-2xl p-6">
        <h3 className="text-sm font-semibold text-zinc-400 uppercase tracking-wider mb-4 flex items-center gap-2">
          <BarChart2 size={16} className="text-zinc-500" />
          Pipeline Summary
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-3">
          {items.map((item) => (
            <div
              key={item.label}
              className="p-3 rounded-xl bg-zinc-900/60 border border-zinc-800/40 flex flex-col gap-1.5"
            >
              <div className={`${item.color}`}>{item.icon}</div>
              <div className="text-xl font-semibold text-zinc-200 tabular-nums">{item.value}</div>
              <div className="text-[11px] text-zinc-500 uppercase tracking-wider">{item.label}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Output files */}
      {(fileItems.length > 0 || sessionData.files.length > 0) && (
        <div className="bg-zinc-900/30 border border-zinc-800/60 rounded-2xl p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold text-zinc-400 uppercase tracking-wider flex items-center gap-2">
              <HardDrive size={16} className="text-zinc-500" />
              Output Files
            </h3>
            <span className="text-xs text-zinc-500 font-mono">{formatBytesLong(totalSize)}</span>
          </div>

          {/* File size bars */}
          <div className="space-y-2">
            {sessionData.files.map((file) => {
              const pct = totalSize > 0 ? (file.size / totalSize) * 100 : 0;
              const ext = file.name.split('.').pop()?.toLowerCase() || '';
              const barColor = ext === 'ply'
                ? 'bg-indigo-500'
                : ext === 'obj'
                ? 'bg-blue-500'
                : ext === 'png' || ext === 'jpg'
                ? 'bg-emerald-500'
                : ext === 'html'
                ? 'bg-amber-500'
                : 'bg-zinc-600';

              return (
                <div key={file.name} className="group">
                  <div className="flex items-center justify-between mb-1">
                    <div className="flex items-center gap-2 min-w-0">
                      <FileText size={12} className="text-zinc-600 shrink-0" />
                      <span className="text-xs text-zinc-300 truncate">{file.name}</span>
                    </div>
                    <span className="text-[11px] text-zinc-500 font-mono shrink-0 ml-2">
                      {formatBytesLong(file.size)}
                    </span>
                  </div>
                  <div className="w-full h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                    <div
                      className={`h-full ${barColor} rounded-full transition-all duration-500`}
                      style={{ width: `${Math.max(pct, 1)}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};

// ── Helpers ──────────────────────────────────────────────────────

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  if (m < 60) return `${m}m ${s.toString().padStart(2, '0')}s`;
  const h = Math.floor(m / 60);
  const rm = m % 60;
  return `${h}h ${rm.toString().padStart(2, '0')}m`;
}

function formatBytesLong(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
}

// ── Main Component ───────────────────────────────────────────────

export const TrainingMetrics: React.FC = () => {
  const [timeRange, setTimeRange] = useState('all');
  const [activeSection, setActiveSection] = useState<'training' | 'pipeline' | 'data'>('training');
  const { metrics: liveMetrics, gpuInfo, status } = usePipelineStore();
  const { currentSession } = useSessionStore();
  const [dbMetrics, setDbMetrics] = useState<any[]>([]);
  const isTraining = status === 'running';

  // Load historical metrics from DB when not actively training
  useEffect(() => {
    if (isTraining || !currentSession?.id) return;
    import('../lib/dbBridge').then(({ getTrainingMetricsFromDb }) => {
      getTrainingMetricsFromDb(currentSession.id).then((data) => {
        if (data && data.length > 0) {
          setDbMetrics(data.map((d: any) => ({
            iter: d.iteration,
            loss: d.loss,
            psnr: d.psnr,
            gaussians: d.num_gaussians,
          })));
        }
      }).catch(() => {});
    }).catch(() => {});
  }, [currentSession?.id, isTraining]);

  // Use live metrics when training, DB metrics when viewing history
  const metrics = isTraining ? liveMetrics : (liveMetrics.length > 0 ? liveMetrics : dbMetrics);

  // Filter data by time range
  const getData = () => {
    if (metrics.length === 0) return [];
    switch (timeRange) {
      case '500':
        return metrics.slice(-500);
      case '1k':
        return metrics.slice(-1000);
      default:
        return metrics;
    }
  };

  const data = getData();

  // Latest values
  const psnrValues = data.filter((d) => d.psnr != null);
  const lossValues = data.filter((d) => d.loss != null);
  const gsValues = data.filter((d) => d.gaussians != null);

  const latestPsnr = psnrValues.length > 0 ? psnrValues[psnrValues.length - 1]?.psnr ?? null : null;
  const prevPsnr = psnrValues.length > 1 ? psnrValues[psnrValues.length - 2]?.psnr ?? null : null;

  const latestLoss = lossValues.length > 0 ? lossValues[lossValues.length - 1]?.loss ?? null : null;
  const prevLoss = lossValues.length > 1 ? lossValues[lossValues.length - 2]?.loss ?? null : null;

  const latestGaussians = gsValues.length > 0 ? gsValues[gsValues.length - 1]?.gaussians ?? null : null;

  // VRAM from GPU info
  const vramUsedGB = gpuInfo?.memory_used
    ? (parseFloat(gpuInfo.memory_used) / 1024).toFixed(1)
    : null;

  const hasPsnr = data.some((d) => d.psnr != null);
  const hasLoss = data.some((d) => d.loss != null);
  const hasGaussians = data.some((d) => d.gaussians != null);

  // Iterations per second
  const [iterPerSec, setIterPerSec] = useState<number | null>(null);
  const lastMetricsTimeRef = useRef<number>(Date.now());
  const lastMetricsCountRef = useRef<number>(metrics.length);

  useEffect(() => {
    const now = Date.now();
    const deltaTime = (now - lastMetricsTimeRef.current) / 1000;
    const deltaCount = metrics.length - lastMetricsCountRef.current;
    if (deltaTime > 2 && deltaCount > 0) {
      setIterPerSec(Math.round(deltaCount / deltaTime));
      lastMetricsTimeRef.current = now;
      lastMetricsCountRef.current = metrics.length;
    }
  }, [metrics.length]);

  const psnrImproving = latestPsnr != null && prevPsnr != null && latestPsnr > prevPsnr;

  const sectionTabs = [
    { id: 'training' as const, label: 'Training Curves' },
    { id: 'pipeline' as const, label: 'Stage Timing' },
    { id: 'data' as const, label: 'Data Summary' },
  ];

  return (
    <div className="flex-1 bg-[#0a0a0b] p-8 overflow-y-auto scrollbar-hide">
      {/* Header */}
      <div className="flex items-center justify-between mb-6 animate-fadeIn">
        <div>
          <h1 className="text-2xl font-light text-zinc-100 flex items-center gap-3">
            <Activity className="text-emerald-500" />
            Performance Metrics
            {isTraining && (
              <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-60" />
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
                </span>
                LIVE
              </span>
            )}
          </h1>
          <p className="text-sm text-zinc-500 mt-1">
            {metrics.length > 0
              ? `${metrics.length} data points collected`
              : 'Real-time performance overview'}
          </p>
        </div>

        <div className="flex items-center gap-3">
          {/* Section tabs */}
          <div className="flex gap-1 bg-zinc-900/50 p-1 rounded-lg border border-zinc-800/60">
            {sectionTabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveSection(tab.id)}
                className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all duration-150 active:scale-95 ${
                  activeSection === tab.id
                    ? 'bg-zinc-800 text-zinc-100 shadow-sm'
                    : 'text-zinc-500 hover:text-zinc-300 hover:bg-white/[0.03]'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {/* Time range filter (only for training view) */}
          {activeSection === 'training' && (
            <div className="flex gap-1 bg-zinc-900/50 p-1 rounded-lg border border-zinc-800/60">
              {['all', '1k', '500'].map((range) => (
                <button
                  key={range}
                  onClick={() => setTimeRange(range)}
                  className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all duration-150 active:scale-95 ${
                    timeRange === range
                      ? 'bg-zinc-800 text-zinc-100 shadow-sm'
                      : 'text-zinc-500 hover:text-zinc-300 hover:bg-white/[0.03]'
                  }`}
                >
                  {range === 'all' ? 'All' : `Last ${range}`}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Summary Stat Cards -- always visible */}
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4 mb-8 animate-slideUp">
        <StatCard
          label="PSNR"
          value={latestPsnr != null ? latestPsnr.toFixed(2) : '--'}
          unit="dB"
          trend={<TrendArrow current={latestPsnr} previous={prevPsnr} />}
          accentColor="text-emerald-400"
          glowing={psnrImproving}
          icon={<Activity size={12} className="text-emerald-500" />}
        />
        <StatCard
          label="L1 Loss"
          value={latestLoss != null ? latestLoss.toFixed(4) : '--'}
          trend={<TrendArrow current={latestLoss} previous={prevLoss} invert />}
          accentColor="text-amber-400"
          icon={<TrendingDown size={12} className="text-amber-500" />}
        />
        <StatCard
          label="Gaussians"
          value={
            latestGaussians != null
              ? latestGaussians >= 1_000_000
                ? `${(latestGaussians / 1_000_000).toFixed(2)}M`
                : latestGaussians >= 1_000
                ? `${(latestGaussians / 1_000).toFixed(0)}K`
                : latestGaussians.toString()
              : '--'
          }
          trend={<TrendArrow current={latestGaussians} previous={gsValues.length > 1 ? gsValues[gsValues.length - 2]?.gaussians ?? null : null} />}
          accentColor="text-blue-400"
          icon={<Box size={12} className="text-blue-500" />}
        />
        <StatCard
          label="Iter/sec"
          value={iterPerSec != null ? iterPerSec.toString() : '--'}
          unit={iterPerSec != null ? 'it/s' : ''}
          trend={<Zap size={12} className={iterPerSec != null ? 'text-indigo-400' : 'text-zinc-600'} />}
          accentColor="text-indigo-400"
          icon={<Zap size={12} className="text-indigo-500" />}
        />
      </div>

      {/* ═══ TRAINING CURVES SECTION ═══ */}
      {activeSection === 'training' && (
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          {/* PSNR Chart */}
          <div className={`bg-zinc-900/30 border border-zinc-800/60 rounded-2xl p-6 transition-all duration-500 ${psnrImproving ? 'shadow-[0_0_20px_rgba(16,185,129,0.06)]' : ''}`}>
            <div className="flex justify-between items-center mb-6">
              <div>
                <h3 className="text-sm font-semibold text-zinc-400 uppercase tracking-wider">PSNR</h3>
                <div className="text-2xl font-light text-zinc-100 mt-1">
                  {latestPsnr != null ? <AnimatedNumber value={latestPsnr} decimals={2} /> : '--'}
                  <span className="text-xs text-zinc-600 font-medium ml-1">dB</span>
                </div>
              </div>
              {hasPsnr && data.length > 10 && (
                <div className="w-20 h-8 opacity-50">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={data.slice(-10)}>
                      <Line type="monotone" dataKey="psnr" stroke="#10b981" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>
            {hasPsnr ? (
              <div className="h-64 chart-crosshair">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={data.filter((d) => d.psnr != null)}>
                    <defs>
                      <linearGradient id="colorPsnr" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#10b981" stopOpacity={0.2} />
                        <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                    <XAxis dataKey="iter" stroke="#52525b" fontSize={10} tickLine={false} axisLine={false} />
                    <YAxis stroke="#52525b" fontSize={10} tickLine={false} axisLine={false} domain={['auto', 'auto']} />
                    <Tooltip content={<CustomTooltip />} />
                    <Area
                      type="monotone" dataKey="psnr" name="PSNR"
                      stroke="#10b981" strokeWidth={2} fill="url(#colorPsnr)" fillOpacity={1}
                      activeDot={{ r: 4, fill: '#10b981', stroke: '#0a0a0b', strokeWidth: 2 }}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <EmptyState label="PSNR data" />
            )}
          </div>

          {/* Loss Chart */}
          <div className="bg-zinc-900/30 border border-zinc-800/60 rounded-2xl p-6">
            <div className="flex justify-between items-center mb-6">
              <div>
                <h3 className="text-sm font-semibold text-zinc-400 uppercase tracking-wider">L1 Loss</h3>
                <div className="text-2xl font-light text-zinc-100 mt-1">
                  {latestLoss != null ? <AnimatedNumber value={latestLoss} decimals={4} /> : '--'}
                </div>
              </div>
            </div>
            {hasLoss ? (
              <div className="h-64 chart-crosshair">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={data.filter((d) => d.loss != null)}>
                    <defs>
                      <linearGradient id="colorLoss" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.2} />
                        <stop offset="95%" stopColor="#f59e0b" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                    <XAxis dataKey="iter" stroke="#52525b" fontSize={10} tickLine={false} axisLine={false} />
                    <YAxis stroke="#52525b" fontSize={10} tickLine={false} axisLine={false} domain={['auto', 'auto']} />
                    <Tooltip content={<CustomTooltip />} />
                    <Area
                      type="monotone" dataKey="loss" name="Loss"
                      stroke="#f59e0b" strokeWidth={2} fill="url(#colorLoss)" fillOpacity={1}
                      activeDot={{ r: 4, fill: '#f59e0b', stroke: '#0a0a0b', strokeWidth: 2 }}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <EmptyState label="Loss data" />
            )}
          </div>

          {/* Gaussians Area */}
          <div className="bg-zinc-900/30 border border-zinc-800/60 rounded-2xl p-6">
            <div className="flex justify-between items-center mb-6">
              <div>
                <h3 className="text-sm font-semibold text-zinc-400 uppercase tracking-wider">Gaussians Count</h3>
                <div className="text-2xl font-light text-zinc-100 mt-1">
                  {latestGaussians != null
                    ? latestGaussians >= 1_000_000
                      ? `${(latestGaussians / 1_000_000).toFixed(2)}`
                      : latestGaussians >= 1_000
                      ? `${(latestGaussians / 1_000).toFixed(0)}K`
                      : latestGaussians
                    : '--'}
                  {latestGaussians != null && latestGaussians >= 1_000_000 && (
                    <span className="text-xs text-zinc-600 font-medium ml-1">M</span>
                  )}
                </div>
              </div>
            </div>
            {hasGaussians ? (
              <div className="h-64 chart-crosshair">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={data.filter((d) => d.gaussians != null)}>
                    <defs>
                      <linearGradient id="colorGaussians" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                    <XAxis dataKey="iter" stroke="#52525b" fontSize={10} tickLine={false} axisLine={false} />
                    <YAxis
                      stroke="#52525b" fontSize={10} tickLine={false} axisLine={false}
                      tickFormatter={(val) =>
                        val >= 1_000_000
                          ? `${(val / 1_000_000).toFixed(1)}M`
                          : val >= 1_000
                          ? `${Math.round(val / 1_000)}K`
                          : val
                      }
                    />
                    <Tooltip content={<CustomTooltip />} />
                    <Area
                      type="monotone" dataKey="gaussians" name="Count"
                      stroke="#3b82f6" fillOpacity={1} fill="url(#colorGaussians)"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <EmptyState label="Gaussian count data" />
            )}
          </div>

          {/* VRAM Card */}
          <div className="bg-zinc-900/30 border border-zinc-800/60 rounded-2xl p-6">
            <div className="flex justify-between items-center mb-6">
              <div>
                <h3 className="text-sm font-semibold text-zinc-400 uppercase tracking-wider">VRAM Usage</h3>
                <div className="text-2xl font-light text-zinc-100 mt-1">
                  {vramUsedGB ?? '--'}
                  <span className="text-xs text-zinc-600 font-medium ml-1">GB</span>
                </div>
              </div>
            </div>
            <div className="h-64 flex items-center justify-center">
              {gpuInfo ? (
                <div className="text-center space-y-4 w-full px-4 animate-fadeIn">
                  <div className="text-sm text-zinc-400">{gpuInfo.name}</div>
                  <div className="w-full bg-zinc-800 rounded-full h-3 overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-purple-600 to-purple-400 rounded-full transition-all duration-500"
                      style={{
                        width: `${
                          gpuInfo.memory_total
                            ? (parseFloat(gpuInfo.memory_used) / parseFloat(gpuInfo.memory_total)) * 100
                            : 0
                        }%`,
                      }}
                    />
                  </div>
                  <div className="flex justify-between text-xs text-zinc-500">
                    <span>{gpuInfo.memory_used} used</span>
                    <span>{gpuInfo.memory_total} total</span>
                  </div>
                  <div className="text-xs text-zinc-500">
                    GPU Utilization:{' '}
                    <span className={
                      parseInt(gpuInfo.utilization || '0', 10) > 80
                        ? 'text-red-400 font-bold'
                        : parseInt(gpuInfo.utilization || '0', 10) > 50
                        ? 'text-amber-400'
                        : 'text-emerald-400'
                    }>
                      {gpuInfo.utilization}
                    </span>
                  </div>
                </div>
              ) : (
                <div className="text-center animate-fadeIn">
                  <div className="text-zinc-700 text-sm mb-1">No GPU data</div>
                  <div className="text-zinc-800 text-xs">GPU info will appear once detected</div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ═══ PIPELINE TIMING SECTION ═══ */}
      {activeSection === 'pipeline' && (
        <StageTimingChart />
      )}

      {/* ═══ DATA SUMMARY SECTION ═══ */}
      {activeSection === 'data' && (
        <DataSummaryCards />
      )}
    </div>
  );
};
