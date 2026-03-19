import React, { useState, useEffect, useRef } from 'react';
import {
  AreaChart, Area, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts';
import { Activity, TrendingUp, TrendingDown, Minus, Zap } from 'lucide-react';
import usePipelineStore from '../store/pipelineStore';

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
      const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
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
}> = ({ label, value, unit, trend, accentColor, glowing }) => (
  <div
    className={`bg-zinc-900/30 border border-zinc-800/60 rounded-xl p-4 flex flex-col gap-1 transition-all duration-500 ${
      glowing ? 'animate-pulse-glow-emerald' : ''
    }`}
  >
    <div className="flex items-center justify-between">
      <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider">{label}</span>
      {trend}
    </div>
    <div className="flex items-baseline gap-1 mt-1">
      <span className={`text-2xl font-semibold ${accentColor}`}>{value}</span>
      {unit && <span className="text-xs text-zinc-600">{unit}</span>}
    </div>
  </div>
);

// ── Main Component ───────────────────────────────────────────────

export const TrainingMetrics: React.FC = () => {
  const [timeRange, setTimeRange] = useState('all');
  const { metrics, gpuInfo } = usePipelineStore();

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

  // Iterations per second (estimate from last 20 metric points)
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

  // Is PSNR improving?
  const psnrImproving = latestPsnr != null && prevPsnr != null && latestPsnr > prevPsnr;

  return (
    <div className="flex-1 bg-[#0a0a0b] p-8 overflow-y-auto scrollbar-hide">

      {/* Header */}
      <div className="flex items-center justify-between mb-8 animate-fadeIn">
        <div>
          <h1 className="text-2xl font-light text-zinc-100 flex items-center gap-3">
            <Activity className="text-emerald-500" />
            Training Metrics
          </h1>
          <p className="text-sm text-zinc-500 mt-1">
            {metrics.length > 0
              ? `${metrics.length} data points collected`
              : 'Real-time performance overview'}
          </p>
        </div>

        <div className="flex gap-1 bg-zinc-900/50 p-1 rounded-lg border border-zinc-800/60">
          {['all', '1k', '500'].map((range) => (
            <button
              key={range}
              onClick={() => setTimeRange(range)}
              className={`px-4 py-1.5 text-xs font-medium rounded-md transition-all duration-150 active:scale-95 ${
                timeRange === range
                  ? 'bg-zinc-800 text-zinc-100 shadow-sm'
                  : 'text-zinc-500 hover:text-zinc-300 hover:bg-white/[0.03]'
              }`}
            >
              {range === 'all' ? 'All Time' : `Last ${range}`}
            </button>
          ))}
        </div>
      </div>

      {/* Summary Stat Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8 animate-slideUp">
        <StatCard
          label="PSNR"
          value={latestPsnr != null ? latestPsnr.toFixed(2) : '--'}
          unit="dB"
          trend={<TrendArrow current={latestPsnr} previous={prevPsnr} />}
          accentColor="text-emerald-400"
          glowing={psnrImproving}
        />
        <StatCard
          label="L1 Loss"
          value={latestLoss != null ? latestLoss.toFixed(4) : '--'}
          trend={<TrendArrow current={latestLoss} previous={prevLoss} invert />}
          accentColor="text-amber-400"
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
        />
        <StatCard
          label="Iter/sec"
          value={iterPerSec != null ? iterPerSec.toString() : '--'}
          unit={iterPerSec != null ? 'it/s' : ''}
          trend={<Zap size={12} className={iterPerSec != null ? 'text-indigo-400' : 'text-zinc-600'} />}
          accentColor="text-indigo-400"
        />
      </div>

      {/* Chart Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

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
                  <YAxis stroke="#52525b" fontSize={10} tickLine={false} axisLine={false} />
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
    </div>
  );
};
