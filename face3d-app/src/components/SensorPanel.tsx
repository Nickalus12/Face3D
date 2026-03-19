import { useEffect, useState, useCallback, useMemo, useRef } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
} from 'recharts';
import {
  Activity, Smartphone, Clock, Radio, Sun, Wind, RotateCcw,
  Gauge, Signal, AlertTriangle, CheckCircle2, Compass,
  ChevronDown, ChevronUp, Maximize2,
} from 'lucide-react';
import useSessionStore from '../store/sessionStore';
import { getSensorSummary, getSensorData, type SensorSummary } from '../lib/tauri';

interface ChartData {
  time: number;
  value: number;
}

interface SensorChartConfig {
  key: string;
  label: string;
  unit: string;
  color: string;
  icon: React.ReactNode;
  description: string;
  gradientId: string;
}

const CHART_CONFIGS: SensorChartConfig[] = [
  {
    key: 'accel_magnitude',
    label: 'Accelerometer',
    unit: 'm/s\u00B2',
    color: '#6366f1',
    icon: <Activity size={14} />,
    description: 'Camera movement intensity',
    gradientId: 'grad-accel',
  },
  {
    key: 'gyro_magnitude',
    label: 'Gyroscope',
    unit: '\u00B0/s',
    color: '#f59e0b',
    icon: <RotateCcw size={14} />,
    description: 'Rotation speed',
    gradientId: 'grad-gyro',
  },
  {
    key: 'barometer',
    label: 'Barometer',
    unit: 'hPa',
    color: '#10b981',
    icon: <Wind size={14} />,
    description: 'Atmospheric pressure (height proxy)',
    gradientId: 'grad-baro',
  },
  {
    key: 'light',
    label: 'Light Sensor',
    unit: 'lux',
    color: '#eab308',
    icon: <Sun size={14} />,
    description: 'Ambient light level',
    gradientId: 'grad-light',
  },
];

// ── Data Quality Indicator ───────────────────────────────────────

interface DataQuality {
  samplingRate: number;
  consistency: 'excellent' | 'good' | 'fair' | 'poor';
  gapCount: number;
  stdDev: number;
}

function analyzeDataQuality(data: ChartData[]): DataQuality {
  if (data.length < 2) {
    return { samplingRate: 0, consistency: 'poor', gapCount: 0, stdDev: 0 };
  }

  // Calculate sampling rate
  const totalTime = data[data.length - 1].time - data[0].time;
  const samplingRate = totalTime > 0 ? data.length / totalTime : 0;

  // Check for gaps (more than 3x the expected interval)
  const expectedInterval = totalTime / (data.length - 1);
  let gapCount = 0;
  for (let i = 1; i < data.length; i++) {
    const interval = data[i].time - data[i - 1].time;
    if (interval > expectedInterval * 3) {
      gapCount++;
    }
  }

  // Standard deviation of values
  const mean = data.reduce((s, d) => s + d.value, 0) / data.length;
  const variance = data.reduce((s, d) => s + (d.value - mean) ** 2, 0) / data.length;
  const stdDev = Math.sqrt(variance);

  // Consistency rating
  let consistency: DataQuality['consistency'] = 'excellent';
  if (gapCount > 10 || samplingRate < 50) consistency = 'poor';
  else if (gapCount > 5 || samplingRate < 100) consistency = 'fair';
  else if (gapCount > 1 || samplingRate < 150) consistency = 'good';

  return { samplingRate, consistency, gapCount, stdDev };
}

// ── Quality Badge ────────────────────────────────────────────────

function QualityBadge({ quality }: { quality: DataQuality['consistency'] }) {
  const styles = {
    excellent: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/20',
    good: 'bg-blue-500/15 text-blue-400 border-blue-500/20',
    fair: 'bg-amber-500/15 text-amber-400 border-amber-500/20',
    poor: 'bg-red-500/15 text-red-400 border-red-500/20',
  };

  const icons = {
    excellent: <CheckCircle2 size={10} />,
    good: <CheckCircle2 size={10} />,
    fair: <AlertTriangle size={10} />,
    poor: <AlertTriangle size={10} />,
  };

  return (
    <span className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[9px] font-semibold uppercase border ${styles[quality]}`}>
      {icons[quality]}
      {quality}
    </span>
  );
}

// ── Sensor Chart ─────────────────────────────────────────────────

function SensorChart({
  config,
  data,
  isLoading,
  isExpanded,
  onToggleExpand,
}: {
  config: SensorChartConfig;
  data: ChartData[] | null;
  isLoading: boolean;
  isExpanded: boolean;
  onToggleExpand: () => void;
}) {
  const hasData = data && data.length > 0;

  // Compute quality and stats
  const quality = useMemo(() => {
    if (!hasData) return null;
    return analyzeDataQuality(data);
  }, [data, hasData]);

  // Min/max/mean
  const stats = useMemo(() => {
    if (!hasData) return null;
    const values = data.map((d) => d.value);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const mean = values.reduce((s, v) => s + v, 0) / values.length;
    return { min, max, mean };
  }, [data, hasData]);

  // Custom tooltip
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-zinc-900/95 border border-zinc-700/50 rounded-lg px-3 py-2 shadow-xl backdrop-blur-sm">
          <div className="text-[11px] text-zinc-400">
            {payload[0].payload.time.toFixed(2)}s
          </div>
          <div className="text-sm font-medium" style={{ color: config.color }}>
            {payload[0].value.toFixed(3)} {config.unit}
          </div>
        </div>
      );
    }
    return null;
  };

  const chartHeight = isExpanded ? 240 : 120;

  return (
    <div className="bg-zinc-900/60 border border-zinc-800/50 rounded-xl overflow-hidden transition-all duration-300">
      {/* Chart Header */}
      <div className="flex items-center justify-between px-4 py-3">
        <div className="flex items-center gap-2">
          <div
            className="p-1.5 rounded-md"
            style={{ backgroundColor: `${config.color}15`, color: config.color }}
          >
            {config.icon}
          </div>
          <div>
            <div className="text-xs font-medium text-zinc-200">{config.label}</div>
            <div className="text-[11px] text-zinc-500">{config.description}</div>
          </div>
        </div>
        <div className="flex items-center gap-3">
          {quality && <QualityBadge quality={quality.consistency} />}
          {hasData && (
            <div className="text-right">
              <div className="text-xs font-mono text-zinc-300">
                {data[data.length - 1].value.toFixed(1)}
              </div>
              <div className="text-[9px] text-zinc-500 uppercase">{config.unit}</div>
            </div>
          )}
          <button
            onClick={onToggleExpand}
            className="p-1 text-zinc-600 hover:text-zinc-300 transition-colors"
            title={isExpanded ? 'Collapse' : 'Expand'}
          >
            {isExpanded ? <ChevronUp size={14} /> : <Maximize2 size={12} />}
          </button>
        </div>
      </div>

      {/* Sampling rate & stats bar */}
      {hasData && quality && (
        <div className="flex items-center gap-4 px-4 pb-2 text-[11px] text-zinc-500">
          <div className="flex items-center gap-1">
            <Signal size={10} />
            <span className="font-mono">{quality.samplingRate.toFixed(0)}Hz</span>
          </div>
          <div className="flex items-center gap-1">
            <Gauge size={10} />
            <span>{data.length.toLocaleString()} samples</span>
          </div>
          {quality.gapCount > 0 && (
            <div className="flex items-center gap-1 text-amber-500">
              <AlertTriangle size={10} />
              <span>{quality.gapCount} gap{quality.gapCount !== 1 ? 's' : ''}</span>
            </div>
          )}
          {stats && (
            <>
              <span className="text-zinc-600">|</span>
              <span>min: <span className="font-mono text-zinc-400">{stats.min.toFixed(1)}</span></span>
              <span>max: <span className="font-mono text-zinc-400">{stats.max.toFixed(1)}</span></span>
              <span>avg: <span className="font-mono text-zinc-400">{stats.mean.toFixed(1)}</span></span>
            </>
          )}
        </div>
      )}

      {/* Chart Area */}
      <div style={{ height: `${chartHeight}px` }} className="w-full px-2 pb-2 transition-all duration-300">
        {isLoading ? (
          <div className="h-full flex items-center justify-center">
            <div className="w-4 h-4 border-2 border-zinc-600 border-t-zinc-300 rounded-full animate-spin" />
          </div>
        ) : hasData ? (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -20 }}>
              <defs>
                <linearGradient id={config.gradientId} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={config.color} stopOpacity={0.15} />
                  <stop offset="95%" stopColor={config.color} stopOpacity={0} />
                </linearGradient>
              </defs>
              <XAxis
                dataKey="time"
                tick={{ fontSize: 9, fill: '#52525b' }}
                tickLine={false}
                axisLine={{ stroke: '#27272a', strokeWidth: 1 }}
                tickFormatter={(v) => `${Math.floor(v)}s`}
                interval="preserveStartEnd"
                minTickGap={40}
              />
              <YAxis
                tick={{ fontSize: 9, fill: '#52525b' }}
                tickLine={false}
                axisLine={false}
                tickFormatter={(v) =>
                  v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v.toFixed(v < 10 ? 1 : 0)
                }
                width={45}
              />
              <Tooltip content={<CustomTooltip />} />
              <Area
                type="monotone"
                dataKey="value"
                stroke={config.color}
                strokeWidth={1.5}
                fill={`url(#${config.gradientId})`}
                fillOpacity={1}
                dot={false}
                activeDot={{
                  r: 3,
                  fill: config.color,
                  stroke: '#0a0a0b',
                  strokeWidth: 2,
                }}
              />
            </AreaChart>
          </ResponsiveContainer>
        ) : (
          <div className="h-full flex items-center justify-center text-zinc-600 text-xs">
            No data available
          </div>
        )}
      </div>
    </div>
  );
}

// ── Orientation Cube (animated) ──────────────────────────────────

function OrientationCube({ qw, qx, qy, qz }: { qw: number; qx: number; qy: number; qz: number }) {
  // Convert quaternion to rotation matrix for CSS transform
  const xx = qx * qx, yy = qy * qy, zz = qz * qz;
  const xy = qx * qy, xz = qx * qz, yz = qy * qz;
  const wx = qw * qx, wy = qw * qy, wz = qw * qz;

  const matrix = [
    1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy), 0,
    2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx), 0,
    2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy), 0,
    0, 0, 0, 1,
  ];

  const matrixStr = `matrix3d(${matrix.join(',')})`;

  return (
    <div className="relative w-20 h-20" style={{ perspective: '200px' }}>
      <div
        className="w-full h-full relative"
        style={{
          transformStyle: 'preserve-3d',
          transform: matrixStr,
          transition: 'transform 0.1s ease-out',
        }}
      >
        {/* Front face */}
        <div
          className="absolute inset-0 border border-indigo-500/40 bg-indigo-500/10 flex items-center justify-center text-[9px] font-bold text-indigo-400"
          style={{ transform: 'translateZ(40px)' }}
        >
          F
        </div>
        {/* Back face */}
        <div
          className="absolute inset-0 border border-zinc-700/40 bg-zinc-800/20 flex items-center justify-center text-[9px] font-bold text-zinc-600"
          style={{ transform: 'rotateY(180deg) translateZ(40px)' }}
        >
          B
        </div>
        {/* Right face */}
        <div
          className="absolute inset-0 border border-emerald-500/40 bg-emerald-500/10 flex items-center justify-center text-[9px] font-bold text-emerald-400"
          style={{ transform: 'rotateY(90deg) translateZ(40px)' }}
        >
          R
        </div>
        {/* Left face */}
        <div
          className="absolute inset-0 border border-amber-500/40 bg-amber-500/10 flex items-center justify-center text-[9px] font-bold text-amber-400"
          style={{ transform: 'rotateY(-90deg) translateZ(40px)' }}
        >
          L
        </div>
        {/* Top face */}
        <div
          className="absolute inset-0 border border-cyan-500/40 bg-cyan-500/10 flex items-center justify-center text-[9px] font-bold text-cyan-400"
          style={{ transform: 'rotateX(90deg) translateZ(40px)' }}
        >
          T
        </div>
        {/* Bottom face */}
        <div
          className="absolute inset-0 border border-red-500/40 bg-red-500/10 flex items-center justify-center text-[9px] font-bold text-red-400"
          style={{ transform: 'rotateX(-90deg) translateZ(40px)' }}
        >
          B
        </div>
      </div>
    </div>
  );
}

// ── Orientation Timeline ─────────────────────────────────────────

function OrientationPreview({ sessionId }: { sessionId: string }) {
  const [orientData, setOrientData] = useState<number[][] | null>(null);
  const [playIdx, setPlayIdx] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    getSensorData(sessionId, 'orientation').then((data) => {
      setOrientData(data);
      setPlayIdx(0);
    });
  }, [sessionId]);

  // Auto-play animation
  useEffect(() => {
    if (!isPlaying || !orientData || orientData.length === 0) return;
    intervalRef.current = setInterval(() => {
      setPlayIdx((i) => {
        const next = i + 1;
        if (next >= orientData.length) {
          setIsPlaying(false);
          return 0;
        }
        return next;
      });
    }, 16); // ~60fps
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [isPlaying, orientData]);

  if (!orientData || orientData.length === 0) return null;

  const current = orientData[playIdx] || orientData[0];
  const [_t, qw, qx, qy, qz] = current;

  return (
    <div className="bg-zinc-900/60 border border-zinc-800/50 rounded-xl p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-md bg-purple-500/15 text-purple-400">
            <Compass size={14} />
          </div>
          <div>
            <div className="text-xs font-medium text-zinc-200">Orientation Preview</div>
            <div className="text-[11px] text-zinc-500">Quaternion visualization from sensor data</div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className={`px-2.5 py-1 rounded-lg text-[11px] font-medium transition-all ${
              isPlaying
                ? 'bg-indigo-500/15 text-indigo-400 border border-indigo-500/30'
                : 'bg-zinc-800 text-zinc-400 border border-zinc-700/40 hover:text-zinc-200'
            }`}
          >
            {isPlaying ? 'Pause' : 'Play'}
          </button>
        </div>
      </div>

      <div className="flex items-center gap-6">
        {/* 3D Orientation Cube */}
        <div className="flex items-center justify-center p-4">
          <OrientationCube qw={qw} qx={qx} qy={qy} qz={qz} />
        </div>

        {/* Quaternion values */}
        <div className="flex-1 space-y-1.5">
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="flex justify-between px-2 py-1 bg-zinc-900/60 rounded">
              <span className="text-zinc-500">qw</span>
              <span className="font-mono text-zinc-300">{qw.toFixed(4)}</span>
            </div>
            <div className="flex justify-between px-2 py-1 bg-zinc-900/60 rounded">
              <span className="text-zinc-500">qx</span>
              <span className="font-mono text-zinc-300">{qx.toFixed(4)}</span>
            </div>
            <div className="flex justify-between px-2 py-1 bg-zinc-900/60 rounded">
              <span className="text-zinc-500">qy</span>
              <span className="font-mono text-zinc-300">{qy.toFixed(4)}</span>
            </div>
            <div className="flex justify-between px-2 py-1 bg-zinc-900/60 rounded">
              <span className="text-zinc-500">qz</span>
              <span className="font-mono text-zinc-300">{qz.toFixed(4)}</span>
            </div>
          </div>

          {/* Timeline scrubber */}
          <div className="mt-2">
            <input
              type="range"
              min={0}
              max={orientData.length - 1}
              value={playIdx}
              onChange={(e) => {
                setPlayIdx(Number(e.target.value));
                setIsPlaying(false);
              }}
              className="w-full h-1 bg-zinc-800 rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-2.5 [&::-webkit-slider-thumb]:h-2.5 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-purple-500 [&::-webkit-slider-thumb]:shadow-lg"
            />
            <div className="flex justify-between text-[9px] text-zinc-600 mt-0.5">
              <span>0s</span>
              <span className="font-mono">{current[0].toFixed(1)}s</span>
              <span>{orientData[orientData.length - 1][0].toFixed(1)}s</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Main Component ───────────────────────────────────────────────

export default function SensorPanel() {
  const { currentSession } = useSessionStore();
  const [summary, setSummary] = useState<SensorSummary | null>(null);
  const [chartData, setChartData] = useState<Record<string, ChartData[] | null>>({});
  const [isLoading, setIsLoading] = useState(false);
  const [loadingCharts, setLoadingCharts] = useState<Record<string, boolean>>({});
  const [expandedCharts, setExpandedCharts] = useState<Record<string, boolean>>({});

  const sessionId = currentSession?.id;

  const toggleChartExpand = useCallback((key: string) => {
    setExpandedCharts((prev) => ({ ...prev, [key]: !prev[key] }));
  }, []);

  const loadSensorData = useCallback(async () => {
    if (!sessionId) {
      setSummary(null);
      setChartData({});
      return;
    }

    setIsLoading(true);

    const s = await getSensorSummary(sessionId);
    setSummary(s);

    if (!s || s.sensors.length === 0) {
      setIsLoading(false);
      return;
    }

    const loading: Record<string, boolean> = {};
    CHART_CONFIGS.forEach((c) => {
      loading[c.key] = true;
    });
    setLoadingCharts(loading);

    const results: Record<string, ChartData[] | null> = {};

    await Promise.all(
      CHART_CONFIGS.map(async (config) => {
        try {
          const raw = await getSensorData(sessionId, config.key);
          if (raw && raw.length > 0) {
            results[config.key] = raw.map(([time, value]) => ({ time, value }));
          } else {
            results[config.key] = null;
          }
        } catch {
          results[config.key] = null;
        }
        setLoadingCharts((prev) => ({ ...prev, [config.key]: false }));
      }),
    );

    setChartData(results);
    setIsLoading(false);
  }, [sessionId]);

  useEffect(() => {
    loadSensorData();
  }, [loadSensorData]);

  const formatDuration = (seconds: number) => {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return m > 0 ? `${m}m ${s}s` : `${s}s`;
  };

  const hasSensors = summary && summary.sensors.length > 0;

  // Overall data quality
  const overallQuality = useMemo(() => {
    if (!hasSensors) return null;
    const totalSamples = summary.sensors.reduce((s, sensor) => s + sensor.samples, 0);
    const avgRate = summary.duration > 0
      ? Math.round(totalSamples / summary.sensors.length / summary.duration)
      : 0;
    return { totalSamples, avgRate };
  }, [summary, hasSensors]);

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden bg-[#0a0a0b]">
      {/* Header */}
      <div className="h-14 flex items-center justify-between px-6 border-b border-zinc-800/40 shrink-0">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-indigo-500/10">
            <Radio size={18} className="text-indigo-400" />
          </div>
          <div>
            <h1 className="text-sm font-semibold text-zinc-100">Sensor Data</h1>
            <p className="text-[11px] text-zinc-500">Samsung Sensor Logger visualization</p>
          </div>
        </div>

        {hasSensors && (
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 text-xs text-zinc-400">
              <Smartphone size={14} className="text-zinc-500" />
              <span className="font-medium text-zinc-300">{summary.device_name}</span>
            </div>
            <div className="flex items-center gap-2 text-xs text-zinc-400">
              <Clock size={14} className="text-zinc-500" />
              <span className="font-medium text-zinc-300">
                {formatDuration(summary.duration)}
              </span>
            </div>
            {overallQuality && (
              <div className="flex items-center gap-2 text-xs text-zinc-400">
                <Signal size={14} className="text-zinc-500" />
                <span className="font-medium text-zinc-300">
                  ~{overallQuality.avgRate}Hz avg
                </span>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-6 scrollbar-hide">
        {!sessionId ? (
          <div className="flex flex-col items-center justify-center h-full text-zinc-500">
            <Radio size={48} strokeWidth={1} className="text-zinc-700 mb-4" />
            <p className="text-sm font-medium text-zinc-400">No session selected</p>
            <p className="text-xs text-zinc-600 mt-1">Select a session to view sensor data</p>
          </div>
        ) : isLoading && !summary ? (
          <div className="flex flex-col items-center justify-center h-full text-zinc-500">
            <div className="w-6 h-6 border-2 border-zinc-600 border-t-zinc-300 rounded-full animate-spin mb-4" />
            <p className="text-xs">Loading sensor data...</p>
          </div>
        ) : !hasSensors ? (
          <div className="flex flex-col items-center justify-center h-full text-zinc-500">
            <Radio size={48} strokeWidth={1} className="text-zinc-700 mb-4" />
            <p className="text-sm font-medium text-zinc-400">No sensor data</p>
            <p className="text-xs text-zinc-600 mt-1">
              This session does not have Sensor Logger data
            </p>
          </div>
        ) : (
          <div className="space-y-6">
            {/* Sensor Summary Bar */}
            <div className="flex flex-wrap gap-2">
              {summary.sensors.map((sensor) => {
                const rate = sensor.duration > 0
                  ? Math.round(sensor.samples / sensor.duration)
                  : 0;
                return (
                  <div
                    key={sensor.name}
                    className="flex items-center gap-2 px-3 py-2 bg-zinc-900/60 border border-zinc-800/50 rounded-lg"
                  >
                    <div className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                    <span className="text-xs font-medium text-zinc-300 capitalize">
                      {sensor.name}
                    </span>
                    <span className="text-[11px] text-zinc-500 font-mono">
                      {sensor.samples.toLocaleString()} samples
                    </span>
                    <span className="text-[9px] text-zinc-600">
                      ({rate}Hz)
                    </span>
                  </div>
                );
              })}

              {/* Total data quality badge */}
              {overallQuality && (
                <div className="flex items-center gap-2 px-3 py-2 bg-zinc-900/60 border border-zinc-800/50 rounded-lg">
                  <Gauge size={12} className="text-zinc-500" />
                  <span className="text-[11px] text-zinc-400">
                    {overallQuality.totalSamples.toLocaleString()} total samples
                  </span>
                </div>
              )}
            </div>

            {/* Orientation Preview (if available) */}
            {sessionId && (
              <OrientationPreview sessionId={sessionId} />
            )}

            {/* 2x2 Chart Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {CHART_CONFIGS.map((config) => (
                <SensorChart
                  key={config.key}
                  config={config}
                  data={chartData[config.key] ?? null}
                  isLoading={loadingCharts[config.key] ?? false}
                  isExpanded={expandedCharts[config.key] ?? false}
                  onToggleExpand={() => toggleChartExpand(config.key)}
                />
              ))}
            </div>

            {/* Recording Info */}
            {summary.recording_time && (
              <div className="text-[11px] text-zinc-600 text-center mt-4">
                Recorded: {summary.recording_time.replace(/_/g, ' ')}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
