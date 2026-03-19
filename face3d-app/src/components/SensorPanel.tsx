import { useEffect, useState, useCallback } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { Activity, Smartphone, Clock, Radio, Sun, Wind, RotateCcw } from 'lucide-react';
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
}

const CHART_CONFIGS: SensorChartConfig[] = [
  {
    key: 'accel_magnitude',
    label: 'Accelerometer',
    unit: 'm/s\u00B2',
    color: '#6366f1',
    icon: <Activity size={14} />,
    description: 'Camera movement intensity',
  },
  {
    key: 'gyro_magnitude',
    label: 'Gyroscope',
    unit: '\u00B0/s',
    color: '#f59e0b',
    icon: <RotateCcw size={14} />,
    description: 'Rotation speed',
  },
  {
    key: 'barometer',
    label: 'Barometer',
    unit: 'hPa',
    color: '#10b981',
    icon: <Wind size={14} />,
    description: 'Atmospheric pressure',
  },
  {
    key: 'light',
    label: 'Light Sensor',
    unit: 'lux',
    color: '#eab308',
    icon: <Sun size={14} />,
    description: 'Ambient light level',
  },
];

function MiniChart({
  config,
  data,
  isLoading,
}: {
  config: SensorChartConfig;
  data: ChartData[] | null;
  isLoading: boolean;
}) {
  const hasData = data && data.length > 0;

  // Custom tooltip
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-zinc-900/95 border border-zinc-700/50 rounded-lg px-3 py-2 shadow-xl backdrop-blur-sm">
          <div className="text-[10px] text-zinc-400">
            {payload[0].payload.time.toFixed(1)}s
          </div>
          <div className="text-sm font-medium" style={{ color: config.color }}>
            {payload[0].value.toFixed(2)} {config.unit}
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="bg-zinc-900/60 border border-zinc-800/50 rounded-xl p-3 flex flex-col gap-2">
      {/* Chart Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div
            className="p-1.5 rounded-md"
            style={{ backgroundColor: `${config.color}15`, color: config.color }}
          >
            {config.icon}
          </div>
          <div>
            <div className="text-xs font-medium text-zinc-200">{config.label}</div>
            <div className="text-[10px] text-zinc-500">{config.description}</div>
          </div>
        </div>
        {hasData && (
          <div className="text-right">
            <div className="text-xs font-mono text-zinc-300">
              {data[data.length - 1].value.toFixed(1)}
            </div>
            <div className="text-[9px] text-zinc-500 uppercase">{config.unit}</div>
          </div>
        )}
      </div>

      {/* Chart Area */}
      <div className="h-[120px] w-full">
        {isLoading ? (
          <div className="h-full flex items-center justify-center">
            <div className="w-4 h-4 border-2 border-zinc-600 border-t-zinc-300 rounded-full animate-spin" />
          </div>
        ) : hasData ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: -20 }}>
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
              <Line
                type="monotone"
                dataKey="value"
                stroke={config.color}
                strokeWidth={1.5}
                dot={false}
                activeDot={{
                  r: 3,
                  fill: config.color,
                  stroke: '#0a0a0b',
                  strokeWidth: 2,
                }}
              />
            </LineChart>
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

export default function SensorPanel() {
  const { currentSession } = useSessionStore();
  const [summary, setSummary] = useState<SensorSummary | null>(null);
  const [chartData, setChartData] = useState<Record<string, ChartData[] | null>>({});
  const [isLoading, setIsLoading] = useState(false);
  const [loadingCharts, setLoadingCharts] = useState<Record<string, boolean>>({});

  const sessionId = currentSession?.id;

  const loadSensorData = useCallback(async () => {
    if (!sessionId) {
      setSummary(null);
      setChartData({});
      return;
    }

    setIsLoading(true);

    // Fetch summary
    const s = await getSensorSummary(sessionId);
    setSummary(s);

    if (!s || s.sensors.length === 0) {
      setIsLoading(false);
      return;
    }

    // Fetch all chart data in parallel
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
            <p className="text-[10px] text-zinc-500">Samsung Sensor Logger visualization</p>
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
              {summary.sensors.map((sensor) => (
                <div
                  key={sensor.name}
                  className="flex items-center gap-2 px-3 py-1.5 bg-zinc-900/60 border border-zinc-800/50 rounded-lg"
                >
                  <div className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                  <span className="text-[11px] font-medium text-zinc-300 capitalize">
                    {sensor.name}
                  </span>
                  <span className="text-[10px] text-zinc-500 font-mono">
                    {sensor.samples.toLocaleString()}
                  </span>
                </div>
              ))}
            </div>

            {/* 2x2 Chart Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {CHART_CONFIGS.map((config) => (
                <MiniChart
                  key={config.key}
                  config={config}
                  data={chartData[config.key] ?? null}
                  isLoading={loadingCharts[config.key] ?? false}
                />
              ))}
            </div>

            {/* Recording Info */}
            {summary.recording_time && (
              <div className="text-[10px] text-zinc-600 text-center mt-4">
                Recorded: {summary.recording_time.replace(/_/g, ' ')}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
