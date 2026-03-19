import React, { useState } from 'react';
import {
  AreaChart, Area, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts';
import { Activity } from 'lucide-react';
import usePipelineStore from '../store/pipelineStore';

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-zinc-900 border border-zinc-800 p-3 rounded-lg shadow-xl text-xs">
        <p className="text-zinc-500 mb-1">Iteration {label}</p>
        <p className="text-emerald-400 font-mono font-bold">
          {payload[0].name}: {typeof payload[0].value === 'number' ? payload[0].value.toFixed(4) : payload[0].value}
        </p>
      </div>
    );
  }
  return null;
};

const EmptyState = ({ label }: { label: string }) => (
  <div className="h-64 flex items-center justify-center">
    <div className="text-center">
      <div className="text-zinc-700 text-sm mb-1">No data yet</div>
      <div className="text-zinc-800 text-xs">
        {label} will appear once training begins
      </div>
    </div>
  </div>
);

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
  const latestPsnr = data.length > 0
    ? data.filter((d) => d.psnr != null).slice(-1)[0]?.psnr
    : null;
  const latestLoss = data.length > 0
    ? data.filter((d) => d.loss != null).slice(-1)[0]?.loss
    : null;
  const latestGaussians = data.length > 0
    ? data.filter((d) => d.gaussians != null).slice(-1)[0]?.gaussians
    : null;

  // VRAM from GPU info
  const vramUsedGB = gpuInfo?.memory_used
    ? (parseFloat(gpuInfo.memory_used) / 1024).toFixed(1)
    : null;

  const hasPsnr = data.some((d) => d.psnr != null);
  const hasLoss = data.some((d) => d.loss != null);
  const hasGaussians = data.some((d) => d.gaussians != null);

  return (
    <div className="flex-1 bg-[#0a0a0b] p-8 overflow-y-auto [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-thumb]:bg-zinc-800 [&::-webkit-scrollbar-track]:bg-transparent">

      {/* Header */}
      <div className="flex items-center justify-between mb-8">
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

        <div className="flex gap-2 bg-zinc-900/50 p-1 rounded-lg border border-zinc-800">
          {['all', '1k', '500'].map((range) => (
            <button
              key={range}
              onClick={() => setTimeRange(range)}
              className={`px-4 py-1.5 text-xs font-medium rounded-md transition-all ${
                timeRange === range
                  ? 'bg-zinc-800 text-zinc-100 shadow-sm'
                  : 'text-zinc-500 hover:text-zinc-300'
              }`}
            >
              {range === 'all' ? 'All Time' : `Last ${range}`}
            </button>
          ))}
        </div>
      </div>

      {/* Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

        {/* PSNR Chart */}
        <div className="bg-zinc-900/30 border border-zinc-800 rounded-2xl p-6">
          <div className="flex justify-between items-center mb-6">
            <div>
              <h3 className="text-sm font-semibold text-zinc-400 uppercase tracking-wider">PSNR</h3>
              <div className="text-2xl font-light text-zinc-100 mt-1">
                {latestPsnr != null ? latestPsnr.toFixed(2) : '--'}
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
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={data.filter((d) => d.psnr != null)}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                  <XAxis dataKey="iter" stroke="#52525b" fontSize={10} tickLine={false} axisLine={false} />
                  <YAxis stroke="#52525b" fontSize={10} tickLine={false} axisLine={false} domain={['auto', 'auto']} />
                  <Tooltip content={<CustomTooltip />} />
                  <Line
                    type="monotone" dataKey="psnr" name="PSNR"
                    stroke="#10b981" strokeWidth={2} dot={false}
                    activeDot={{ r: 4, fill: '#10b981', stroke: '#0a0a0b', strokeWidth: 2 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <EmptyState label="PSNR data" />
          )}
        </div>

        {/* Loss Chart */}
        <div className="bg-zinc-900/30 border border-zinc-800 rounded-2xl p-6">
          <div className="flex justify-between items-center mb-6">
            <div>
              <h3 className="text-sm font-semibold text-zinc-400 uppercase tracking-wider">L1 Loss</h3>
              <div className="text-2xl font-light text-zinc-100 mt-1">
                {latestLoss != null ? latestLoss.toFixed(4) : '--'}
              </div>
            </div>
          </div>
          {hasLoss ? (
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={data.filter((d) => d.loss != null)}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                  <XAxis dataKey="iter" stroke="#52525b" fontSize={10} tickLine={false} axisLine={false} />
                  <YAxis stroke="#52525b" fontSize={10} tickLine={false} axisLine={false} />
                  <Tooltip content={<CustomTooltip />} />
                  <Line
                    type="monotone" dataKey="loss" name="Loss"
                    stroke="#f59e0b" strokeWidth={2} dot={false}
                    activeDot={{ r: 4, fill: '#f59e0b', stroke: '#0a0a0b', strokeWidth: 2 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <EmptyState label="Loss data" />
          )}
        </div>

        {/* Gaussians Area */}
        <div className="bg-zinc-900/30 border border-zinc-800 rounded-2xl p-6">
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
            <div className="h-64">
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
        <div className="bg-zinc-900/30 border border-zinc-800 rounded-2xl p-6">
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
              <div className="text-center space-y-4 w-full px-4">
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
                  GPU Utilization: <span className="text-zinc-300">{gpuInfo.utilization}</span>
                </div>
              </div>
            ) : (
              <div className="text-center">
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
