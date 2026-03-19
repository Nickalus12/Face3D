import React, { useState } from 'react';
import { AreaChart, Area, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Activity } from 'lucide-react';

const mockData = Array.from({ length: 50 }, (_, i) => ({
  iter: i * 100,
  psnr: 20 + Math.log(i + 1) * 4 + Math.random(),
  loss: 0.5 * Math.exp(-i / 10) + Math.random() * 0.05,
  gaussians: 100000 + i * 25000 + Math.random() * 10000,
  vram: 4 + (i / 50) * 6 + Math.random() * 0.5
}));

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-zinc-900 border border-zinc-800 p-3 rounded-lg shadow-xl text-xs">
        <p className="text-zinc-500 mb-1">Iteration {label}</p>
        <p className="text-emerald-400 font-mono font-bold">
          {payload[0].name}: {payload[0].value.toFixed(4)}
        </p>
      </div>
    );
  }
  return null;
};

export const TrainingMetrics: React.FC = () => {
  const [timeRange, setTimeRange] = useState('all');

  return (
    <div className="flex-1 bg-[#0a0a0b] p-8 overflow-y-auto [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-thumb]:bg-zinc-800 [&::-webkit-scrollbar-track]:bg-transparent">
      
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-light text-zinc-100 flex items-center gap-3">
            <Activity className="text-emerald-500" />
            Training Metrics
          </h1>
          <p className="text-sm text-zinc-500 mt-1">Real-time performance overview</p>
        </div>

        <div className="flex gap-2 bg-zinc-900/50 p-1 rounded-lg border border-zinc-800">
          {['all', '1k', '500'].map(range => (
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
              <div className="text-2xl font-light text-zinc-100 mt-1">34.21 <span className="text-xs text-zinc-600 font-medium">dB</span></div>
            </div>
            <div className="w-20 h-8 opacity-50">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={mockData.slice(-10)}>
                  <Line type="monotone" dataKey="psnr" stroke="#10b981" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={mockData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                <XAxis dataKey="iter" stroke="#52525b" fontSize={10} tickLine={false} axisLine={false} />
                <YAxis stroke="#52525b" fontSize={10} tickLine={false} axisLine={false} domain={['auto', 'auto']} />
                <Tooltip content={<CustomTooltip />} />
                <Line type="monotone" dataKey="psnr" name="PSNR" stroke="#10b981" strokeWidth={2} dot={false} activeDot={{ r: 4, fill: "#10b981", stroke: "#0a0a0b", strokeWidth: 2 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Loss Chart */}
        <div className="bg-zinc-900/30 border border-zinc-800 rounded-2xl p-6">
          <div className="flex justify-between items-center mb-6">
            <div>
              <h3 className="text-sm font-semibold text-zinc-400 uppercase tracking-wider">L1 Loss</h3>
              <div className="text-2xl font-light text-zinc-100 mt-1">0.0412</div>
            </div>
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={mockData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                <XAxis dataKey="iter" stroke="#52525b" fontSize={10} tickLine={false} axisLine={false} />
                <YAxis stroke="#52525b" fontSize={10} tickLine={false} axisLine={false} />
                <Tooltip content={<CustomTooltip />} />
                <Line type="monotone" dataKey="loss" name="Loss" stroke="#f59e0b" strokeWidth={2} dot={false} activeDot={{ r: 4, fill: "#f59e0b", stroke: "#0a0a0b", strokeWidth: 2 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Gaussians Area */}
        <div className="bg-zinc-900/30 border border-zinc-800 rounded-2xl p-6">
          <div className="flex justify-between items-center mb-6">
            <div>
              <h3 className="text-sm font-semibold text-zinc-400 uppercase tracking-wider">Gaussians Count</h3>
              <div className="text-2xl font-light text-zinc-100 mt-1">1.24 <span className="text-xs text-zinc-600 font-medium">M</span></div>
            </div>
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={mockData}>
                <defs>
                  <linearGradient id="colorGaussians" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                <XAxis dataKey="iter" stroke="#52525b" fontSize={10} tickLine={false} axisLine={false} />
                <YAxis stroke="#52525b" fontSize={10} tickLine={false} axisLine={false} tickFormatter={(val) => `${(val/1000000).toFixed(1)}M`} />
                <Tooltip content={<CustomTooltip />} />
                <Area type="monotone" dataKey="gaussians" name="Count" stroke="#3b82f6" fillOpacity={1} fill="url(#colorGaussians)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* VRAM Area */}
        <div className="bg-zinc-900/30 border border-zinc-800 rounded-2xl p-6">
          <div className="flex justify-between items-center mb-6">
            <div>
              <h3 className="text-sm font-semibold text-zinc-400 uppercase tracking-wider">VRAM Usage</h3>
              <div className="text-2xl font-light text-zinc-100 mt-1">11.4 <span className="text-xs text-zinc-600 font-medium">GB</span></div>
            </div>
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={mockData}>
                <defs>
                  <linearGradient id="colorVram" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                <XAxis dataKey="iter" stroke="#52525b" fontSize={10} tickLine={false} axisLine={false} />
                <YAxis stroke="#52525b" fontSize={10} tickLine={false} axisLine={false} />
                <Tooltip content={<CustomTooltip />} />
                <Area type="monotone" dataKey="vram" name="VRAM" stroke="#8b5cf6" fillOpacity={1} fill="url(#colorVram)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

      </div>
    </div>
  );
};
