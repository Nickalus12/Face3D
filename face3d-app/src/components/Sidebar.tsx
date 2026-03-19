import { useState } from 'react';
import { Home, GitMerge, Box, BarChart2, Image as ImageIcon, Settings } from 'lucide-react';
import usePipelineStore from '../store/pipelineStore';

const NAV_ITEMS = [
  { id: 'home', icon: Home, label: 'Dashboard' },
  { id: 'pipeline', icon: GitMerge, label: 'Pipeline Config' },
  { id: 'view', icon: Box, label: '3D Workspace' },
  { id: 'metrics', icon: BarChart2, label: 'Performance Metrics' },
  { id: 'gallery', icon: ImageIcon, label: 'Export Gallery' },
];

export default function Sidebar() {
  const [active, setActive] = useState('view');
  const { gpuInfo, status, stages } = usePipelineStore();

  const isRunning = status === 'running';
  const completedStages = stages.filter((s) => s.status === 'complete').length;

  // Parse VRAM used from gpuInfo string like "4200 MiB"
  const vramUsed = gpuInfo?.memory_used
    ? parseFloat(gpuInfo.memory_used) / 1024
    : null;
  const vramTotal = gpuInfo?.memory_total
    ? parseFloat(gpuInfo.memory_total) / 1024
    : null;
  const utilization = gpuInfo?.utilization
    ? parseInt(gpuInfo.utilization, 10)
    : 0;

  // GPU bar height as a percentage
  const gpuBarHeight = gpuInfo ? Math.max(5, utilization) : 0;

  return (
    <div className="w-[56px] h-full bg-[#0a0a0b] border-r border-white/5 flex flex-col items-center py-5 z-40 shrink-0 relative">
      {/* App Logo */}
      <div className="w-9 h-9 bg-gradient-to-br from-indigo-500 via-purple-500 to-indigo-600 rounded-[10px] flex items-center justify-center shadow-lg shadow-indigo-500/20 mb-8 cursor-pointer hover:scale-105 transition-all duration-300 ring-1 ring-white/10">
        <span className="text-white font-black text-sm tracking-tighter">F3</span>
      </div>

      {/* Main Navigation */}
      <nav className="flex-1 flex flex-col gap-2 w-full items-center">
        {NAV_ITEMS.map((item) => {
          const isActive = active === item.id;
          const Icon = item.icon;
          return (
            <div key={item.id} className="relative group w-full flex justify-center">
              {/* Active Indicator Line */}
              {isActive && (
                <div className="absolute left-0 top-1/2 -translate-y-1/2 w-[3px] h-5 bg-indigo-500 rounded-r-full shadow-[0_0_12px_rgba(99,102,241,0.8)]" />
              )}

              <button
                onClick={() => setActive(item.id)}
                className={`p-2.5 rounded-xl transition-all duration-200 group-active:scale-95 ${
                  isActive
                    ? 'bg-indigo-500/10 text-indigo-400'
                    : 'text-zinc-500 hover:text-zinc-200 hover:bg-white/5'
                }`}
              >
                <Icon size={18} strokeWidth={isActive ? 2.5 : 2} />
              </button>

              {/* Tooltip */}
              <div className="absolute left-[60px] top-1/2 -translate-y-1/2 px-2.5 py-1.5 bg-[#1c1c1f] text-zinc-100 text-[11px] font-medium tracking-wide rounded-md opacity-0 group-hover:opacity-100 -translate-x-2 group-hover:translate-x-0 pointer-events-none transition-all duration-200 z-50 whitespace-nowrap shadow-xl border border-white/10 flex items-center gap-2">
                {item.label}
                {item.id === 'pipeline' && isRunning && (
                  <span className="text-emerald-400 text-[9px]">{completedStages}/14</span>
                )}
              </div>
            </div>
          );
        })}
      </nav>

      {/* Bottom Actions & Mini GPU Indicator */}
      <div className="w-full flex flex-col items-center gap-4 mt-auto">
        <div className="relative group w-full flex justify-center">
          <button className="p-2.5 rounded-xl text-zinc-500 hover:text-zinc-200 hover:bg-white/5 transition-all duration-200">
            <Settings size={18} strokeWidth={2} />
          </button>
          <div className="absolute left-[60px] top-1/2 -translate-y-1/2 px-2.5 py-1.5 bg-[#1c1c1f] text-zinc-100 text-[11px] font-medium tracking-wide rounded-md opacity-0 group-hover:opacity-100 -translate-x-2 group-hover:translate-x-0 pointer-events-none transition-all duration-200 z-50 whitespace-nowrap shadow-xl border border-white/10">
            Settings
          </div>
        </div>

        {/* Mini GPU Load Bar */}
        <div className="w-full px-2 flex flex-col items-center group relative cursor-help pb-2">
          <div className="text-[8px] font-bold text-zinc-600 mb-1.5 uppercase tracking-wider">GPU</div>
          <div className="w-1.5 h-10 bg-black rounded-full overflow-hidden flex flex-col justify-end border border-white/10">
            <div
              className={`w-full transition-all duration-500 ${
                utilization > 80
                  ? 'bg-gradient-to-t from-red-500 via-amber-400 to-amber-400 shadow-[0_0_10px_rgba(239,68,68,0.5)]'
                  : utilization > 50
                  ? 'bg-gradient-to-t from-amber-500 via-amber-400 to-emerald-400 shadow-[0_0_10px_rgba(245,158,11,0.5)]'
                  : 'bg-gradient-to-t from-emerald-500 via-emerald-400 to-emerald-300 shadow-[0_0_10px_rgba(52,211,153,0.5)]'
              }`}
              style={{ height: `${gpuBarHeight}%` }}
            />
          </div>

          <div className="absolute left-[60px] bottom-0 px-3 py-2 bg-[#1c1c1f] text-zinc-100 text-[11px] font-medium rounded-lg opacity-0 group-hover:opacity-100 -translate-x-2 group-hover:translate-x-0 pointer-events-none transition-all duration-200 z-50 whitespace-nowrap border border-white/10 shadow-2xl flex flex-col gap-1">
            <div className="text-zinc-400 text-[10px] uppercase tracking-wider border-b border-white/5 pb-1 mb-1">
              Hardware Status
            </div>
            <div className="flex justify-between gap-4">
              <span>{gpuInfo?.name ?? 'No GPU detected'}</span>
              <span className="text-emerald-400">{gpuInfo ? `${utilization}%` : '--'}</span>
            </div>
            <div className="flex justify-between gap-4">
              <span>VRAM Usage</span>
              <span className="text-zinc-300">
                {vramUsed !== null && vramTotal !== null
                  ? `${vramUsed.toFixed(1)} / ${vramTotal.toFixed(0)} GB`
                  : '--'}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
