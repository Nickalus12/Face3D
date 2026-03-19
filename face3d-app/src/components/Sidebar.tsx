import { useState, useCallback } from 'react';
import { Home, GitMerge, Box, BarChart2, Image as ImageIcon, Settings } from 'lucide-react';
import usePipelineStore from '../store/pipelineStore';

export type ViewId = 'home' | 'pipeline' | 'view' | 'metrics' | 'gallery' | 'settings';

const NAV_ITEMS: { id: ViewId; icon: typeof Home; label: string; shortcut: string }[] = [
  { id: 'home', icon: Home, label: 'Dashboard', shortcut: 'Ctrl+1' },
  { id: 'pipeline', icon: GitMerge, label: 'Pipeline', shortcut: 'Ctrl+2' },
  { id: 'view', icon: Box, label: '3D View', shortcut: 'Ctrl+3' },
  { id: 'gallery', icon: ImageIcon, label: 'Gallery', shortcut: 'Ctrl+4' },
  { id: 'metrics', icon: BarChart2, label: 'Metrics', shortcut: 'Ctrl+5' },
];

interface SidebarProps {
  activeView?: ViewId;
  onViewChange?: (view: ViewId) => void;
}

export default function Sidebar({ activeView, onViewChange }: SidebarProps) {
  const [localActive, setLocalActive] = useState<ViewId>('view');
  const active = activeView ?? localActive;
  const setActive = useCallback(
    (v: ViewId) => {
      if (onViewChange) onViewChange(v);
      else setLocalActive(v);
    },
    [onViewChange],
  );

  const { gpuInfo } = usePipelineStore();
  const utilization = gpuInfo?.utilization ? parseInt(gpuInfo.utilization, 10) : 0;

  return (
    <div className="w-[68px] h-full bg-[#08080a] border-r border-border flex flex-col items-center py-5 z-50 shrink-0 relative">
      {/* App Logo */}
      <div className="w-10 h-10 bg-gradient-to-br from-indigo-500 via-purple-500 to-indigo-600 rounded-card flex items-center justify-center shadow-lg shadow-indigo-500/20 mb-8 cursor-pointer hover:scale-105 active:scale-95 transition-all duration-300 ring-1 ring-white/10">
        <span className="text-white font-black text-xs tracking-tight">F3D</span>
      </div>

      {/* Main Navigation */}
      <nav className="flex-1 flex flex-col gap-5 w-full items-center">
        {NAV_ITEMS.map((item) => {
          const isActive = active === item.id;
          const Icon = item.icon;

          return (
            <div key={item.id} className="relative group w-full flex justify-center">
              <button
                onClick={() => setActive(item.id)}
                className={`relative p-3 rounded-lg transition-all duration-200 group-active:scale-90 ${
                  isActive
                    ? 'bg-indigo-500/15 text-indigo-400 shadow-[0_0_16px_rgba(99,102,241,0.12)]'
                    : 'text-zinc-500 hover:text-zinc-200 hover:bg-white/[0.05]'
                }`}
              >
                <Icon size={20} strokeWidth={isActive ? 2.5 : 1.8} />
              </button>

              {/* Tooltip */}
              <div className="absolute left-[72px] top-1/2 -translate-y-1/2 px-2.5 py-1.5 bg-[#1c1c1f] text-zinc-100 text-xs font-medium tracking-wide rounded-md opacity-0 group-hover:opacity-100 -translate-x-2 group-hover:translate-x-0 pointer-events-none transition-all duration-200 delay-200 z-50 whitespace-nowrap shadow-xl shadow-black/30 border border-white/10 flex items-center gap-2">
                {item.label}
                {item.shortcut && (
                  <span className="text-zinc-500 text-[9px] font-mono">{item.shortcut}</span>
                )}
              </div>
            </div>
          );
        })}
      </nav>

      {/* Bottom: Settings + GPU */}
      <div className="w-full flex flex-col items-center gap-5 mt-auto">
        <div className="w-6 h-px bg-white/5" />

        {/* Settings */}
        <div className="relative group w-full flex justify-center">
          <button
            onClick={() => setActive('settings')}
            className={`relative p-3 rounded-lg transition-all duration-200 active:scale-90 ${
              active === 'settings'
                ? 'bg-indigo-500/15 text-indigo-400 shadow-[0_0_16px_rgba(99,102,241,0.12)]'
                : 'text-zinc-500 hover:text-zinc-200 hover:bg-white/[0.05]'
            }`}
          >
            <Settings size={20} strokeWidth={active === 'settings' ? 2.5 : 1.8} />
          </button>
          <div className="absolute left-[72px] top-1/2 -translate-y-1/2 px-2.5 py-1.5 bg-[#1c1c1f] text-zinc-100 text-xs font-medium tracking-wide rounded-md opacity-0 group-hover:opacity-100 -translate-x-2 group-hover:translate-x-0 pointer-events-none transition-all duration-200 delay-200 z-50 whitespace-nowrap shadow-xl shadow-black/30 border border-white/10">
            Settings
          </div>
        </div>

        {/* Simplified GPU indicator */}
        {gpuInfo && (
          <div
            className={`text-[10px] font-semibold tabular-nums pb-1 ${
              utilization > 80
                ? 'text-red-400'
                : utilization > 50
                ? 'text-amber-400'
                : 'text-zinc-600'
            }`}
          >
            GPU {utilization}%
          </div>
        )}
      </div>
    </div>
  );
}
