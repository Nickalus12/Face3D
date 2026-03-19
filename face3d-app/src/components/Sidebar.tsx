import { useState, useCallback, useEffect } from 'react';
import { Home, GitMerge, Box, BarChart2, Image as ImageIcon, Settings, Activity, Columns2, Radio, Plus } from 'lucide-react';
import usePipelineStore from '../store/pipelineStore';
import useSessionStore from '../store/sessionStore';
import useSettingsStore from '../store/settingsStore';
import { getImageCounts } from '../lib/tauri';
import NewScanWizard from './NewScanWizard';

export type ViewId = 'home' | 'pipeline' | 'view' | 'metrics' | 'gallery' | 'compare' | 'sensors' | 'settings';

const NAV_ITEMS: { id: ViewId; icon: typeof Home; label: string; shortcut: string }[] = [
  { id: 'home', icon: Home, label: 'Dashboard', shortcut: 'Ctrl+1' },
  { id: 'pipeline', icon: GitMerge, label: 'Pipeline Config', shortcut: 'Ctrl+2' },
  { id: 'view', icon: Box, label: '3D Workspace', shortcut: 'Ctrl+3' },
  { id: 'metrics', icon: BarChart2, label: 'Performance Metrics', shortcut: 'Ctrl+4' },
  { id: 'gallery', icon: ImageIcon, label: 'Export Gallery', shortcut: '' },
  { id: 'compare', icon: Columns2, label: 'Compare Sessions', shortcut: 'Ctrl+5' },
  { id: 'sensors', icon: Radio, label: 'Sensor Data', shortcut: 'Ctrl+6' },
];

interface SidebarProps {
  activeView?: ViewId;
  onViewChange?: (view: ViewId) => void;
}

export default function Sidebar({ activeView, onViewChange }: SidebarProps) {
  const [localActive, setLocalActive] = useState<ViewId>('view');
  const [wizardOpen, setWizardOpen] = useState(false);
  const active = activeView ?? localActive;
  const setActive = useCallback(
    (v: ViewId) => {
      if (onViewChange) onViewChange(v);
      else setLocalActive(v);
    },
    [onViewChange],
  );

  const { gpuInfo, status, stages } = usePipelineStore();
  const { currentSession } = useSessionStore();
  const [imageCount, setImageCount] = useState(0);

  const isRunning = status === 'running';
  const completedStages = stages.filter((s) => s.status === 'complete').length;

  // Fetch image counts when session changes
  useEffect(() => {
    if (!currentSession) {
      setImageCount(0);
      return;
    }
    getImageCounts(currentSession.id).then(([renders, previews, frames]) => {
      setImageCount(renders + previews + frames);
    });
  }, [currentSession?.id]);

  const vramUsed = gpuInfo?.memory_used ? parseFloat(gpuInfo.memory_used) / 1024 : null;
  const vramTotal = gpuInfo?.memory_total ? parseFloat(gpuInfo.memory_total) / 1024 : null;
  const utilization = gpuInfo?.utilization ? parseInt(gpuInfo.utilization, 10) : 0;

  const gpuBarHeight = gpuInfo ? Math.max(5, utilization) : 0;

  return (
    <>
    <div className="w-[60px] h-full bg-[#08080a] border-r border-zinc-800/50 flex flex-col items-center py-4 z-50 shrink-0 relative">
      {/* App Logo */}
      <div className="w-10 h-10 bg-gradient-to-br from-indigo-500 via-purple-500 to-indigo-600 rounded-xl flex items-center justify-center shadow-lg shadow-indigo-500/20 mb-8 cursor-pointer hover:scale-105 active:scale-95 transition-all duration-300 ring-1 ring-white/10">
        <span className="text-white font-black text-[10px] tracking-tight">F3D</span>
      </div>

      {/* New Scan Button */}
      <div className="relative group w-full flex justify-center mb-5">
        <button
          onClick={() => setWizardOpen(true)}
          className="p-2.5 rounded-xl bg-indigo-600/20 text-indigo-400 hover:bg-indigo-600/30 hover:text-indigo-300 border border-indigo-500/20 hover:border-indigo-500/40 transition-all duration-200 group-active:scale-90 shadow-lg shadow-indigo-500/10"
        >
          <Plus size={18} strokeWidth={2.5} />
        </button>
        <div className="absolute left-[64px] top-1/2 -translate-y-1/2 px-2.5 py-1.5 bg-[#1c1c1f] text-zinc-100 text-[11px] font-medium tracking-wide rounded-md opacity-0 group-hover:opacity-100 -translate-x-2 group-hover:translate-x-0 pointer-events-none transition-all duration-200 delay-200 z-50 whitespace-nowrap shadow-xl shadow-black/30 border border-white/10">
          New Scan
        </div>
      </div>

      {/* Divider after New Scan */}
      <div className="w-7 h-px bg-zinc-700/40 mb-5" />

      {/* Main Navigation */}
      <nav className="flex-1 flex flex-col gap-2.5 w-full items-center">
        {NAV_ITEMS.map((item, index) => {
          const isActive = active === item.id;
          const Icon = item.icon;

          return (
            <div key={item.id}>
              {/* Divider between nav groups (after pipeline config) */}
              {index === 2 && (
                <div className="w-6 h-px bg-white/5 mx-auto my-1.5" />
              )}
              <div className="relative group w-full flex justify-center">
                {/* Active Indicator Bar - slides with transition */}
                <div
                  className={`absolute left-0 top-1/2 -translate-y-1/2 w-[3px] rounded-r-full transition-all duration-300 ease-out ${
                    isActive
                      ? 'h-5 bg-indigo-500 shadow-[0_0_12px_rgba(99,102,241,0.8)] opacity-100'
                      : 'h-0 bg-indigo-500 opacity-0'
                  }`}
                />

                <button
                  onClick={() => setActive(item.id)}
                  className={`relative p-2.5 rounded-xl transition-all duration-200 group-active:scale-90 ${
                    isActive
                      ? 'bg-indigo-500/10 text-indigo-400 shadow-[0_0_12px_rgba(99,102,241,0.1)]'
                      : 'text-zinc-500 hover:text-zinc-200 hover:bg-white/[0.04] hover:shadow-[0_0_8px_rgba(99,102,241,0.06)]'
                  }`}
                >
                  <Icon size={18} strokeWidth={isActive ? 2.5 : 2} />
                  {item.id === 'gallery' && imageCount > 0 && (
                    <span className="absolute -top-0.5 -right-0.5 min-w-[16px] h-[16px] flex items-center justify-center px-1 bg-indigo-500 text-white text-[8px] font-bold rounded-full leading-none">
                      {imageCount > 999 ? '999+' : imageCount}
                    </span>
                  )}
                </button>

                {/* Tooltip - slides in from left with delay */}
                <div className="absolute left-[60px] top-1/2 -translate-y-1/2 px-2.5 py-1.5 bg-[#1c1c1f] text-zinc-100 text-[11px] font-medium tracking-wide rounded-md opacity-0 group-hover:opacity-100 -translate-x-2 group-hover:translate-x-0 pointer-events-none transition-all duration-200 delay-200 z-50 whitespace-nowrap shadow-xl shadow-black/30 border border-white/10 flex items-center gap-2">
                  {item.label}
                  {item.shortcut && (
                    <span className="text-zinc-500 text-[9px] font-mono">{item.shortcut}</span>
                  )}
                  {item.id === 'pipeline' && isRunning && (
                    <span className="text-emerald-400 text-[9px] font-bold">{completedStages}/14</span>
                  )}
                  {item.id === 'gallery' && imageCount > 0 && (
                    <span className="text-indigo-400 text-[9px]">{imageCount} images</span>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </nav>

      {/* Bottom Actions & Mini GPU Indicator */}
      <div className="w-full flex flex-col items-center gap-4 mt-auto">
        {/* Divider */}
        <div className="w-6 h-px bg-white/5" />

        {/* Settings */}
        <div className="relative group w-full flex justify-center">
          {active === 'settings' && (
            <div className="absolute left-0 top-1/2 -translate-y-1/2 w-[3px] h-5 bg-indigo-500 rounded-r-full shadow-[0_0_12px_rgba(99,102,241,0.8)]" />
          )}
          <button
            onClick={() => setActive('settings')}
            className={`relative p-2.5 rounded-xl active:scale-90 transition-all duration-200 ${
              active === 'settings'
                ? 'bg-indigo-500/10 text-indigo-400 shadow-[0_0_12px_rgba(99,102,241,0.1)]'
                : 'text-zinc-500 hover:text-zinc-200 hover:bg-white/[0.04]'
            }`}
          >
            <Settings size={18} strokeWidth={active === 'settings' ? 2.5 : 2} />
            {useSettingsStore.getState().isDirty && (
              <span className="absolute top-1 right-1 w-2 h-2 bg-amber-400 rounded-full shadow-[0_0_6px_rgba(251,191,36,0.6)] animate-pulse" />
            )}
          </button>
          <div className="absolute left-[60px] top-1/2 -translate-y-1/2 px-2.5 py-1.5 bg-[#1c1c1f] text-zinc-100 text-[11px] font-medium tracking-wide rounded-md opacity-0 group-hover:opacity-100 -translate-x-2 group-hover:translate-x-0 pointer-events-none transition-all duration-200 delay-200 z-50 whitespace-nowrap shadow-xl shadow-black/30 border border-white/10 flex items-center gap-2">
            Settings
            {useSettingsStore.getState().isDirty && (
              <span className="text-amber-400 text-[9px]">unsaved</span>
            )}
          </div>
        </div>

        {/* Mini GPU Load Bar */}
        <div className="w-full px-2 flex flex-col items-center group relative cursor-help pb-1">
          <div className="text-[8px] font-bold text-zinc-600 mb-1.5 uppercase tracking-wider flex items-center gap-1">
            <Activity size={8} className={utilization > 80 ? 'text-red-400' : 'text-zinc-600'} />
            GPU
          </div>
          <div
            className={`w-1.5 h-10 bg-black rounded-full overflow-hidden flex flex-col justify-end border border-white/10 ${
              utilization > 80 ? 'animate-pulse-glow-red' : ''
            }`}
          >
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

          {/* GPU Tooltip */}
          <div className="absolute left-[60px] bottom-0 px-3 py-2 bg-[#1c1c1f] text-zinc-100 text-[11px] font-medium rounded-lg opacity-0 group-hover:opacity-100 -translate-x-2 group-hover:translate-x-0 pointer-events-none transition-all duration-200 delay-200 z-50 whitespace-nowrap border border-white/10 shadow-2xl shadow-black/40 flex flex-col gap-1">
            <div className="text-zinc-400 text-[10px] uppercase tracking-wider border-b border-white/5 pb-1 mb-1">
              Hardware Status
            </div>
            <div className="flex justify-between gap-4">
              <span>{gpuInfo?.name ?? 'No GPU detected'}</span>
              <span
                className={
                  utilization > 80
                    ? 'text-red-400'
                    : utilization > 50
                    ? 'text-amber-400'
                    : 'text-emerald-400'
                }
              >
                {gpuInfo ? `${utilization}%` : '--'}
              </span>
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

        {/* Version */}
        <div className="text-[8px] text-zinc-700 font-mono tracking-tight pb-1">v0.1.0</div>
      </div>
    </div>

    {/* New Scan Wizard Modal */}
    <NewScanWizard isOpen={wizardOpen} onClose={() => setWizardOpen(false)} />
    </>
  );
}
