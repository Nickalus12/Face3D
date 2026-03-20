import { useState, useCallback } from 'react';
import { Home, Workflow, Cuboid, Images, ChartLine, ScanSearch, Settings } from 'lucide-react';
import usePipelineStore from '../store/pipelineStore';

export type ViewId = 'home' | 'pipeline' | 'view' | 'gallery' | 'metrics' | 'analyze' | 'settings';

const NAV_ITEMS: { id: ViewId; icon: typeof Home; label: string; shortcut: string }[] = [
  { id: 'home', icon: Home, label: 'Dashboard', shortcut: 'Ctrl+1' },
  { id: 'pipeline', icon: Workflow, label: 'Pipeline', shortcut: 'Ctrl+2' },
  { id: 'view', icon: Cuboid, label: '3D View', shortcut: 'Ctrl+3' },
  { id: 'gallery', icon: Images, label: 'Gallery', shortcut: 'Ctrl+4' },
  { id: 'metrics', icon: ChartLine, label: 'Metrics', shortcut: 'Ctrl+5' },
  { id: 'analyze', icon: ScanSearch, label: 'Analyze', shortcut: 'Ctrl+6' },
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

  const { status } = usePipelineStore();
  const isRunning = status === 'running';

  return (
    <div className="w-[56px] h-full bg-[#08080a] border-r border-zinc-800/30 flex flex-col items-center py-4 z-50 shrink-0">
      {/* Navigation */}
      <nav className="flex-1 flex flex-col gap-3 w-full items-center pt-2">
        {NAV_ITEMS.map((item) => {
          const isActive = active === item.id;
          const Icon = item.icon;
          const showPulse = item.id === 'pipeline' && isRunning;

          return (
            <div key={item.id} className="relative group w-full flex justify-center">
              <button
                onClick={() => setActive(item.id)}
                aria-label={item.label}
                className={`relative p-2.5 rounded-lg transition-all duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500/50 ${
                  isActive
                    ? 'bg-indigo-500/15 text-white'
                    : 'text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800/40'
                }`}
              >
                <Icon size={18} strokeWidth={isActive ? 2.2 : 1.6} />
                {showPulse && (
                  <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-indigo-500 rounded-full animate-pulse" />
                )}
              </button>

              {/* Tooltip */}
              <div className="absolute left-[60px] top-1/2 -translate-y-1/2 px-2 py-1 bg-zinc-900 text-zinc-200 text-xs rounded-md opacity-0 group-hover:opacity-100 -translate-x-1 group-hover:translate-x-0 pointer-events-none transition-all duration-150 delay-150 z-50 whitespace-nowrap shadow-lg border border-zinc-800">
                {item.label}
              </div>
            </div>
          );
        })}
      </nav>

      {/* Settings at bottom */}
      <div className="w-full flex flex-col items-center">
        <div className="relative group w-full flex justify-center">
          <button
            onClick={() => setActive('settings')}
            aria-label="Settings"
            className={`p-2.5 rounded-lg transition-all duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500/50 ${
              active === 'settings'
                ? 'bg-indigo-500/15 text-white'
                : 'text-zinc-600 hover:text-zinc-300 hover:bg-zinc-800/40'
            }`}
          >
            <Settings size={18} strokeWidth={active === 'settings' ? 2.2 : 1.6} />
          </button>
          <div className="absolute left-[60px] top-1/2 -translate-y-1/2 px-2 py-1 bg-zinc-900 text-zinc-200 text-xs rounded-md opacity-0 group-hover:opacity-100 -translate-x-1 group-hover:translate-x-0 pointer-events-none transition-all duration-150 delay-150 z-50 whitespace-nowrap shadow-lg border border-zinc-800">
            Settings
          </div>
        </div>
      </div>
    </div>
  );
}
