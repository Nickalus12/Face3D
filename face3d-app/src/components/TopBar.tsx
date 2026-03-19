import { ChevronRight, Plus, FolderOpen, Settings } from 'lucide-react';
import useSessionStore from '../store/sessionStore';
import { type ViewId } from './Sidebar';

// ── View mode config ────────────────────────────────────────────

const VIEW_TABS: { id: ViewId; label: string }[] = [
  { id: 'view', label: '3D' },
  { id: 'metrics', label: 'Metrics' },
  { id: 'gallery', label: 'Gallery' },
  { id: 'sensors', label: 'Sensors' },
  { id: 'compare', label: 'Compare' },
  { id: 'settings', label: 'Settings' },
];

const VIEW_LABELS: Record<string, string> = {
  view: '3D View',
  metrics: 'Metrics',
  gallery: 'Gallery',
  sensors: 'Sensors',
  compare: 'Compare',
  settings: 'Settings',
  home: 'Dashboard',
  pipeline: 'Pipeline',
};

interface TopBarProps {
  activeView: ViewId;
  onViewChange: (view: ViewId) => void;
  onNewScan: () => void;
  onOpenFolder: () => void;
}

export default function TopBar({ activeView, onViewChange, onNewScan, onOpenFolder }: TopBarProps) {
  const { currentSession } = useSessionStore();

  return (
    <div className="h-10 bg-zinc-900 border-b border-zinc-800/40 flex items-center justify-between px-4 shrink-0 z-30 select-none">
      {/* Left: Breadcrumb */}
      <div className="flex items-center gap-1.5 text-xs min-w-0">
        <button
          onClick={() => onViewChange('home')}
          className="text-zinc-500 hover:text-zinc-300 transition-colors duration-150 shrink-0"
        >
          Sessions
        </button>
        {currentSession && (
          <>
            <ChevronRight size={12} className="text-zinc-700 shrink-0" />
            <span className="text-zinc-300 font-medium truncate max-w-[120px]">
              {currentSession.name}
            </span>
          </>
        )}
        <ChevronRight size={12} className="text-zinc-700 shrink-0" />
        <span className="text-zinc-400 shrink-0">
          {VIEW_LABELS[activeView] ?? activeView}
        </span>
      </div>

      {/* Center: View mode pill tabs */}
      <div className="flex items-center gap-0.5 bg-zinc-800/50 p-0.5 rounded-lg border border-zinc-800/60">
        {VIEW_TABS.map((tab) => {
          const isActive = activeView === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => onViewChange(tab.id)}
              className={`px-3 py-1 text-[11px] font-medium rounded-md transition-all duration-150 ${
                isActive
                  ? 'bg-zinc-700/80 text-zinc-100 shadow-sm'
                  : 'text-zinc-500 hover:text-zinc-300 hover:bg-white/[0.04]'
              }`}
            >
              {tab.label}
            </button>
          );
        })}
      </div>

      {/* Right: Action buttons */}
      <div className="flex items-center gap-1.5">
        <button
          onClick={onNewScan}
          className="flex items-center gap-1.5 px-2.5 py-1 text-[11px] font-medium text-zinc-400 hover:text-zinc-200 hover:bg-white/[0.04] rounded-md transition-all duration-150 active:scale-95"
        >
          <Plus size={13} /> New Scan
        </button>
        <button
          onClick={onOpenFolder}
          className="p-1.5 text-zinc-500 hover:text-zinc-300 hover:bg-white/[0.04] rounded-md transition-all duration-150 active:scale-95"
          title="Open project folder"
        >
          <FolderOpen size={14} />
        </button>
        <button
          onClick={() => onViewChange('settings')}
          className="p-1.5 text-zinc-500 hover:text-zinc-300 hover:bg-white/[0.04] rounded-md transition-all duration-150 active:scale-95"
          title="Settings"
        >
          <Settings size={14} />
        </button>
      </div>
    </div>
  );
}
