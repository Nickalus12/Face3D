import { ChevronRight, Plus, FolderOpen } from 'lucide-react';
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
    <div className="h-12 bg-zinc-900/80 backdrop-blur-sm border-b border-zinc-800/40 flex items-center justify-between px-5 shrink-0 z-30 select-none">
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

      {/* Center: View mode pill tabs — well-spaced, clearly separate */}
      <div className="flex items-center gap-1 bg-zinc-800/40 p-1 rounded-xl border border-zinc-700/40">
        {VIEW_TABS.map((tab) => {
          const isActive = activeView === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => onViewChange(tab.id)}
              className={`px-4 py-1.5 text-xs font-medium rounded-lg transition-all duration-150 whitespace-nowrap ${
                isActive
                  ? 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 shadow-sm'
                  : 'text-zinc-500 hover:text-zinc-200 hover:bg-zinc-700/40 border border-transparent'
              }`}
            >
              {tab.label}
            </button>
          );
        })}
      </div>

      {/* Right: Action buttons — no settings icon here (it's in the tab bar) */}
      <div className="flex items-center gap-2">
        <button
          onClick={onNewScan}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-emerald-400 bg-emerald-500/10 hover:bg-emerald-500/20 border border-emerald-500/20 rounded-lg transition-all duration-150 active:scale-95"
        >
          <Plus size={14} /> New Scan
        </button>
        <button
          onClick={onOpenFolder}
          className="p-1.5 text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800/60 rounded-lg transition-all duration-150 active:scale-95"
          title="Open project folder"
        >
          <FolderOpen size={15} />
        </button>
      </div>
    </div>
  );
}
