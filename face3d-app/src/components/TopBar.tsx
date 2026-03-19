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

  // Color accents per tab for active state
  const TAB_COLORS: Record<string, { bg: string; text: string; border: string }> = {
    view: { bg: 'bg-violet-500/15', text: 'text-violet-300', border: 'border-violet-500/30' },
    metrics: { bg: 'bg-blue-500/15', text: 'text-blue-300', border: 'border-blue-500/30' },
    gallery: { bg: 'bg-emerald-500/15', text: 'text-emerald-300', border: 'border-emerald-500/30' },
    sensors: { bg: 'bg-amber-500/15', text: 'text-amber-300', border: 'border-amber-500/30' },
    compare: { bg: 'bg-rose-500/15', text: 'text-rose-300', border: 'border-rose-500/30' },
  };

  return (
    <div className="h-14 bg-surface-raised/90 backdrop-blur-md border-b border-border flex items-center justify-between px-8 shrink-0 z-30 select-none">
      {/* Left: Breadcrumb — more padding from edge */}
      <div className="flex items-center gap-2 text-sm min-w-0">
        <button
          onClick={() => onViewChange('home')}
          className="text-zinc-500 hover:text-zinc-200 transition-colors duration-150 shrink-0 hover:bg-white/[0.06] px-2.5 py-1 rounded-input"
        >
          Sessions
        </button>
        {currentSession && (
          <>
            <ChevronRight size={12} className="text-zinc-700 shrink-0" />
            <span className="text-zinc-200 font-medium truncate max-w-[160px] px-2 py-1">
              {currentSession.name}
            </span>
          </>
        )}
        <ChevronRight size={12} className="text-zinc-700 shrink-0" />
        <span className="text-indigo-400 font-medium shrink-0 px-2 py-1">
          {VIEW_LABELS[activeView] ?? activeView}
        </span>
      </div>

      {/* Center: View mode tabs — clearly separated buttons with distinct colors */}
      <div className="flex items-center gap-2 p-1.5 rounded-2xl bg-zinc-900/60 border border-border">
        {VIEW_TABS.map((tab) => {
          const isActive = activeView === tab.id;
          const colors = TAB_COLORS[tab.id] || TAB_COLORS.view;
          return (
            <button
              key={tab.id}
              onClick={() => onViewChange(tab.id)}
              className={`px-5 py-2 text-xs font-semibold rounded-button transition-all duration-200 whitespace-nowrap ${
                isActive
                  ? `${colors.bg} ${colors.text} border ${colors.border} shadow-md`
                  : 'text-zinc-500 hover:text-zinc-200 hover:bg-zinc-800/60 border border-transparent'
              }`}
            >
              {tab.label}
            </button>
          );
        })}
      </div>

      {/* Right: Action buttons — more spacing from edge */}
      <div className="flex items-center gap-3">
        <button
          onClick={onNewScan}
          className="flex items-center gap-2 px-4 py-2 text-xs font-semibold text-emerald-400 bg-emerald-500/10 hover:bg-emerald-500/20 border border-emerald-500/25 rounded-button transition-all duration-200 active:scale-95 hover:shadow-lg hover:shadow-emerald-500/10"
        >
          <Plus size={14} /> New Scan
        </button>
        <button
          onClick={onOpenFolder}
          className="p-2.5 text-zinc-500 hover:text-zinc-200 hover:bg-zinc-800/60 rounded-button transition-all duration-150 active:scale-95"
          title="Open project folder"
        >
          <FolderOpen size={16} />
        </button>
      </div>
    </div>
  );
}
