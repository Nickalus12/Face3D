import { ChevronRight } from 'lucide-react';
import useSessionStore from '../store/sessionStore';
import { type ViewId } from './Sidebar';

const VIEW_TABS: { id: ViewId; label: string }[] = [
  { id: 'view', label: '3D' },
  { id: 'metrics', label: 'Metrics' },
  { id: 'gallery', label: 'Gallery' },
];

const VIEW_LABELS: Record<string, string> = {
  view: '3D View',
  metrics: 'Metrics',
  gallery: 'Gallery',
  settings: 'Settings',
  home: 'Dashboard',
  pipeline: 'Pipeline',
};

interface TopBarProps {
  activeView: ViewId;
  onViewChange: (view: ViewId) => void;
}

export default function TopBar({ activeView, onViewChange }: TopBarProps) {
  const { currentSession } = useSessionStore();

  return (
    <div className="h-12 bg-surface-raised/90 backdrop-blur-md border-b border-border flex items-center justify-between px-8 shrink-0 z-30 select-none">
      {/* Left: Breadcrumb */}
      <div className="flex items-center gap-3 text-sm font-medium min-w-0">
        <button
          onClick={() => onViewChange('home')}
          className="text-zinc-500 hover:text-zinc-200 transition-all duration-150 shrink-0 hover:bg-white/[0.06] px-2.5 py-1 rounded-md active:scale-[0.97]"
        >
          Sessions
        </button>
        {currentSession && (
          <>
            <ChevronRight size={12} className="text-zinc-700 shrink-0" />
            <span className="text-zinc-200 truncate max-w-[180px] px-2 py-1">
              {currentSession.name}
            </span>
          </>
        )}
        <ChevronRight size={12} className="text-zinc-700 shrink-0" />
        <span className="text-indigo-400 shrink-0 px-2 py-1">
          {VIEW_LABELS[activeView] ?? activeView}
        </span>
      </div>

      {/* Center: View tabs */}
      <div className="flex items-center gap-3 p-1 rounded-xl bg-zinc-900/50 border border-border">
        {VIEW_TABS.map((tab) => {
          const isActive = activeView === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => onViewChange(tab.id)}
              className={`px-4 py-1.5 text-sm font-medium rounded-lg transition-all duration-200 whitespace-nowrap active:scale-[0.97] ${
                isActive
                  ? 'bg-indigo-500/20 text-indigo-300'
                  : 'text-zinc-500 hover:text-zinc-200'
              }`}
            >
              {tab.label}
            </button>
          );
        })}
      </div>

      {/* Right: empty spacer to keep center aligned */}
      <div className="min-w-0 flex-shrink" />
    </div>
  );
}
