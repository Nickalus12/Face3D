import { ChevronRight } from 'lucide-react';
import useSessionStore from '../store/sessionStore';
import { type ViewId } from './Sidebar';

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
    <div className="h-11 bg-[#0e0e11]/90 backdrop-blur-md border-b border-zinc-800/40 flex items-center px-6 shrink-0 z-30 select-none">
      {/* Breadcrumb only */}
      <div className="flex items-center gap-2 text-sm min-w-0">
        <button
          onClick={() => onViewChange('home')}
          className="text-zinc-500 hover:text-zinc-200 transition-colors duration-150 shrink-0 px-1.5 py-0.5 rounded hover:bg-white/[0.04]"
        >
          Sessions
        </button>
        {currentSession && (
          <>
            <ChevronRight size={11} className="text-zinc-700 shrink-0" />
            <span className="text-zinc-300 font-medium truncate max-w-[180px]">
              {currentSession.name}
            </span>
          </>
        )}
        <ChevronRight size={11} className="text-zinc-700 shrink-0" />
        <span className="text-zinc-500">
          {VIEW_LABELS[activeView] ?? activeView}
        </span>
      </div>
    </div>
  );
}
