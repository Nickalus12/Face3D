import { useEffect, useState, useCallback } from 'react';
import type { ViewId } from './Sidebar';

interface KeyboardShortcutsProps {
  onViewChange: (view: ViewId) => void;
  onToggleConsole: () => void;
  onNewScan?: () => void;
  onStartStop?: () => void;
  onOpenSessionFolder?: () => void;
}

interface Shortcut {
  keys: string;
  label: string;
  category: string;
}

const SHORTCUTS: Shortcut[] = [
  { keys: 'Ctrl+1', label: 'Dashboard', category: 'Navigation' },
  { keys: 'Ctrl+2', label: 'Pipeline', category: 'Navigation' },
  { keys: 'Ctrl+3', label: '3D View', category: 'Navigation' },
  { keys: 'Ctrl+4', label: 'Gallery', category: 'Navigation' },
  { keys: 'Ctrl+5', label: 'Metrics', category: 'Navigation' },
  { keys: 'Ctrl+6', label: 'Analyze', category: 'Navigation' },
  { keys: 'Ctrl+`', label: 'Toggle Console', category: 'Panels' },
  { keys: 'Ctrl+N', label: 'New Scan Wizard', category: 'Actions' },
  { keys: 'Space', label: 'Start / Stop Pipeline', category: 'Actions' },
  { keys: 'Ctrl+Shift+O', label: 'Open Session Folder', category: 'Actions' },
  { keys: 'Ctrl+?', label: 'Show Keyboard Shortcuts', category: 'Help' },
];

export default function KeyboardShortcuts({
  onViewChange,
  onToggleConsole,
  onNewScan,
  onStartStop,
  onOpenSessionFolder,
}: KeyboardShortcutsProps) {
  const [showHelp, setShowHelp] = useState(false);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable) {
        return;
      }

      const ctrl = e.ctrlKey || e.metaKey;
      const shift = e.shiftKey;

      if (ctrl && !shift) {
        const viewMap: Record<string, ViewId> = {
          '1': 'home',
          '2': 'pipeline',
          '3': 'view',
          '4': 'gallery',
          '5': 'metrics',
          '6': 'analyze',
        };
        if (viewMap[e.key]) {
          e.preventDefault();
          onViewChange(viewMap[e.key]);
          return;
        }
      }

      if (ctrl && e.key === '`') {
        e.preventDefault();
        onToggleConsole();
        return;
      }

      if (ctrl && !shift && e.key === 'n') {
        e.preventDefault();
        onNewScan?.();
        return;
      }

      if (e.key === ' ' && !ctrl && !shift) {
        e.preventDefault();
        onStartStop?.();
        return;
      }

      if (ctrl && shift && (e.key === 'o' || e.key === 'O')) {
        e.preventDefault();
        onOpenSessionFolder?.();
        return;
      }

      if (ctrl && shift && e.key === '?') {
        e.preventDefault();
        setShowHelp((prev) => !prev);
        return;
      }

      if (e.key === 'Escape' && showHelp) {
        setShowHelp(false);
        return;
      }
    },
    [onViewChange, onToggleConsole, onNewScan, onStartStop, onOpenSessionFolder, showHelp],
  );

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  if (!showHelp) return null;

  const categories = SHORTCUTS.reduce((acc, s) => {
    if (!acc[s.category]) acc[s.category] = [];
    acc[s.category].push(s);
    return acc;
  }, {} as Record<string, Shortcut[]>);

  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center animate-fadeIn"
      onClick={() => setShowHelp(false)}
    >
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" />
      <div
        className="relative bg-[#111113] border border-zinc-800/60 rounded-2xl shadow-2xl shadow-black/50 w-[420px] max-h-[70vh] overflow-hidden animate-scaleIn"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="px-6 pt-5 pb-4 border-b border-zinc-800/40">
          <h2 className="text-zinc-100 text-base font-semibold">Keyboard Shortcuts</h2>
          <p className="text-zinc-500 text-xs mt-1">Quick navigation and actions</p>
        </div>
        <div className="px-6 py-4 space-y-5 overflow-y-auto scrollbar-hide max-h-[50vh]">
          {Object.entries(categories).map(([category, shortcuts]) => (
            <div key={category}>
              <div className="text-[11px] font-bold text-zinc-500 uppercase tracking-wider mb-2">
                {category}
              </div>
              <div className="space-y-1">
                {shortcuts.map((s) => (
                  <div
                    key={s.keys}
                    className="flex items-center justify-between py-1.5 px-2 rounded-lg hover:bg-white/[0.02] transition-colors"
                  >
                    <span className="text-zinc-300 text-sm">{s.label}</span>
                    <kbd className="px-2 py-0.5 bg-zinc-800/80 border border-zinc-700/50 rounded text-xs text-zinc-400 font-mono tracking-tight">
                      {s.keys}
                    </kbd>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
        <div className="px-6 py-3 border-t border-zinc-800/40 flex justify-end">
          <button
            onClick={() => setShowHelp(false)}
            className="px-3 py-1.5 text-xs text-zinc-400 hover:text-zinc-200 bg-zinc-800/50 hover:bg-zinc-800 border border-zinc-700/50 rounded-lg transition-all duration-150 active:scale-95"
          >
            Close <span className="text-zinc-600 ml-1">Esc</span>
          </button>
        </div>
      </div>
    </div>
  );
}
