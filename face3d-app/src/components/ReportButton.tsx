import React, { useState, useEffect } from 'react';
import { FileText } from 'lucide-react';
import { getReportPath } from '../lib/tauri';

function isTauri(): boolean {
  return typeof window !== 'undefined' && !!(window as any).__TAURI_INTERNALS__;
}

interface ReportButtonProps {
  sessionId: string;
}

export const ReportButton: React.FC<ReportButtonProps> = ({ sessionId }) => {
  const [reportPath, setReportPath] = useState<string | null>(null);

  useEffect(() => {
    if (!sessionId) {
      setReportPath(null);
      return;
    }
    getReportPath(sessionId).then(setReportPath);
  }, [sessionId]);

  if (!reportPath) return null;

  const handleClick = async () => {
    if (!isTauri()) return;
    try {
      // Use the Tauri opener plugin to open in default browser
      const { openUrl } = await import('@tauri-apps/plugin-opener');
      await openUrl(`file:///${reportPath.replace(/\\/g, '/')}`);
    } catch (e) {
      console.error('Failed to open report:', e);
      // Fallback: try shell open
      try {
        const { invoke } = await import('@tauri-apps/api/core');
        await invoke('open_folder', { path: reportPath });
      } catch {
        // silently fail
      }
    }
  };

  return (
    <button
      onClick={handleClick}
      className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-zinc-400 hover:text-zinc-200 bg-white/[0.03] hover:bg-white/[0.06] border border-zinc-800 hover:border-zinc-700 rounded-lg transition-all"
      title="Open quality report in browser"
    >
      <FileText size={13} />
      Quality Report
    </button>
  );
};
