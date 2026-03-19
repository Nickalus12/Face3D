import React, { useState, useEffect } from "react";
import {
  CheckCircle2,
  CircleDashed,
  HardDrive,
  Folder,
  Trash2,
  Download,
  Box,
  Image as ImageIcon,
  PlayCircle,
} from "lucide-react";
import useSessionStore from "../store/sessionStore";
import usePipelineStore, { PIPELINE_STAGES } from "../store/pipelineStore";
import { openFolder, listRenders } from "../lib/tauri";
import { convertFileSrc } from "@tauri-apps/api/core";

// ── Types ─────────────────────────────────────────────────────

interface DetailPanelProps {
  isExpanded?: boolean;
  onToggle?: () => void;
}

// ── Helpers ───────────────────────────────────────────────────

function formatBytes(bytes: number): string {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
}

function isTauri(): boolean {
  return typeof window !== "undefined" && !!(window as any).__TAURI_INTERNALS__;
}

function formatRelativeTime(dateStr: string): string {
  try {
    const date = new Date(dateStr);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMin = Math.floor(diffMs / 60000);
    if (diffMin < 1) return "just now";
    if (diffMin < 60) return `${diffMin}m ago`;
    const diffHr = Math.floor(diffMin / 60);
    if (diffHr < 24) return `${diffHr}h ago`;
    const diffDay = Math.floor(diffHr / 24);
    if (diffDay < 7) return `${diffDay}d ago`;
    return date.toLocaleDateString();
  } catch {
    return "";
  }
}

// ── Session Thumbnail ─────────────────────────────────────────

const SessionThumbnail: React.FC<{ sessionId: string; hasRenders: boolean }> = ({
  sessionId,
  hasRenders,
}) => {
  const [thumbSrc, setThumbSrc] = useState<string | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    if (!hasRenders || !isTauri()) return;
    let cancelled = false;

    listRenders(sessionId).then((renders) => {
      if (cancelled || renders.length === 0) return;
      try {
        const src = convertFileSrc(renders[0]);
        setThumbSrc(src);
      } catch {
        setError(true);
      }
    });

    return () => { cancelled = true; };
  }, [sessionId, hasRenders]);

  if (thumbSrc && !error) {
    return (
      <img
        src={thumbSrc}
        alt=""
        className="w-full h-full object-cover"
        onError={() => setError(true)}
      />
    );
  }

  return hasRenders ? (
    <ImageIcon size={20} className="text-zinc-600" />
  ) : (
    <Box size={20} className="text-zinc-700" />
  );
};

// ── Component ─────────────────────────────────────────────────

export const DetailPanel: React.FC<DetailPanelProps> = ({
  isExpanded = true,
  onToggle: _onToggle,
}) => {
  const [searchQuery, setSearchQuery] = useState("");
  const [contextMenu, setContextMenu] = useState<{
    x: number;
    y: number;
    sessionId: string;
  } | null>(null);

  const {
    sessions,
    currentSession,
    sessionFiles,
    selectSession,
  } = useSessionStore();
  const { stages, status } = usePipelineStore();

  const isRunning = status === "running";

  // ── Context menu close on outside click ──────────────────

  useEffect(() => {
    if (!contextMenu) return;
    const handleClick = () => setContextMenu(null);
    window.addEventListener("click", handleClick);
    return () => window.removeEventListener("click", handleClick);
  }, [contextMenu]);

  // ── Session helpers ──────────────────────────────────────

  const handleContextMenu = (
    e: React.MouseEvent,
    sessionId: string
  ) => {
    e.preventDefault();
    setContextMenu({ x: e.clientX, y: e.clientY, sessionId });
  };

  const handleOpenFolder = (sessionId: string) => {
    openFolder(`data/output/${sessionId}`);
    setContextMenu(null);
  };

  // ── Session total file size ──────────────────────────────

  const getSessionSize = (sessionId: string): number | null => {
    if (currentSession?.id === sessionId && sessionFiles.length > 0) {
      return sessionFiles.reduce((sum, f) => sum + f.size, 0);
    }
    return null;
  };

  // ── Pipeline progress for running session ─────────────────

  const pipelineProgress = (() => {
    if (!isRunning) return null;
    const completed = stages.filter((s) => s.status === "complete").length;
    const total = PIPELINE_STAGES.length;
    return { completed, total, percent: Math.round((completed / total) * 100) };
  })();

  // ── Filtered sessions ────────────────────────────────────

  const filteredSessions = searchQuery.trim()
    ? sessions.filter(
        (s) =>
          s.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
          s.id.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : sessions;

  if (!isExpanded) return null;

  return (
    <div className="flex flex-col h-full w-full">
      {/* Search - fixed at top */}
      <div className="shrink-0 p-3 pb-2">
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Search sessions..."
          className="w-full bg-zinc-800/60 border border-zinc-700/50 rounded-lg px-3 py-2.5 text-sm text-zinc-200 placeholder:text-zinc-500 focus:outline-none focus:border-indigo-500/50 focus:ring-1 focus:ring-indigo-500/20 transition-colors"
        />
      </div>

      {/* Session list - scrollable */}
      <div className="flex-1 overflow-y-auto px-3 pb-3 space-y-2 scrollbar-hide">
        {filteredSessions.length === 0 ? (
          <div className="text-center py-12">
            <div className="w-14 h-14 rounded-card bg-zinc-800/40 border border-border flex items-center justify-center mx-auto mb-4">
              <Box size={24} className="text-zinc-600" />
            </div>
            <p className="text-sm text-zinc-400 font-medium">
              {searchQuery ? "No matching sessions" : "No sessions yet"}
            </p>
            <p className="text-xs text-zinc-500 mt-1.5 leading-relaxed">
              {searchQuery
                ? "Try a different search term"
                : "Start a new scan to create your first session"}
            </p>
            {!searchQuery && (
              <button className="mt-4 px-4 py-2 rounded-button text-xs font-medium text-indigo-400 bg-indigo-500/10 border border-indigo-500/20 hover:bg-indigo-500/20 hover:border-indigo-500/30 transition-all duration-150">
                Start a new scan
              </button>
            )}
          </div>
        ) : (
          filteredSessions.map((session) => {
            const isSelected = currentSession?.id === session.id;
            const sessionSize = getSessionSize(session.id);
            const showPipeline = isRunning && isSelected;

            return (
              <div
                key={session.id}
                onClick={() => selectSession(session)}
                onContextMenu={(e) => handleContextMenu(e, session.id)}
                className={`group relative rounded-card cursor-pointer transition-all duration-200 overflow-hidden ${
                  isSelected
                    ? "bg-indigo-500/5 border-l-2 border-l-indigo-500 border-y border-r border-y-border border-r-border"
                    : "border border-border hover:bg-zinc-800/40"
                }`}
              >
                <div className="p-3">
                  {/* Thumbnail */}
                  <div className={`w-full h-16 rounded-lg border mb-2.5 flex items-center justify-center overflow-hidden transition-all duration-200 ${
                    isSelected
                      ? "bg-zinc-800/40 border-indigo-500/15"
                      : "bg-zinc-800/50 border-zinc-700/30 group-hover:border-zinc-600/50"
                  }`}>
                    <SessionThumbnail
                      sessionId={session.id}
                      hasRenders={session.has_renders}
                    />
                  </div>

                  {/* Session name */}
                  <p className={`text-sm font-medium truncate mb-1.5 ${
                    isSelected ? "text-zinc-200" : "text-zinc-200 group-hover:text-white"
                  }`}>
                    {session.name}
                  </p>

                  {/* Quality grade + size */}
                  {(session.quality_grade || session.total_size_bytes) && (
                    <div className="flex items-center gap-2 mb-1.5 text-xs text-zinc-500">
                      {session.quality_grade && (
                        <span className={`font-bold ${
                          session.quality_grade === 'A' ? 'text-emerald-400' :
                          session.quality_grade === 'B' ? 'text-blue-400' :
                          session.quality_grade === 'C' ? 'text-amber-400' :
                          'text-zinc-400'
                        }`}>
                          {session.quality_grade}
                        </span>
                      )}
                      {session.total_size_bytes && session.total_size_bytes > 0 && (
                        <span>{formatBytes(session.total_size_bytes)}</span>
                      )}
                    </div>
                  )}

                  {/* Status badges */}
                  <div className="flex gap-1.5 mb-2 flex-wrap">
                    {session.has_gaussians && (
                      <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[11px] font-medium text-emerald-400">
                        <CheckCircle2 size={10} className="text-emerald-500" /> Splat
                      </span>
                    )}
                    {session.has_mesh && (
                      <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[11px] font-medium text-blue-400">
                        <CheckCircle2 size={10} className="text-blue-500" /> Mesh
                      </span>
                    )}
                    {session.has_renders && (
                      <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[11px] font-medium text-purple-400">
                        <CheckCircle2 size={10} className="text-purple-500" /> Renders
                      </span>
                    )}
                    {!session.has_gaussians &&
                      !session.has_mesh &&
                      !session.has_renders && (
                        <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[11px] font-medium text-zinc-500">
                          <CircleDashed size={10} /> Pending
                        </span>
                      )}
                  </div>

                  {/* Size + date row */}
                  <div className="flex items-center justify-between text-xs text-zinc-600">
                    {sessionSize != null ? (
                      <span className="flex items-center gap-1">
                        <HardDrive size={10} />
                        {formatBytes(sessionSize)}
                      </span>
                    ) : session.total_size_bytes && session.total_size_bytes > 0 ? (
                      <span className="flex items-center gap-1">
                        <HardDrive size={10} />
                        {formatBytes(session.total_size_bytes)}
                      </span>
                    ) : <span />}
                    {session.created_at && (
                      <span title={session.created_at}>
                        {formatRelativeTime(session.created_at)}
                      </span>
                    )}
                  </div>

                  {/* Pipeline progress bar (inline on active session) */}
                  {showPipeline && pipelineProgress && (
                    <div className="mt-2.5 pt-2 border-t border-zinc-700/30">
                      <div className="flex items-center justify-between mb-1.5">
                        <div className="flex items-center gap-1.5">
                          <PlayCircle size={11} className="text-indigo-400" />
                          <span className="text-[11px] font-medium text-indigo-300">
                            Pipeline running
                          </span>
                        </div>
                        <span className="text-[11px] text-zinc-500">
                          {pipelineProgress.completed}/{pipelineProgress.total}
                        </span>
                      </div>
                      <div className="w-full h-1 bg-zinc-800 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-indigo-600 to-indigo-400 rounded-full transition-all duration-500"
                          style={{ width: `${pipelineProgress.percent}%` }}
                        />
                      </div>
                    </div>
                  )}
                </div>
              </div>
            );
          })
        )}
      </div>

      {/* Context Menu */}
      {contextMenu && (
        <div
          className="fixed z-[80] bg-surface-overlay border border-border-strong rounded-button shadow-2xl py-1 min-w-[160px] animate-[scaleIn_100ms_ease-out]"
          style={{ top: contextMenu.y, left: contextMenu.x }}
        >
          <button
            onClick={() => handleOpenFolder(contextMenu.sessionId)}
            className="w-full flex items-center gap-2 px-3 py-2 text-xs text-zinc-300 hover:bg-white/5 transition-colors"
          >
            <Folder size={14} /> Open Folder
          </button>
          <button className="w-full flex items-center gap-2 px-3 py-2 text-xs text-zinc-300 hover:bg-white/5 transition-colors">
            <Download size={14} /> Export
          </button>
          <div className="border-t border-zinc-800 my-1" />
          <button className="w-full flex items-center gap-2 px-3 py-2 text-xs text-red-400 hover:bg-red-500/10 transition-colors">
            <Trash2 size={14} /> Delete
          </button>
        </div>
      )}

      <style>{`
        @keyframes scaleIn {
          from { opacity: 0; transform: scale(0.95); }
          to { opacity: 1; transform: scale(1); }
        }
      `}</style>
    </div>
  );
};
