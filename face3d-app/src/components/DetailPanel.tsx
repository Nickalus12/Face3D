import React, { useState, useEffect } from "react";
import { Folder, Trash2 } from "lucide-react";
import useSessionStore from "../store/sessionStore";
import { openFolder } from "../lib/tauri";

// ── Types ─────────────────────────────────────────────────────

interface DetailPanelProps {
  isExpanded?: boolean;
}

// ── Component ─────────────────────────────────────────────────

export const DetailPanel: React.FC<DetailPanelProps> = ({
  isExpanded = true,
}) => {
  const [searchQuery, setSearchQuery] = useState("");
  const [contextMenu, setContextMenu] = useState<{
    x: number;
    y: number;
    sessionId: string;
  } | null>(null);

  const { sessions, currentSession, selectSession } = useSessionStore();

  // ── Context menu close on outside click ──────────────────
  useEffect(() => {
    if (!contextMenu) return;
    const handleClick = () => setContextMenu(null);
    window.addEventListener("click", handleClick);
    return () => window.removeEventListener("click", handleClick);
  }, [contextMenu]);

  // ── Handlers ───────────────────────────────────────────────
  const handleContextMenu = (e: React.MouseEvent, sessionId: string) => {
    e.preventDefault();
    setContextMenu({ x: e.clientX, y: e.clientY, sessionId });
  };

  const handleOpenFolder = (sessionId: string) => {
    openFolder(`data/output/${sessionId}`);
    setContextMenu(null);
  };

  // ── Filtered sessions ────────────────────────────────────
  const filteredSessions = searchQuery.trim()
    ? sessions.filter((s) =>
        s.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        s.id.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : sessions;

  if (!isExpanded) return null;

  const isComplete = (s: { has_gaussians: boolean; has_mesh: boolean; has_renders: boolean }) =>
    s.has_gaussians || s.has_mesh || s.has_renders;

  return (
    <div className="flex flex-col h-full w-full">
      {/* Search */}
      <div className="shrink-0 p-2">
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Search sessions..."
          className="w-full bg-zinc-800/50 rounded-lg px-3 py-2 text-sm text-zinc-200 placeholder:text-zinc-500 focus:outline-none focus:ring-1 focus:ring-indigo-500/30 transition-colors"
        />
      </div>

      {/* Session list */}
      <div className="flex-1 overflow-y-auto scrollbar-hide">
        {filteredSessions.length === 0 ? (
          <div className="px-3 py-8 text-center">
            <p className="text-sm text-zinc-500">
              {searchQuery ? "No matching sessions" : "No sessions yet"}
            </p>
          </div>
        ) : (
          filteredSessions.map((session) => {
            const isSelected = currentSession?.id === session.id;
            const complete = isComplete(session);

            return (
              <div
                key={session.id}
                onClick={() => selectSession(session)}
                onContextMenu={(e) => handleContextMenu(e, session.id)}
                className={`flex items-center gap-2.5 px-3 py-2 cursor-pointer transition-colors duration-100 ${
                  isSelected
                    ? "bg-indigo-500/10 text-white border-l-2 border-l-indigo-500"
                    : "text-zinc-400 hover:bg-zinc-800/40 border-l-2 border-l-transparent"
                }`}
              >
                {/* Status dot */}
                <span
                  className={`shrink-0 w-2 h-2 rounded-full ${
                    complete ? "bg-emerald-500" : "bg-zinc-600"
                  }`}
                />
                {/* Session name */}
                <span className="text-sm truncate">{session.name}</span>
              </div>
            );
          })
        )}
      </div>

      {/* Context Menu */}
      {contextMenu && (
        <div
          className="fixed z-[80] bg-zinc-900 border border-zinc-700 rounded-lg shadow-2xl py-1 min-w-[140px]"
          style={{ top: contextMenu.y, left: contextMenu.x }}
        >
          <button
            onClick={() => handleOpenFolder(contextMenu.sessionId)}
            className="w-full flex items-center gap-2 px-3 py-2 text-xs text-zinc-300 hover:bg-white/5 transition-all active:scale-[0.98]"
          >
            <Folder size={14} /> Open Folder
          </button>
          <div className="border-t border-zinc-800 my-1" />
          <button className="w-full flex items-center gap-2 px-3 py-2 text-xs text-red-400 hover:bg-red-500/10 transition-all active:scale-[0.98]">
            <Trash2 size={14} /> Delete
          </button>
        </div>
      )}
    </div>
  );
};
