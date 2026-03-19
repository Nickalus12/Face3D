import React, { useState, useEffect } from "react";
import { Folder, Trash2, ArrowUpDown, Filter } from "lucide-react";
import useSessionStore from "../store/sessionStore";
import { openFolder } from "../lib/tauri";
import { invoke } from "@tauri-apps/api/core";

// ── Types ─────────────────────────────────────────────────────

interface DetailPanelProps {
  isExpanded?: boolean;
}

type SortMode = "newest" | "oldest" | "name" | "status";

function formatRelativeTime(dateStr?: string): string {
  if (!dateStr) return "";
  try {
    const d = new Date(dateStr);
    const now = Date.now();
    const ms = now - d.getTime();
    const min = Math.floor(ms / 60000);
    if (min < 1) return "just now";
    if (min < 60) return `${min}m`;
    const hr = Math.floor(min / 60);
    if (hr < 24) return `${hr}h`;
    const day = Math.floor(hr / 24);
    if (day < 30) return `${day}d`;
    return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
  } catch { return ""; }
}

function formatDateTime(dateStr?: string): string {
  if (!dateStr) return "";
  try {
    const d = new Date(dateStr);
    const now = new Date();
    const isToday = d.toDateString() === now.toDateString();
    const time = d.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
    if (isToday) return `Today ${time}`;
    const yesterday = new Date(now);
    yesterday.setDate(yesterday.getDate() - 1);
    if (d.toDateString() === yesterday.toDateString()) return `Yesterday ${time}`;
    return d.toLocaleDateString(undefined, { month: "short", day: "numeric" }) + ` ${time}`;
  } catch { return ""; }
}

function formatSize(bytes?: number): string {
  if (!bytes || bytes === 0) return "";
  if (bytes < 1048576) return `${(bytes / 1024).toFixed(0)}K`;
  if (bytes < 1073741824) return `${(bytes / 1048576).toFixed(0)}M`;
  return `${(bytes / 1073741824).toFixed(1)}G`;
}

// ── Component ─────────────────────────────────────────────────

export const DetailPanel: React.FC<DetailPanelProps> = ({ isExpanded = true }) => {
  const [searchQuery, setSearchQuery] = useState("");
  const [sortMode, setSortMode] = useState<SortMode>("newest");
  const [filterComplete, setFilterComplete] = useState<"all" | "complete" | "pending">("all");
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number; sessionId: string } | null>(null);

  const { sessions, currentSession, selectSession, fetchSessions } = useSessionStore();

  useEffect(() => {
    if (!contextMenu) return;
    const close = () => setContextMenu(null);
    const esc = (e: KeyboardEvent) => { if (e.key === "Escape") setContextMenu(null); };
    window.addEventListener("click", close);
    window.addEventListener("keydown", esc);
    return () => { window.removeEventListener("click", close); window.removeEventListener("keydown", esc); };
  }, [contextMenu]);

  const handleDelete = async (sessionId: string) => {
    setContextMenu(null);
    try {
      await invoke("delete_session", { sessionId });
      await fetchSessions();
    } catch (e) {
      console.error("Delete failed:", e);
    }
  };

  const isComplete = (s: { has_gaussians: boolean; has_mesh: boolean; has_renders: boolean }) =>
    s.has_gaussians || s.has_mesh || s.has_renders;

  // Filter
  let filtered = sessions;
  if (searchQuery.trim()) {
    const q = searchQuery.toLowerCase();
    filtered = filtered.filter(s => s.name.toLowerCase().includes(q) || s.id.toLowerCase().includes(q));
  }
  if (filterComplete === "complete") filtered = filtered.filter(isComplete);
  if (filterComplete === "pending") filtered = filtered.filter(s => !isComplete(s));

  // Sort
  const sorted = [...filtered].sort((a, b) => {
    if (sortMode === "newest") return (b.created_at ?? "").localeCompare(a.created_at ?? "");
    if (sortMode === "oldest") return (a.created_at ?? "").localeCompare(b.created_at ?? "");
    if (sortMode === "name") return a.name.localeCompare(b.name);
    if (sortMode === "status") {
      const aComplete = isComplete(a) ? 1 : 0;
      const bComplete = isComplete(b) ? 1 : 0;
      return bComplete - aComplete;
    }
    return 0;
  });

  if (!isExpanded) return null;

  const completeCount = sessions.filter(isComplete).length;
  const pendingCount = sessions.length - completeCount;

  return (
    <div className="flex flex-col h-full w-full">
      {/* Header */}
      <div className="shrink-0 px-3 pt-3 pb-1">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">
            Sessions
            <span className="text-zinc-600 font-normal ml-1">{sessions.length}</span>
          </span>
          <div className="flex items-center gap-1">
            {/* Filter toggle */}
            <button
              onClick={() => setFilterComplete(f => f === "all" ? "complete" : f === "complete" ? "pending" : "all")}
              className={`p-1 rounded text-xs transition-colors ${
                filterComplete !== "all" ? "text-indigo-400 bg-indigo-500/10" : "text-zinc-600 hover:text-zinc-400"
              }`}
              title={`Filter: ${filterComplete}`}
            >
              <Filter size={12} />
            </button>
            {/* Sort toggle */}
            <button
              onClick={() => setSortMode(m => m === "newest" ? "name" : m === "name" ? "status" : "newest")}
              className="p-1 rounded text-zinc-600 hover:text-zinc-400 transition-colors"
              title={`Sort: ${sortMode}`}
            >
              <ArrowUpDown size={12} />
            </button>
          </div>
        </div>

        {/* Filter chips */}
        {filterComplete !== "all" && (
          <div className="flex items-center gap-1 mb-2">
            <span className={`text-[10px] px-1.5 py-0.5 rounded-full ${
              filterComplete === "complete" ? "bg-emerald-500/10 text-emerald-400" : "bg-zinc-700/50 text-zinc-400"
            }`}>
              {filterComplete === "complete" ? `${completeCount} complete` : `${pendingCount} pending`}
            </span>
            <button
              onClick={() => setFilterComplete("all")}
              className="text-[10px] text-zinc-600 hover:text-zinc-400"
            >
              clear
            </button>
          </div>
        )}

        {/* Search */}
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Search..."
          aria-label="Search sessions"
          className="w-full bg-zinc-800/50 rounded-lg px-3 py-1.5 text-sm text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus-visible:ring-1 focus-visible:ring-indigo-500/40 transition-colors"
        />
      </div>

      {/* Session list */}
      <div className="flex-1 overflow-y-auto mt-1">
        {sorted.length === 0 ? (
          <div className="px-3 py-8 text-center">
            <p className="text-xs text-zinc-600">
              {searchQuery ? "No matches" : "No sessions"}
            </p>
          </div>
        ) : (
          sorted.map((session) => {
            const isSelected = currentSession?.id === session.id;
            const complete = isComplete(session);
            const time = formatRelativeTime(session.created_at);
            const size = formatSize(session.total_size_bytes);

            return (
              <button
                key={session.id}
                onClick={() => selectSession(session)}
                onContextMenu={(e) => { e.preventDefault(); setContextMenu({ x: e.clientX, y: e.clientY, sessionId: session.id }); }}
                className={`w-full text-left px-3 py-2.5 transition-colors duration-100 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-inset focus-visible:ring-indigo-500/40 ${
                  isSelected
                    ? "bg-indigo-500/10 border-l-2 border-l-indigo-500"
                    : "hover:bg-zinc-800/40 border-l-2 border-l-transparent"
                }`}
              >
                <div className="flex items-center gap-2 mb-0.5">
                  <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${complete ? "bg-emerald-500" : "bg-zinc-600"}`} />
                  <span className={`text-sm font-medium truncate flex-1 ${isSelected ? "text-white" : "text-zinc-300"}`}>
                    {session.name}
                  </span>
                  {session.quality_grade && (
                    <span className={`text-[10px] font-bold shrink-0 ${
                      session.quality_grade === "A" ? "text-emerald-400" :
                      session.quality_grade === "B" ? "text-blue-400" :
                      session.quality_grade === "C" ? "text-amber-400" : "text-zinc-500"
                    }`}>
                      {session.quality_grade}
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-2 pl-3.5 text-[11px] text-zinc-600">
                  {complete ? (
                    <span className="text-emerald-500/70">Complete</span>
                  ) : (
                    <span>Pending</span>
                  )}
                  {size && <span>· {size}</span>}
                  {session.created_at && (
                    <span className="ml-auto" title={session.created_at}>
                      {formatDateTime(session.created_at)}
                    </span>
                  )}
                </div>
              </button>
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
            onClick={() => { openFolder(`data/output/${contextMenu.sessionId}`); setContextMenu(null); }}
            className="w-full flex items-center gap-2 px-3 py-2 text-xs text-zinc-300 hover:bg-white/5 transition-all active:scale-[0.98]"
          >
            <Folder size={14} /> Open Folder
          </button>
          <div className="border-t border-zinc-800 my-1" />
          <button
            onClick={() => handleDelete(contextMenu.sessionId)}
            className="w-full flex items-center gap-2 px-3 py-2 text-xs text-red-400 hover:bg-red-500/10 transition-all active:scale-[0.98]"
          >
            <Trash2 size={14} /> Delete
          </button>
        </div>
      )}
    </div>
  );
};
