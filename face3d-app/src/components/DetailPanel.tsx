import React, { useState, useRef, useEffect, useCallback } from "react";
import {
  CheckCircle2,
  CircleDashed,
  AlertCircle,
  PlayCircle,
  HardDrive,
  ChevronRight,
  Folder,
  Trash2,
  Download,
  FileText,
  Box,
  Image as ImageIcon,
  ExternalLink,
  Activity,
  Layers,
  Timer,
} from "lucide-react";
import useSessionStore from "../store/sessionStore";
import usePipelineStore, { PIPELINE_STAGES } from "../store/pipelineStore";
import { openFolder } from "../lib/tauri";

// ── Types ─────────────────────────────────────────────────────

type TabId = "sessions" | "pipeline" | "info";

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

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}m ${s.toString().padStart(2, "0")}s`;
}

// ── Tab Indicator ─────────────────────────────────────────────

const TABS: { id: TabId; label: string }[] = [
  { id: "sessions", label: "Sessions" },
  { id: "pipeline", label: "Pipeline" },
  { id: "info", label: "Info" },
];

// ── Component ─────────────────────────────────────────────────

export const DetailPanel: React.FC<DetailPanelProps> = ({
  isExpanded = true,
  onToggle: _onToggle,
}) => {
  const [activeTab, setActiveTab] = useState<TabId>("sessions");
  const [expandedStage, setExpandedStage] = useState<number | null>(null);
  const [contextMenu, setContextMenu] = useState<{
    x: number;
    y: number;
    sessionId: string;
  } | null>(null);
  const tabBarRef = useRef<HTMLDivElement>(null);
  const [indicatorStyle, setIndicatorStyle] = useState({ left: 0, width: 0 });

  const {
    sessions,
    currentSession,
    sessionFiles,
    selectSession,
  } = useSessionStore();
  const { stages, status, startedAt, metrics } = usePipelineStore();

  const isRunning = status === "running";

  // ── Tab indicator animation ──────────────────────────────

  const updateIndicator = useCallback(() => {
    if (!tabBarRef.current) return;
    const activeIdx = TABS.findIndex((t) => t.id === activeTab);
    const buttons = tabBarRef.current.querySelectorAll<HTMLButtonElement>(
      "[data-tab-button]"
    );
    if (buttons[activeIdx]) {
      const btn = buttons[activeIdx];
      setIndicatorStyle({
        left: btn.offsetLeft,
        width: btn.offsetWidth,
      });
    }
  }, [activeTab]);

  useEffect(() => {
    updateIndicator();
  }, [updateIndicator]);

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

  const totalSize = sessionFiles.reduce((sum, f) => sum + f.size, 0);

  // ── Latest metrics ───────────────────────────────────────

  const latestMetric = metrics.length > 0 ? metrics[metrics.length - 1] : null;

  // ── Elapsed timer ────────────────────────────────────────

  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    if (!startedAt || !isRunning) {
      setElapsed(0);
      return;
    }
    const tick = setInterval(
      () => setElapsed(Math.floor((Date.now() - startedAt) / 1000)),
      1000
    );
    return () => clearInterval(tick);
  }, [startedAt, isRunning]);

  if (!isExpanded) return null;

  return (
    <div className="flex flex-col h-full w-full">
      {/* Tab Bar */}
      <div className="relative shrink-0 border-b border-zinc-800/40">
        <div ref={tabBarRef} className="flex px-3 pt-2 relative">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              data-tab-button
              onClick={() => setActiveTab(tab.id)}
              className={`px-3 pb-2.5 pt-1 text-xs font-semibold tracking-wide transition-colors relative ${
                activeTab === tab.id
                  ? "text-zinc-100"
                  : "text-zinc-500 hover:text-zinc-300"
              }`}
            >
              {tab.label}
            </button>
          ))}
          {/* Animated underline */}
          <div
            className="absolute bottom-0 h-[2px] bg-indigo-500 rounded-full transition-all duration-300 ease-out"
            style={{
              left: indicatorStyle.left,
              width: indicatorStyle.width,
            }}
          />
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3 scrollbar-hide">
        {/* ═══════════════ SESSIONS TAB ═══════════════ */}
        {activeTab === "sessions" && (
          <div className="space-y-2">
            {sessions.length === 0 ? (
              <div className="text-center py-8">
                <Box size={32} className="text-zinc-700 mx-auto mb-3" />
                <p className="text-sm text-zinc-500">No sessions yet</p>
                <p className="text-xs text-zinc-600 mt-1">
                  Start a new scan to create one
                </p>
              </div>
            ) : (
              sessions.map((session) => {
                const isSelected = currentSession?.id === session.id;
                return (
                  <div
                    key={session.id}
                    onClick={() => selectSession(session)}
                    onContextMenu={(e) => handleContextMenu(e, session.id)}
                    className={`group p-3 rounded-xl border cursor-pointer transition-all duration-200 ${
                      isSelected
                        ? "bg-indigo-500/5 border-indigo-500/30 ring-1 ring-indigo-500/10"
                        : "bg-zinc-900/30 border-zinc-800 hover:border-zinc-600 hover:bg-zinc-900/60"
                    }`}
                  >
                    {/* Header row */}
                    <div className="flex justify-between items-start mb-2">
                      <span
                        className={`text-sm font-medium transition-colors ${
                          isSelected
                            ? "text-indigo-300"
                            : "text-zinc-200 group-hover:text-white"
                        }`}
                      >
                        {session.name}
                      </span>
                      <span className="text-[10px] text-zinc-600">
                        {session.id}
                      </span>
                    </div>

                    {/* Status badges */}
                    <div className="flex gap-1.5 mb-2.5 flex-wrap">
                      {session.has_gaussians && (
                        <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                          <CheckCircle2 size={10} /> Gaussians
                        </span>
                      )}
                      {session.has_mesh && (
                        <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                          <CheckCircle2 size={10} /> Mesh
                        </span>
                      )}
                      {session.has_renders && (
                        <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                          <CheckCircle2 size={10} /> Renders
                        </span>
                      )}
                      {!session.has_gaussians &&
                        !session.has_mesh &&
                        !session.has_renders && (
                          <span className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-zinc-800 text-zinc-500 border border-zinc-700/50">
                            Pending
                          </span>
                        )}
                    </div>

                    {/* Thumbnail placeholder */}
                    <div className="w-full h-20 rounded-lg bg-zinc-800/50 border border-zinc-700/30 mb-2 flex items-center justify-center overflow-hidden">
                      {session.has_renders ? (
                        <ImageIcon
                          size={20}
                          className="text-zinc-600"
                        />
                      ) : (
                        <Box size={20} className="text-zinc-700" />
                      )}
                    </div>

                    {/* File stats */}
                    {isSelected && sessionFiles.length > 0 && (
                      <div className="flex items-center gap-3 text-xs text-zinc-500">
                        <span className="flex items-center gap-1">
                          <HardDrive size={11} />
                          {formatBytes(totalSize)}
                        </span>
                        <span className="flex items-center gap-1">
                          <FileText size={11} />
                          {sessionFiles.length} files
                        </span>
                      </div>
                    )}
                  </div>
                );
              })
            )}
          </div>
        )}

        {/* ═══════════════ PIPELINE TAB ═══════════════ */}
        {activeTab === "pipeline" && (
          <div className="space-y-1">
            {/* Overall progress */}
            {isRunning && (
              <div className="mb-4 p-3 rounded-lg bg-indigo-500/5 border border-indigo-500/20">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-semibold text-indigo-300">
                    Running
                  </span>
                  <span className="text-xs text-indigo-400 font-mono">
                    {formatDuration(elapsed)}
                  </span>
                </div>
                <div className="w-full h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-indigo-600 to-indigo-400 rounded-full transition-all duration-500"
                    style={{
                      width: `${
                        (stages.filter((s) => s.status === "complete").length /
                          PIPELINE_STAGES.length) *
                        100
                      }%`,
                    }}
                  />
                </div>
                <p className="text-[10px] text-zinc-500 mt-1.5">
                  {stages.filter((s) => s.status === "complete").length} of{" "}
                  {PIPELINE_STAGES.length} stages complete
                </p>
              </div>
            )}

            {/* Stage stepper */}
            <div className="relative pl-2">
              <div className="absolute left-[13px] top-2 bottom-6 w-[1px] bg-zinc-800/60" />

              <div className="space-y-1.5">
                {stages.map((stage) => {
                  const isDone = stage.status === "complete";
                  const isCurrent = stage.status === "running";
                  const isError = stage.status === "error";
                  const isSkipped = stage.status === "skipped";
                  const isExpanded = expandedStage === stage.id;

                  // Per-stage progress estimate (mock for running stage)
                  const stageProgress = isDone
                    ? 100
                    : isCurrent
                    ? 50
                    : 0;

                  return (
                    <div key={stage.id} className="relative z-10">
                      <div
                        className="flex items-start gap-3 cursor-pointer group py-1.5"
                        onClick={() =>
                          setExpandedStage(isExpanded ? null : stage.id)
                        }
                      >
                        {/* Icon */}
                        <div
                          className={`mt-0.5 bg-[#0e0e11] rounded-full p-0.5 shrink-0 ${
                            isDone
                              ? "text-emerald-500"
                              : isCurrent
                              ? "text-indigo-400 animate-pulse"
                              : isError
                              ? "text-red-500"
                              : isSkipped
                              ? "text-zinc-600"
                              : "text-zinc-700"
                          }`}
                        >
                          {isDone && <CheckCircle2 size={16} />}
                          {isCurrent && <PlayCircle size={16} />}
                          {isError && <AlertCircle size={16} />}
                          {(stage.status === "pending" || isSkipped) && (
                            <CircleDashed size={16} />
                          )}
                        </div>

                        {/* Label + time */}
                        <div className="flex-1 min-w-0">
                          <div className="flex justify-between items-center">
                            <span
                              className={`text-sm truncate ${
                                isCurrent
                                  ? "text-zinc-100 font-medium"
                                  : isDone
                                  ? "text-zinc-300"
                                  : isError
                                  ? "text-red-300"
                                  : "text-zinc-500"
                              } group-hover:text-zinc-200 transition-colors`}
                            >
                              {stage.name}
                            </span>
                            <div className="flex items-center gap-2 shrink-0 ml-2">
                              {isDone && stage.elapsed != null && (
                                <span className="text-[10px] text-zinc-500 font-mono">
                                  {formatDuration(stage.elapsed)}
                                </span>
                              )}
                              {isCurrent && (
                                <span className="text-[10px] text-indigo-400 font-mono animate-pulse">
                                  {stageProgress}%
                                </span>
                              )}
                              <ChevronRight
                                size={12}
                                className={`text-zinc-600 transition-transform duration-200 ${
                                  isExpanded ? "rotate-90" : ""
                                }`}
                              />
                            </div>
                          </div>

                          {/* Mini progress bar for current stage */}
                          {isCurrent && (
                            <div className="w-full h-0.5 bg-zinc-800 rounded-full mt-1 overflow-hidden">
                              <div
                                className="h-full bg-indigo-500 rounded-full transition-all duration-500"
                                style={{ width: `${stageProgress}%` }}
                              />
                            </div>
                          )}
                        </div>
                      </div>

                      {/* Expandable details */}
                      <div
                        className={`overflow-hidden transition-all duration-200 ease-out ${
                          isExpanded
                            ? "max-h-40 opacity-100"
                            : "max-h-0 opacity-0"
                        }`}
                      >
                        <div className="ml-8 mt-1 mb-2 p-3 rounded-lg bg-zinc-900/40 border border-zinc-800/50 text-xs text-zinc-400 space-y-1.5">
                          <div className="flex justify-between">
                            <span>Status</span>
                            <span
                              className={`font-medium ${
                                isDone
                                  ? "text-emerald-400"
                                  : isCurrent
                                  ? "text-indigo-400"
                                  : isError
                                  ? "text-red-400"
                                  : "text-zinc-500"
                              }`}
                            >
                              {stage.status.charAt(0).toUpperCase() +
                                stage.status.slice(1)}
                            </span>
                          </div>
                          {stage.elapsed != null && (
                            <div className="flex justify-between">
                              <span>Duration</span>
                              <span className="text-zinc-200">
                                {formatDuration(stage.elapsed)}
                              </span>
                            </div>
                          )}
                          <div className="flex justify-between">
                            <span>Stage</span>
                            <span className="text-zinc-200">
                              {stage.id} / {PIPELINE_STAGES.length}
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}

        {/* ═══════════════ INFO TAB ═══════════════ */}
        {activeTab === "info" && (
          <div className="space-y-4">
            {!currentSession ? (
              <div className="text-center py-8">
                <Activity size={32} className="text-zinc-700 mx-auto mb-3" />
                <p className="text-sm text-zinc-500">No session selected</p>
              </div>
            ) : (
              <>
                {/* Session header */}
                <div className="p-3 rounded-xl bg-zinc-900/30 border border-zinc-800">
                  <h3 className="text-sm font-semibold text-zinc-200">
                    {currentSession.name}
                  </h3>
                  <p className="text-xs text-zinc-500 mt-0.5 font-mono">
                    {currentSession.id}
                  </p>
                </div>

                {/* Output files */}
                <div>
                  <div className="text-[10px] font-bold tracking-wider text-zinc-500 uppercase mb-2">
                    Output Files
                  </div>
                  {sessionFiles.length === 0 ? (
                    <p className="text-xs text-zinc-600 italic">
                      No output files yet
                    </p>
                  ) : (
                    <div className="space-y-1">
                      {sessionFiles.map((file) => (
                        <div
                          key={file.name}
                          className="flex items-center justify-between py-1.5 px-2 rounded-lg hover:bg-white/[0.02] transition-colors group"
                        >
                          <div className="flex items-center gap-2 min-w-0">
                            <FileText
                              size={12}
                              className="text-zinc-600 shrink-0"
                            />
                            <span className="text-xs text-zinc-300 truncate">
                              {file.name}
                            </span>
                          </div>
                          <div className="flex items-center gap-2 shrink-0">
                            <span className="text-[10px] text-zinc-600 font-mono">
                              {formatBytes(file.size)}
                            </span>
                            <ExternalLink
                              size={10}
                              className="text-zinc-700 group-hover:text-zinc-400 transition-colors cursor-pointer"
                            />
                          </div>
                        </div>
                      ))}
                      <div className="flex items-center justify-between pt-2 border-t border-zinc-800/40 mt-2">
                        <span className="text-[10px] text-zinc-500 uppercase tracking-wider font-semibold">
                          Total
                        </span>
                        <span className="text-xs text-zinc-300 font-mono">
                          {formatBytes(totalSize)}
                        </span>
                      </div>
                    </div>
                  )}
                </div>

                {/* Quality Metrics */}
                <div>
                  <div className="text-[10px] font-bold tracking-wider text-zinc-500 uppercase mb-2">
                    Quality Metrics
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <div className="p-3 rounded-lg bg-zinc-900/40 border border-zinc-800/40">
                      <Activity
                        size={14}
                        className="text-emerald-400 mb-1"
                      />
                      <p className="text-lg font-medium text-zinc-200">
                        {latestMetric?.psnr
                          ? `${latestMetric.psnr.toFixed(1)} dB`
                          : "--"}
                      </p>
                      <p className="text-[10px] text-zinc-500">PSNR</p>
                    </div>
                    <div className="p-3 rounded-lg bg-zinc-900/40 border border-zinc-800/40">
                      <Layers size={14} className="text-blue-400 mb-1" />
                      <p className="text-lg font-medium text-zinc-200">
                        {latestMetric?.gaussians
                          ? latestMetric.gaussians >= 1_000_000
                            ? `${(latestMetric.gaussians / 1_000_000).toFixed(1)}M`
                            : latestMetric.gaussians >= 1_000
                            ? `${Math.round(latestMetric.gaussians / 1_000)}K`
                            : latestMetric.gaussians.toString()
                          : "--"}
                      </p>
                      <p className="text-[10px] text-zinc-500">Gaussians</p>
                    </div>
                  </div>
                </div>

                {/* Training Summary */}
                <div>
                  <div className="text-[10px] font-bold tracking-wider text-zinc-500 uppercase mb-2">
                    Training Summary
                  </div>
                  <div className="space-y-2 p-3 rounded-lg bg-zinc-900/40 border border-zinc-800/40 text-xs">
                    <div className="flex justify-between">
                      <span className="text-zinc-500 flex items-center gap-1.5">
                        <Timer size={12} /> Duration
                      </span>
                      <span className="text-zinc-200">
                        {startedAt && !isRunning
                          ? formatDuration(
                              Math.floor(
                                (Date.now() - startedAt) / 1000
                              )
                            )
                          : isRunning
                          ? formatDuration(elapsed)
                          : "--"}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-zinc-500">Iterations</span>
                      <span className="text-zinc-200">
                        {latestMetric?.iter?.toLocaleString() ?? "--"}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-zinc-500">Final Loss</span>
                      <span className="text-zinc-200">
                        {latestMetric?.loss?.toFixed(4) ?? "--"}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Quick actions */}
                <div className="flex gap-2">
                  <button
                    onClick={() =>
                      openFolder(`data/output/${currentSession.id}`)
                    }
                    className="flex-1 flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg text-xs text-zinc-400 bg-zinc-900/40 border border-zinc-800/40 hover:border-zinc-600 hover:text-zinc-200 transition-colors"
                  >
                    <Folder size={12} /> Open Folder
                  </button>
                  <button className="flex-1 flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg text-xs text-zinc-400 bg-zinc-900/40 border border-zinc-800/40 hover:border-zinc-600 hover:text-zinc-200 transition-colors">
                    <Download size={12} /> Export
                  </button>
                </div>
              </>
            )}
          </div>
        )}
      </div>

      {/* Context Menu */}
      {contextMenu && (
        <div
          className="fixed z-[80] bg-[#1a1a1e] border border-zinc-700/60 rounded-lg shadow-2xl py-1 min-w-[160px] animate-[scaleIn_100ms_ease-out]"
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
