import {
  Play,
  Square,
  FolderSearch,
  Loader2,
} from "lucide-react";
import usePipelineStore from "../store/pipelineStore";
import useSessionStore from "../store/sessionStore";
import { selectContentDir } from "../lib/tauri";
import { clsx } from "clsx";

export default function PipelineControl() {
  const {
    status,
    currentStage,
    stages,
    contentDir,
    sessionName,
    setContentDir,
    setSessionName,
    startPipeline,
    stopPipeline,
  } = usePipelineStore();

  const { createSession } = useSessionStore();

  const isRunning = status === "running" || status === "stopping";
  const completedStages = stages.filter((s) => s.status === "complete").length;
  const progress = Math.round((completedStages / stages.length) * 100);

  async function handlePickDirectory() {
    const dir = await selectContentDir();
    if (dir) {
      setContentDir(dir);
    }
  }

  async function handleStart() {
    if (!contentDir) return;
    const name = sessionName.trim() || `session_${Date.now()}`;
    if (sessionName.trim()) {
      createSession(name, contentDir);
    }
    await startPipeline(contentDir, name);
  }

  return (
    <div className="bg-zinc-900 rounded-xl border border-zinc-800/50 p-4">
      <h2 className="text-sm font-semibold text-zinc-300 mb-3">
        Pipeline Control
      </h2>

      <div className="space-y-3">
        {/* Session Name */}
        <div>
          <label className="block text-[11px] font-medium text-zinc-500 uppercase tracking-wider mb-1">
            Session Name
          </label>
          <input
            type="text"
            value={sessionName}
            onChange={(e) => setSessionName(e.target.value)}
            placeholder="e.g. Front Profile Scan"
            disabled={isRunning}
            className="w-full px-3 py-1.5 bg-zinc-800 border border-zinc-700 rounded-md text-sm text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:border-emerald-500/50 focus:ring-1 focus:ring-emerald-500/20 disabled:opacity-50"
          />
        </div>

        {/* Content Directory */}
        <div>
          <label className="block text-[11px] font-medium text-zinc-500 uppercase tracking-wider mb-1">
            Content Directory
          </label>
          <div className="flex gap-2">
            <input
              type="text"
              value={contentDir}
              onChange={(e) => setContentDir(e.target.value)}
              placeholder="D:/Projects/3D/data/raw/..."
              disabled={isRunning}
              className="flex-1 px-3 py-1.5 bg-zinc-800 border border-zinc-700 rounded-md text-sm text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:border-emerald-500/50 focus:ring-1 focus:ring-emerald-500/20 disabled:opacity-50 truncate"
            />
            <button
              onClick={handlePickDirectory}
              disabled={isRunning}
              className="px-2.5 py-1.5 bg-zinc-800 border border-zinc-700 rounded-md hover:bg-zinc-700 hover:border-zinc-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              title="Browse"
            >
              <FolderSearch className="w-4 h-4 text-zinc-400" />
            </button>
          </div>
        </div>

        {/* Progress */}
        {isRunning && (
          <div className="space-y-1.5">
            <div className="flex justify-between text-xs">
              <span className="text-zinc-400">
                Stage {currentStage}/14
                {stages[currentStage - 1] &&
                  ` -- ${stages[currentStage - 1].name}`}
              </span>
              <span className="text-emerald-400 font-medium">{progress}%</span>
            </div>
            <div className="w-full h-1.5 bg-zinc-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-emerald-600 to-emerald-400 rounded-full transition-all duration-700"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex gap-2 pt-1">
          {!isRunning ? (
            <button
              onClick={handleStart}
              disabled={!contentDir}
              className={clsx(
                "flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors",
                contentDir
                  ? "bg-emerald-600 hover:bg-emerald-500 text-white"
                  : "bg-zinc-800 text-zinc-500 cursor-not-allowed",
              )}
            >
              <Play className="w-4 h-4" />
              Start Pipeline
            </button>
          ) : (
            <button
              onClick={stopPipeline}
              className="flex-1 flex items-center justify-center gap-2 px-4 py-2 bg-red-600 hover:bg-red-500 text-white rounded-lg text-sm font-medium transition-colors"
            >
              <Square className="w-3.5 h-3.5" />
              Stop Pipeline
            </button>
          )}
        </div>

        {/* Running indicator */}
        {isRunning && (
          <div className="flex items-center gap-2 text-xs text-amber-400">
            <Loader2 className="w-3.5 h-3.5 animate-spin" />
            Pipeline running...
          </div>
        )}
      </div>
    </div>
  );
}
