import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";

// ── Types matching Rust backend ──────────────────────────────────

export interface Session {
  id: string;
  name: string;
  has_gaussians: boolean;
  has_mesh: boolean;
  has_renders: boolean;
}

export interface GpuInfo {
  name: string;
  memory_total: string;
  memory_used: string;
  utilization: string;
}

export interface PipelineLogPayload {
  line: string;
  level: string;
}

// ── Tauri detection ──────────────────────────────────────────────

function isTauri(): boolean {
  return typeof window !== "undefined" && !!(window as any).__TAURI_INTERNALS__;
}

// ── Session commands ─────────────────────────────────────────────

export async function listSessions(): Promise<Session[]> {
  if (!isTauri()) {
    return [];
  }
  try {
    return await invoke<Session[]>("list_sessions");
  } catch (e) {
    console.error("list_sessions failed:", e);
    return [];
  }
}

export async function getSessionFiles(
  sessionId: string,
): Promise<[string, number][]> {
  if (!isTauri()) {
    return [];
  }
  try {
    return await invoke<[string, number][]>("get_session_files", {
      sessionId,
    });
  } catch (e) {
    console.error("get_session_files failed:", e);
    return [];
  }
}

// ── Pipeline commands ────────────────────────────────────────────

export async function startPipeline(
  contentDir: string,
  session: string,
): Promise<void> {
  if (!isTauri()) {
    console.warn("[dev] startPipeline called outside Tauri");
    return;
  }
  await invoke("start_pipeline", { contentDir, session });
}

export async function stopPipeline(): Promise<void> {
  if (!isTauri()) {
    console.warn("[dev] stopPipeline called outside Tauri");
    return;
  }
  await invoke("stop_pipeline");
}

export async function isPipelineRunning(): Promise<boolean> {
  if (!isTauri()) return false;
  try {
    return await invoke<boolean>("is_pipeline_running");
  } catch {
    return false;
  }
}

// ── GPU ──────────────────────────────────────────────────────────

export async function getGpuInfo(): Promise<GpuInfo | null> {
  if (!isTauri()) {
    return null;
  }
  try {
    return await invoke<GpuInfo>("get_gpu_info");
  } catch (e) {
    console.error("get_gpu_info failed:", e);
    return null;
  }
}

// ── File system helpers ──────────────────────────────────────────

export async function openFolder(path: string): Promise<void> {
  if (!isTauri()) return;
  try {
    await invoke("open_folder", { path });
  } catch (e) {
    console.error("open_folder failed:", e);
  }
}

export async function selectContentDir(): Promise<string | null> {
  if (!isTauri()) {
    return null;
  }
  try {
    const selected = await open({
      directory: true,
      multiple: false,
      title: "Select Content Directory",
    });
    if (selected && typeof selected === "string") {
      return selected;
    }
    return null;
  } catch {
    return null;
  }
}

// ── Event listeners ──────────────────────────────────────────────

export function onPipelineLog(
  handler: (payload: PipelineLogPayload) => void,
): Promise<UnlistenFn> | null {
  if (!isTauri()) return null;
  return listen<PipelineLogPayload>("pipeline-log", (event) => {
    handler(event.payload);
  });
}

export function onPipelineComplete(
  handler: (exitCode: number) => void,
): Promise<UnlistenFn> | null {
  if (!isTauri()) return null;
  return listen<number>("pipeline-complete", (event) => {
    handler(event.payload);
  });
}
