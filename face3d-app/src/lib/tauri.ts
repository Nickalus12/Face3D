import { invoke } from "@tauri-apps/api/core";

export interface Session {
  id: string;
  name: string;
  created: string;
  stage: number;
  contentDir: string;
}

export interface GpuInfo {
  name: string;
  vramTotal: number;
  vramUsed: number;
  temperature: number;
  utilization: number;
}

export interface PipelineStatus {
  isRunning: boolean;
  currentStage: number;
  sessionId: string;
}

export async function listSessions(): Promise<Session[]> {
  try {
    return await invoke<Session[]>("list_sessions");
  } catch {
    // Return mock data during development
    return [
      {
        id: "session_20260318_001",
        name: "Front Profile Scan",
        created: "2026-03-18T14:30:00",
        stage: 14,
        contentDir: "D:/Projects/3D/data/processed/session_20260318_001",
      },
      {
        id: "session_20260317_002",
        name: "Side Angle Capture",
        created: "2026-03-17T09:15:00",
        stage: 8,
        contentDir: "D:/Projects/3D/data/processed/session_20260317_002",
      },
      {
        id: "session_20260316_001",
        name: "Full Head Scan",
        created: "2026-03-16T16:45:00",
        stage: 3,
        contentDir: "D:/Projects/3D/data/processed/session_20260316_001",
      },
    ];
  }
}

export async function startPipeline(
  contentDir: string,
  session: string,
  startStage?: number,
): Promise<void> {
  try {
    await invoke("start_pipeline", { contentDir, session, startStage });
  } catch (e) {
    console.warn("start_pipeline invoke failed (dev mode):", e);
  }
}

export async function stopPipeline(): Promise<void> {
  try {
    await invoke("stop_pipeline");
  } catch (e) {
    console.warn("stop_pipeline invoke failed (dev mode):", e);
  }
}

export async function getGpuInfo(): Promise<GpuInfo> {
  try {
    return await invoke<GpuInfo>("get_gpu_info");
  } catch {
    return {
      name: "NVIDIA RTX 3080",
      vramTotal: 16384,
      vramUsed: 4200,
      temperature: 52,
      utilization: 15,
    };
  }
}

export async function openFolder(path: string): Promise<void> {
  try {
    await invoke("open_folder", { path });
  } catch (e) {
    console.warn("open_folder invoke failed (dev mode):", e);
  }
}

export async function getPipelineStatus(): Promise<PipelineStatus> {
  try {
    return await invoke<PipelineStatus>("get_pipeline_status");
  } catch {
    return { isRunning: false, currentStage: 0, sessionId: "" };
  }
}
