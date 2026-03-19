import { create } from "zustand";
import yaml from "js-yaml";
import {
  getPipelineConfig,
  savePipelineConfig,
  getPythonInfo,
  getSystemInfo,
  getGpuInfo,
  type PythonInfo,
  type SystemInfo,
  type GpuInfo,
} from "../lib/api";

// ── Types ────────────────────────────────────────────────────────

export interface TrainingParams {
  iterations: number;
  max_gaussians: number;
  sh_degree: number;
  strategy: "default" | "mcmc";
  use_2dgs: boolean;
  use_appearance_embedding: boolean;
  progressive_training: boolean;
  grow_grad2d: number;
  refine_stop_iter: number;
  refine_every: number;
}

export interface CaptureParams {
  max_frames: number;
  blur_threshold: number;
  require_face: boolean;
  scene_change_only: boolean;
}

export interface ExportParams {
  texture_resolution: number;
  turntable_views: number;
  export_compressed: boolean;
}

export interface PathsConfig {
  project_root: string;
  colmap_binary: string;
  flame_model: string;
  data_root: string;
}

export interface Preset {
  name: string;
  description: string;
  estimated_time: string;
  estimated_vram: string;
  training: Partial<TrainingParams>;
}

export const PRESETS: Preset[] = [
  {
    name: "Quick Test",
    description: "Fast iteration for testing pipeline flow",
    estimated_time: "~2 min",
    estimated_vram: "~4 GB",
    training: {
      iterations: 1000,
      max_gaussians: 100000,
      progressive_training: false,
    },
  },
  {
    name: "Balanced",
    description: "Good quality with reasonable training time",
    estimated_time: "~8 min",
    estimated_vram: "~8 GB",
    training: {
      iterations: 3000,
      max_gaussians: 300000,
    },
  },
  {
    name: "Maximum Quality",
    description: "Highest quality output, longer training",
    estimated_time: "~30 min",
    estimated_vram: "~14 GB",
    training: {
      iterations: 10000,
      max_gaussians: 500000,
      sh_degree: 3,
    },
  },
];

// Default values matching pipeline.yaml
const DEFAULT_TRAINING: TrainingParams = {
  iterations: 3000,
  max_gaussians: 300000,
  sh_degree: 2,
  strategy: "default",
  use_2dgs: false,
  use_appearance_embedding: false,
  progressive_training: true,
  grow_grad2d: 0.0005,
  refine_stop_iter: 1800,
  refine_every: 500,
};

const DEFAULT_CAPTURE: CaptureParams = {
  max_frames: 80,
  blur_threshold: 30,
  require_face: false,
  scene_change_only: true,
};

const DEFAULT_EXPORT: ExportParams = {
  texture_resolution: 2048,
  turntable_views: 30,
  export_compressed: false,
};

const DEFAULT_PATHS: PathsConfig = {
  project_root: "D:/Projects/3D",
  colmap_binary:
    "D:/Projects/3D/COLMAP/COLMAP-3.9.1-windows-cuda/COLMAP.bat",
  flame_model: "D:/Projects/3D/Models/Flame/flame2023_Open.pkl",
  data_root: "D:/Projects/3D/data",
};

// ── Store ────────────────────────────────────────────────────────

interface SettingsState {
  // Loaded raw yaml for round-trip preservation
  rawYaml: string | null;
  parsedConfig: Record<string, any> | null;

  // Editable sections
  training: TrainingParams;
  capture: CaptureParams;
  export_: ExportParams;
  paths: PathsConfig;

  // Saved snapshots for dirty detection
  savedTraining: TrainingParams;
  savedCapture: CaptureParams;
  savedExport: ExportParams;
  savedPaths: PathsConfig;

  // System info (read-only)
  gpuInfo: GpuInfo | null;
  pythonInfo: PythonInfo | null;
  systemInfo: SystemInfo | null;

  // State
  loading: boolean;
  saving: boolean;
  isDirty: boolean;
  activePreset: string | null;

  // Actions
  loadConfig: () => Promise<void>;
  saveConfig: () => Promise<boolean>;
  resetToDefaults: () => void;
  applyPreset: (preset: Preset) => void;
  updateTraining: (partial: Partial<TrainingParams>) => void;
  updateCapture: (partial: Partial<CaptureParams>) => void;
  updateExport: (partial: Partial<ExportParams>) => void;
  updatePaths: (partial: Partial<PathsConfig>) => void;
  fetchSystemInfo: () => Promise<void>;
  computeDirty: () => void;
}

const useSettingsStore = create<SettingsState>((set, get) => ({
  rawYaml: null,
  parsedConfig: null,

  training: { ...DEFAULT_TRAINING },
  capture: { ...DEFAULT_CAPTURE },
  export_: { ...DEFAULT_EXPORT },
  paths: { ...DEFAULT_PATHS },

  savedTraining: { ...DEFAULT_TRAINING },
  savedCapture: { ...DEFAULT_CAPTURE },
  savedExport: { ...DEFAULT_EXPORT },
  savedPaths: { ...DEFAULT_PATHS },

  gpuInfo: null,
  pythonInfo: null,
  systemInfo: null,

  loading: false,
  saving: false,
  isDirty: false,
  activePreset: null,

  loadConfig: async () => {
    set({ loading: true });
    const rawYaml = await getPipelineConfig();
    if (!rawYaml) {
      set({ loading: false });
      return;
    }

    try {
      const parsed = yaml.load(rawYaml) as Record<string, any>;

      const training: TrainingParams = {
        iterations: parsed?.splatting?.training?.iterations ?? DEFAULT_TRAINING.iterations,
        max_gaussians: parsed?.splatting?.training?.max_num_gaussians ?? DEFAULT_TRAINING.max_gaussians,
        sh_degree: DEFAULT_TRAINING.sh_degree,
        strategy: DEFAULT_TRAINING.strategy,
        use_2dgs: DEFAULT_TRAINING.use_2dgs,
        use_appearance_embedding: DEFAULT_TRAINING.use_appearance_embedding,
        progressive_training: DEFAULT_TRAINING.progressive_training,
        grow_grad2d: parsed?.splatting?.training?.grow_grad2d ?? DEFAULT_TRAINING.grow_grad2d,
        refine_stop_iter: parsed?.splatting?.training?.refine_stop_iter ?? DEFAULT_TRAINING.refine_stop_iter,
        refine_every: parsed?.splatting?.training?.refine_every ?? DEFAULT_TRAINING.refine_every,
      };

      const capture: CaptureParams = {
        max_frames: parsed?.capture?.max_frames ?? DEFAULT_CAPTURE.max_frames,
        blur_threshold: parsed?.capture?.filtering?.blur_threshold ?? DEFAULT_CAPTURE.blur_threshold,
        require_face: parsed?.capture?.filtering?.require_face ?? DEFAULT_CAPTURE.require_face,
        scene_change_only: parsed?.capture?.scene_change_only ?? DEFAULT_CAPTURE.scene_change_only,
      };

      const export_: ExportParams = {
        texture_resolution: parsed?.splatting?.export?.texture_resolution ?? DEFAULT_EXPORT.texture_resolution,
        turntable_views: parsed?.splatting?.export?.turntable_views ?? DEFAULT_EXPORT.turntable_views,
        export_compressed: DEFAULT_EXPORT.export_compressed,
      };

      const paths: PathsConfig = {
        project_root: "D:/Projects/3D",
        colmap_binary: parsed?.reconstruction?.colmap?.binary ?? DEFAULT_PATHS.colmap_binary,
        flame_model: parsed?.reconstruction?.flame?.model_path ?? DEFAULT_PATHS.flame_model,
        data_root: parsed?.session?.data_root ?? DEFAULT_PATHS.data_root,
      };

      set({
        rawYaml,
        parsedConfig: parsed,
        training,
        capture,
        export_: export_,
        paths,
        savedTraining: { ...training },
        savedCapture: { ...capture },
        savedExport: { ...export_ },
        savedPaths: { ...paths },
        loading: false,
        isDirty: false,
      });
    } catch (e) {
      console.error("Failed to parse pipeline.yaml:", e);
      set({ rawYaml, loading: false });
    }
  },

  saveConfig: async () => {
    const state = get();
    if (!state.parsedConfig) return false;

    set({ saving: true });

    // Deep clone parsedConfig and update with current values
    const config = JSON.parse(JSON.stringify(state.parsedConfig));

    // Training
    if (!config.splatting) config.splatting = {};
    if (!config.splatting.training) config.splatting.training = {};
    config.splatting.training.iterations = state.training.iterations;
    config.splatting.training.max_num_gaussians = state.training.max_gaussians;
    config.splatting.training.grow_grad2d = state.training.grow_grad2d;
    config.splatting.training.refine_stop_iter = state.training.refine_stop_iter;
    config.splatting.training.refine_every = state.training.refine_every;

    // Capture
    if (!config.capture) config.capture = {};
    config.capture.max_frames = state.capture.max_frames;
    config.capture.scene_change_only = state.capture.scene_change_only;
    if (!config.capture.filtering) config.capture.filtering = {};
    config.capture.filtering.blur_threshold = state.capture.blur_threshold;
    config.capture.filtering.require_face = state.capture.require_face;

    // Export
    if (!config.splatting.export) config.splatting.export = {};
    config.splatting.export.texture_resolution = state.export_.texture_resolution;
    config.splatting.export.turntable_views = state.export_.turntable_views;

    // Paths
    if (!config.reconstruction) config.reconstruction = {};
    if (!config.reconstruction.colmap) config.reconstruction.colmap = {};
    config.reconstruction.colmap.binary = state.paths.colmap_binary;
    if (!config.reconstruction.flame) config.reconstruction.flame = {};
    config.reconstruction.flame.model_path = state.paths.flame_model;
    if (!config.session) config.session = {};
    config.session.data_root = state.paths.data_root;

    const yamlStr = yaml.dump(config, {
      lineWidth: 120,
      noRefs: true,
      quotingType: '"',
      forceQuotes: false,
    });

    const success = await savePipelineConfig(yamlStr);

    if (success) {
      set({
        saving: false,
        isDirty: false,
        parsedConfig: config,
        rawYaml: yamlStr,
        savedTraining: { ...state.training },
        savedCapture: { ...state.capture },
        savedExport: { ...state.export_ },
        savedPaths: { ...state.paths },
      });
    } else {
      set({ saving: false });
    }

    return success;
  },

  resetToDefaults: () => {
    set({
      training: { ...DEFAULT_TRAINING },
      capture: { ...DEFAULT_CAPTURE },
      export_: { ...DEFAULT_EXPORT },
      paths: { ...DEFAULT_PATHS },
      activePreset: null,
    });
    get().computeDirty();
  },

  applyPreset: (preset: Preset) => {
    set((state) => ({
      training: { ...state.training, ...preset.training },
      activePreset: preset.name,
    }));
    get().computeDirty();
  },

  updateTraining: (partial) => {
    set((state) => ({
      training: { ...state.training, ...partial },
      activePreset: null,
    }));
    get().computeDirty();
  },

  updateCapture: (partial) => {
    set((state) => ({
      capture: { ...state.capture, ...partial },
    }));
    get().computeDirty();
  },

  updateExport: (partial) => {
    set((state) => ({
      export_: { ...state.export_, ...partial },
    }));
    get().computeDirty();
  },

  updatePaths: (partial) => {
    set((state) => ({
      paths: { ...state.paths, ...partial },
    }));
    get().computeDirty();
  },

  fetchSystemInfo: async () => {
    const [gpuInfo, pythonInfo, systemInfo] = await Promise.all([
      getGpuInfo(),
      getPythonInfo(),
      getSystemInfo(),
    ]);
    set({ gpuInfo, pythonInfo, systemInfo });
  },

  computeDirty: () => {
    const s = get();
    const dirty =
      JSON.stringify(s.training) !== JSON.stringify(s.savedTraining) ||
      JSON.stringify(s.capture) !== JSON.stringify(s.savedCapture) ||
      JSON.stringify(s.export_) !== JSON.stringify(s.savedExport) ||
      JSON.stringify(s.paths) !== JSON.stringify(s.savedPaths);
    set({ isDirty: dirty });
  },
}));

export default useSettingsStore;
