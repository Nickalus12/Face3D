import { useEffect, useState } from 'react';
import {
  ChevronDown,
  ChevronRight,
  Save,
  RotateCcw,
  Zap,
  FolderOpen,
  Sparkles,
  Camera,
  Package,
  Monitor,
  Check,
  Loader2,
} from 'lucide-react';
import useSettingsStore, { PRESETS, type Preset } from '../store/settingsStore';

// ── Collapsible Section ─────────────────────────────────────────

function Section({
  title,
  icon: Icon,
  children,
  defaultOpen = true,
}: {
  title: string;
  icon: React.ComponentType<{ size?: number; className?: string }>;
  children: React.ReactNode;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-xl overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-3 px-4 py-3 hover:bg-white/[0.02] transition-colors"
      >
        <Icon size={16} className="text-zinc-400 shrink-0" />
        <span className="text-[11px] font-bold tracking-wider text-zinc-500 uppercase flex-1 text-left">
          {title}
        </span>
        {open ? (
          <ChevronDown size={14} className="text-zinc-500" />
        ) : (
          <ChevronRight size={14} className="text-zinc-500" />
        )}
      </button>
      {open && <div className="px-4 pb-4 space-y-4">{children}</div>}
    </div>
  );
}

// ── Input Components ────────────────────────────────────────────

function SliderInput({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  suffix = '',
  formatValue,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (v: number) => void;
  suffix?: string;
  formatValue?: (v: number) => string;
}) {
  const display = formatValue ? formatValue(value) : `${value}${suffix}`;
  return (
    <div className="space-y-1.5">
      <div className="flex justify-between items-center">
        <label className="text-xs text-zinc-400">{label}</label>
        <span className="text-xs font-mono text-zinc-300">{display}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full h-1.5 bg-zinc-800 rounded-full appearance-none cursor-pointer
          [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3.5 [&::-webkit-slider-thumb]:h-3.5
          [&::-webkit-slider-thumb]:bg-indigo-500 [&::-webkit-slider-thumb]:rounded-full
          [&::-webkit-slider-thumb]:shadow-[0_0_8px_rgba(99,102,241,0.4)] [&::-webkit-slider-thumb]:cursor-pointer
          [&::-webkit-slider-thumb]:hover:bg-indigo-400 [&::-webkit-slider-thumb]:transition-colors"
      />
    </div>
  );
}

function NumberInput({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  suffix = '',
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (v: number) => void;
  suffix?: string;
}) {
  return (
    <div className="flex items-center justify-between gap-3">
      <label className="text-xs text-zinc-400 shrink-0">{label}</label>
      <div className="flex items-center gap-1">
        <button
          onClick={() => onChange(Math.max(min, value - step))}
          className="w-6 h-6 bg-zinc-800 border border-zinc-700 rounded text-zinc-400 hover:text-white hover:bg-zinc-700 transition-colors text-xs flex items-center justify-center"
        >
          -
        </button>
        <input
          type="number"
          value={value}
          min={min}
          max={max}
          step={step}
          onChange={(e) => {
            const v = Number(e.target.value);
            if (v >= min && v <= max) onChange(v);
          }}
          className="w-24 h-6 bg-zinc-800 border border-zinc-700 rounded text-xs text-zinc-200 text-center font-mono focus:outline-none focus:border-indigo-500 transition-colors"
        />
        <button
          onClick={() => onChange(Math.min(max, value + step))}
          className="w-6 h-6 bg-zinc-800 border border-zinc-700 rounded text-zinc-400 hover:text-white hover:bg-zinc-700 transition-colors text-xs flex items-center justify-center"
        >
          +
        </button>
        {suffix && <span className="text-[10px] text-zinc-500 ml-1">{suffix}</span>}
      </div>
    </div>
  );
}

function Toggle({
  label,
  checked,
  onChange,
  description,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
  description?: string;
}) {
  return (
    <div className="flex items-center justify-between gap-3">
      <div className="flex-1 min-w-0">
        <div className="text-xs text-zinc-400">{label}</div>
        {description && <div className="text-[10px] text-zinc-600 mt-0.5">{description}</div>}
      </div>
      <button
        onClick={() => onChange(!checked)}
        className={`relative w-9 h-5 rounded-full transition-colors duration-200 shrink-0 ${
          checked ? 'bg-indigo-500' : 'bg-zinc-700'
        }`}
      >
        <div
          className={`absolute top-0.5 w-4 h-4 bg-white rounded-full shadow-md transition-transform duration-200 ${
            checked ? 'translate-x-4' : 'translate-x-0.5'
          }`}
        />
      </button>
    </div>
  );
}

function SelectInput({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string | number;
  options: { value: string | number; label: string }[];
  onChange: (v: string | number) => void;
}) {
  return (
    <div className="flex items-center justify-between gap-3">
      <label className="text-xs text-zinc-400 shrink-0">{label}</label>
      <select
        value={value}
        onChange={(e) => {
          const opt = options.find((o) => String(o.value) === e.target.value);
          if (opt) onChange(opt.value);
        }}
        className="h-7 px-2 bg-zinc-800 border border-zinc-700 rounded text-xs text-zinc-200 focus:outline-none focus:border-indigo-500 transition-colors cursor-pointer"
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </div>
  );
}

function TextInput({
  label,
  value,
  onChange,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <div className="space-y-1">
      <label className="text-xs text-zinc-400">{label}</label>
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full h-7 px-2 bg-zinc-800 border border-zinc-700 rounded text-xs text-zinc-200 font-mono focus:outline-none focus:border-indigo-500 transition-colors"
      />
    </div>
  );
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between items-start gap-3 py-1">
      <span className="text-xs text-zinc-500 shrink-0">{label}</span>
      <span className="text-xs text-zinc-300 text-right font-mono break-all">{value}</span>
    </div>
  );
}

// ── Preset Card ─────────────────────────────────────────────────

function PresetCard({
  preset,
  isActive,
  onApply,
}: {
  preset: Preset;
  isActive: boolean;
  onApply: () => void;
}) {
  return (
    <button
      onClick={onApply}
      className={`flex-1 p-3 rounded-lg border text-left transition-all ${
        isActive
          ? 'bg-indigo-500/10 border-indigo-500/30 ring-1 ring-indigo-500/20'
          : 'bg-zinc-800/50 border-zinc-700/50 hover:bg-zinc-800 hover:border-zinc-600'
      }`}
    >
      <div className="flex items-center justify-between mb-1.5">
        <span className={`text-xs font-semibold ${isActive ? 'text-indigo-300' : 'text-zinc-300'}`}>
          {preset.name}
        </span>
        {isActive && <Check size={12} className="text-indigo-400" />}
      </div>
      <div className="text-[10px] text-zinc-500 mb-2">{preset.description}</div>
      <div className="flex gap-3">
        <span className="text-[10px] text-zinc-400">
          <span className="text-zinc-600">Time:</span> {preset.estimated_time}
        </span>
        <span className="text-[10px] text-zinc-400">
          <span className="text-zinc-600">VRAM:</span> {preset.estimated_vram}
        </span>
      </div>
    </button>
  );
}

// ── VRAM Estimate ───────────────────────────────────────────────

function estimateVram(gaussians: number): string {
  // Rough: ~200 bytes per gaussian for training
  const gb = (gaussians * 200) / 1_073_741_824;
  return `~${Math.max(2, gb + 2).toFixed(1)} GB`;
}

// ── Main Component ──────────────────────────────────────────────

export default function SettingsPanel() {
  const {
    training,
    capture,
    export_,
    paths,
    gpuInfo,
    pythonInfo,
    systemInfo,
    loading,
    saving,
    isDirty,
    activePreset,
    loadConfig,
    saveConfig,
    resetToDefaults,
    applyPreset,
    updateTraining,
    updateCapture,
    updateExport,
    updatePaths,
    fetchSystemInfo,
  } = useSettingsStore();

  const [saveSuccess, setSaveSuccess] = useState(false);

  useEffect(() => {
    loadConfig();
    fetchSystemInfo();
  }, [loadConfig, fetchSystemInfo]);

  const handleSave = async () => {
    const ok = await saveConfig();
    if (ok) {
      setSaveSuccess(true);
      setTimeout(() => setSaveSuccess(false), 2000);
    }
  };

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <Loader2 size={24} className="text-zinc-500 animate-spin" />
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-zinc-800/60 shrink-0 bg-[#0a0a0b]/80 backdrop-blur-sm">
        <div>
          <h1 className="text-sm font-semibold text-zinc-200 tracking-wide">Settings</h1>
          <p className="text-[10px] text-zinc-500 mt-0.5">Configure pipeline parameters and system paths</p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={resetToDefaults}
            className="flex items-center gap-1.5 px-3 py-1.5 text-[11px] text-zinc-400 bg-zinc-800 border border-zinc-700 rounded-lg hover:text-zinc-200 hover:bg-zinc-700 transition-colors"
          >
            <RotateCcw size={12} />
            Reset to Defaults
          </button>
          <button
            onClick={handleSave}
            disabled={!isDirty || saving}
            className={`flex items-center gap-1.5 px-4 py-1.5 text-[11px] font-medium rounded-lg transition-all ${
              saveSuccess
                ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                : isDirty
                ? 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 hover:bg-indigo-500/30 shadow-lg shadow-indigo-500/10'
                : 'bg-zinc-800 text-zinc-500 border border-zinc-700/50 cursor-not-allowed'
            }`}
          >
            {saving ? (
              <Loader2 size={12} className="animate-spin" />
            ) : saveSuccess ? (
              <Check size={12} />
            ) : (
              <Save size={12} />
            )}
            {saveSuccess ? 'Saved' : saving ? 'Saving...' : 'Save'}
            {isDirty && !saveSuccess && (
              <span className="w-1.5 h-1.5 bg-amber-400 rounded-full ml-1 animate-pulse" />
            )}
          </button>
        </div>
      </div>

      {/* Scrollable Content */}
      <div className="flex-1 overflow-y-auto p-6 space-y-4 scrollbar-hide">
        {/* Pipeline Presets */}
        <Section title="Pipeline Presets" icon={Zap} defaultOpen={true}>
          <div className="flex gap-3">
            {PRESETS.map((preset) => (
              <PresetCard
                key={preset.name}
                preset={preset}
                isActive={activePreset === preset.name}
                onApply={() => applyPreset(preset)}
              />
            ))}
          </div>
        </Section>

        {/* Training Parameters */}
        <Section title="Training Parameters" icon={Sparkles} defaultOpen={true}>
          <SliderInput
            label="Iterations"
            value={training.iterations}
            min={1000}
            max={30000}
            step={500}
            onChange={(v) => updateTraining({ iterations: v })}
            formatValue={(v) => v.toLocaleString()}
          />

          <NumberInput
            label="Max Gaussians"
            value={training.max_gaussians}
            min={100000}
            max={1000000}
            step={50000}
            onChange={(v) => updateTraining({ max_gaussians: v })}
          />
          <div className="text-[10px] text-zinc-600 -mt-2 pl-1">
            Est. VRAM: {estimateVram(training.max_gaussians)}
          </div>

          <SelectInput
            label="SH Degree"
            value={training.sh_degree}
            options={[
              { value: 0, label: '0 — Fastest, flat color' },
              { value: 1, label: '1 — Basic shading' },
              { value: 2, label: '2 — Good quality (default)' },
              { value: 3, label: '3 — Best quality, slowest' },
            ]}
            onChange={(v) => updateTraining({ sh_degree: Number(v) })}
          />

          <SelectInput
            label="Strategy"
            value={training.strategy}
            options={[
              { value: 'default', label: 'Default' },
              { value: 'mcmc', label: 'MCMC' },
            ]}
            onChange={(v) => updateTraining({ strategy: v as 'default' | 'mcmc' })}
          />

          <Toggle
            label="Use 2DGS"
            checked={training.use_2dgs}
            onChange={(v) => updateTraining({ use_2dgs: v })}
            description="2D Gaussian Splatting (flatter splats, better surfaces)"
          />

          <Toggle
            label="Appearance Embedding"
            checked={training.use_appearance_embedding}
            onChange={(v) => updateTraining({ use_appearance_embedding: v })}
            description="Per-image appearance correction"
          />

          <Toggle
            label="Progressive Training"
            checked={training.progressive_training}
            onChange={(v) => updateTraining({ progressive_training: v })}
            description="Start at low resolution and increase"
          />
        </Section>

        {/* Capture Settings */}
        <Section title="Capture Settings" icon={Camera} defaultOpen={false}>
          <SliderInput
            label="Max Frames"
            value={capture.max_frames}
            min={20}
            max={300}
            step={10}
            onChange={(v) => updateCapture({ max_frames: v })}
          />

          <SliderInput
            label="Blur Threshold"
            value={capture.blur_threshold}
            min={10}
            max={200}
            step={5}
            onChange={(v) => updateCapture({ blur_threshold: v })}
            formatValue={(v) => `${v} (Laplacian variance)`}
          />

          <Toggle
            label="Require Face"
            checked={capture.require_face}
            onChange={(v) => updateCapture({ require_face: v })}
            description="Only keep frames with detected faces"
          />

          <Toggle
            label="Scene Change Only"
            checked={capture.scene_change_only}
            onChange={(v) => updateCapture({ scene_change_only: v })}
            description="Use ffmpeg scene detection (10x faster)"
          />
        </Section>

        {/* Export Settings */}
        <Section title="Export Settings" icon={Package} defaultOpen={false}>
          <SelectInput
            label="Texture Resolution"
            value={export_.texture_resolution}
            options={[
              { value: 1024, label: '1024 — Fast' },
              { value: 2048, label: '2048 — Balanced' },
              { value: 4096, label: '4096 — High Quality' },
            ]}
            onChange={(v) => updateExport({ texture_resolution: Number(v) })}
          />

          <SliderInput
            label="Turntable Views"
            value={export_.turntable_views}
            min={10}
            max={120}
            step={5}
            onChange={(v) => updateExport({ turntable_views: v })}
          />

          <Toggle
            label="Export Compressed"
            checked={export_.export_compressed}
            onChange={(v) => updateExport({ export_compressed: v })}
            description="Compress exported PLY files"
          />
        </Section>

        {/* System Info */}
        <Section title="System Info" icon={Monitor} defaultOpen={false}>
          <div className="space-y-0.5">
            <div className="text-[10px] font-bold tracking-wider text-zinc-600 uppercase mb-2">GPU</div>
            <InfoRow label="Name" value={gpuInfo?.name ?? 'Loading...'} />
            <InfoRow label="VRAM Total" value={gpuInfo?.memory_total ?? '--'} />
            <InfoRow label="VRAM Used" value={gpuInfo?.memory_used ?? '--'} />
            <InfoRow label="Utilization" value={gpuInfo?.utilization ?? '--'} />
          </div>

          <div className="border-t border-zinc-800 pt-3 space-y-0.5">
            <div className="text-[10px] font-bold tracking-wider text-zinc-600 uppercase mb-2">Python</div>
            <InfoRow label="Version" value={pythonInfo?.version ?? 'Loading...'} />
            <InfoRow label="Conda Env" value={pythonInfo?.env_name ?? '--'} />
            <InfoRow label="Packages" value={pythonInfo ? `${pythonInfo.packages_count} installed` : '--'} />
          </div>

          <div className="border-t border-zinc-800 pt-3 space-y-0.5">
            <div className="text-[10px] font-bold tracking-wider text-zinc-600 uppercase mb-2">System</div>
            <InfoRow label="OS" value={systemInfo?.os_version ?? 'Loading...'} />
            <InfoRow label="CPU" value={systemInfo?.cpu ?? '--'} />
            <InfoRow label="RAM" value={systemInfo ? `${systemInfo.ram_gb.toFixed(1)} GB` : '--'} />
            <InfoRow label="CUDA" value={systemInfo?.cuda_version ?? '--'} />
          </div>

          <div className="border-t border-zinc-800 pt-3 space-y-0.5">
            <div className="text-[10px] font-bold tracking-wider text-zinc-600 uppercase mb-2">Disk (D:)</div>
            <InfoRow
              label="Free"
              value={systemInfo ? `${systemInfo.disk_free_gb.toFixed(1)} GB` : '--'}
            />
            <InfoRow
              label="Total"
              value={systemInfo ? `${systemInfo.disk_total_gb.toFixed(1)} GB` : '--'}
            />
          </div>
        </Section>

        {/* Paths */}
        <Section title="Paths" icon={FolderOpen} defaultOpen={false}>
          <TextInput
            label="Project Root"
            value={paths.project_root}
            onChange={(v) => updatePaths({ project_root: v })}
          />
          <TextInput
            label="COLMAP Binary"
            value={paths.colmap_binary}
            onChange={(v) => updatePaths({ colmap_binary: v })}
          />
          <TextInput
            label="FLAME Model"
            value={paths.flame_model}
            onChange={(v) => updatePaths({ flame_model: v })}
          />
          <TextInput
            label="Data Root"
            value={paths.data_root}
            onChange={(v) => updatePaths({ data_root: v })}
          />
        </Section>

        {/* Bottom spacer */}
        <div className="h-4" />
      </div>
    </div>
  );
}
