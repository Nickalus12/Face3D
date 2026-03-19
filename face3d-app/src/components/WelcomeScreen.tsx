import { Upload, CheckCircle2, Clock, Award, Scan, ArrowLeft } from 'lucide-react';
import useSessionStore from '../store/sessionStore';

interface WelcomeScreenProps {
  onNewScan: () => void;
  onSelectSession: (session: any) => void;
}

function formatRelativeTime(dateStr?: string): string {
  if (!dateStr) return '';
  const date = new Date(dateStr);
  const now = Date.now();
  const diff = now - date.getTime();
  const minutes = Math.floor(diff / 60000);
  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 30) return `${days}d ago`;
  return date.toLocaleDateString();
}

function gradeColor(grade?: string): string {
  if (!grade) return 'text-zinc-600';
  if (grade === 'A') return 'text-emerald-400';
  if (grade === 'B') return 'text-blue-400';
  if (grade === 'C') return 'text-amber-400';
  return 'text-zinc-500';
}

export default function WelcomeScreen({ onNewScan }: WelcomeScreenProps) {
  const { sessions } = useSessionStore();
  const completedSessions = sessions.filter(s => s.has_gaussians);

  // Compute stats
  const totalScans = sessions.length;
  const completedCount = completedSessions.length;
  const avgGrade = (() => {
    const graded = completedSessions.filter(s => s.quality_grade);
    if (graded.length === 0) return '--';
    const scores = graded.map(s => {
      if (s.quality_grade === 'A') return 4;
      if (s.quality_grade === 'B') return 3;
      if (s.quality_grade === 'C') return 2;
      return 1;
    });
    const avg = scores.reduce((a, b) => a + b, 0) / scores.length;
    if (avg >= 3.5) return 'A';
    if (avg >= 2.5) return 'B';
    if (avg >= 1.5) return 'C';
    return 'D';
  })();
  const lastRun = sessions
    .filter(s => s.created_at)
    .sort((a, b) => new Date(b.created_at!).getTime() - new Date(a.created_at!).getTime())[0];

  return (
    <div className="h-full overflow-y-auto bg-[#0a0a0b]">
      <div className="max-w-[640px] mx-auto px-8 py-10">

        {/* Header */}
        <div className="flex items-center gap-3 mb-8">
          <div className="w-9 h-9 bg-gradient-to-br from-indigo-500 via-purple-500 to-indigo-600 rounded-xl flex items-center justify-center shadow-lg shadow-indigo-500/20 ring-1 ring-white/10">
            <span className="text-white font-black text-[11px] tracking-tighter">F3D</span>
          </div>
          <div>
            <h1 className="text-lg font-bold text-zinc-100 tracking-tight leading-tight">Face3D</h1>
            <p className="text-xs text-zinc-500">3D face reconstruction</p>
          </div>
        </div>

        {/* Stats row — only when there's data */}
        {sessions.length > 0 && (
          <div className="grid grid-cols-4 gap-3 mb-6">
            <div className="stat-card">
              <div className="flex items-center gap-1.5 mb-1">
                <Scan size={12} className="text-zinc-500" />
                <span className="text-[11px] text-zinc-500 font-medium">Total</span>
              </div>
              <span className="text-lg font-bold text-zinc-200 tabular-nums">{totalScans}</span>
            </div>
            <div className="stat-card">
              <div className="flex items-center gap-1.5 mb-1">
                <CheckCircle2 size={12} className="text-emerald-500/70" />
                <span className="text-[11px] text-zinc-500 font-medium">Done</span>
              </div>
              <span className="text-lg font-bold text-zinc-200 tabular-nums">{completedCount}</span>
            </div>
            <div className="stat-card">
              <div className="flex items-center gap-1.5 mb-1">
                <Award size={12} className="text-zinc-500" />
                <span className="text-[11px] text-zinc-500 font-medium">Avg Grade</span>
              </div>
              <span className={`text-lg font-bold tabular-nums ${gradeColor(avgGrade === '--' ? undefined : avgGrade)}`}>
                {avgGrade}
              </span>
            </div>
            <div className="stat-card">
              <div className="flex items-center gap-1.5 mb-1">
                <Clock size={12} className="text-zinc-500" />
                <span className="text-[11px] text-zinc-500 font-medium">Last Run</span>
              </div>
              <span className="text-sm font-medium text-zinc-300 truncate">
                {lastRun ? formatRelativeTime(lastRun.created_at) : '--'}
              </span>
            </div>
          </div>
        )}

        {/* New Scan CTA */}
        <button
          onClick={onNewScan}
          className="w-full group flex items-center gap-4 p-4 rounded-xl border border-indigo-500/20 bg-indigo-500/[0.04] hover:bg-indigo-500/[0.08] hover:border-indigo-500/30 transition-all duration-200 text-left active:scale-[0.99] mb-6 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500/50 focus-visible:ring-offset-2 focus-visible:ring-offset-[#0a0a0b]"
        >
          <div className="w-10 h-10 rounded-lg bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center group-hover:bg-indigo-500/20 transition-colors shrink-0">
            <Upload size={18} className="text-indigo-400" />
          </div>
          <div className="flex-1 min-w-0">
            <span className="text-sm font-semibold text-zinc-100 group-hover:text-white transition-colors">
              New Scan
            </span>
            <p className="text-xs text-zinc-500 mt-0.5">Import video, photos, and sensor data</p>
          </div>
        </button>

        {/* Contextual hint */}
        {sessions.length > 0 ? (
          <div className="flex items-center gap-2 text-xs text-zinc-500">
            <ArrowLeft size={12} />
            <span>Select a session from the sidebar to view it</span>
          </div>
        ) : (
          <p className="text-sm text-zinc-500">No sessions yet. Start a new scan to begin.</p>
        )}
      </div>
    </div>
  );
}
