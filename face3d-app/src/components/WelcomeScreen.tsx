import { Upload, ArrowRight, CheckCircle2, Box, Sparkles } from 'lucide-react';
import useSessionStore from '../store/sessionStore';

interface WelcomeScreenProps {
  onNewScan: () => void;
  onSelectSession: (session: any) => void;
}

export default function WelcomeScreen({ onNewScan, onSelectSession }: WelcomeScreenProps) {
  const { sessions } = useSessionStore();
  const completedSessions = sessions.filter(s => s.has_gaussians);
  const pendingSessions = sessions.filter(s => !s.has_gaussians);

  return (
    <div className="flex-1 overflow-y-auto bg-[#0a0a0b]">
      <div className="max-w-2xl mx-auto px-8 py-12">

        {/* Header */}
        <div className="mb-10">
          <div className="flex items-center gap-4 mb-4">
            <div className="w-12 h-12 bg-gradient-to-br from-indigo-500 via-purple-500 to-indigo-600 rounded-2xl flex items-center justify-center shadow-lg shadow-indigo-500/20 ring-1 ring-white/10">
              <span className="text-white font-black text-sm tracking-tighter">F3D</span>
            </div>
            <div>
              <h1 className="text-2xl font-bold text-zinc-100 tracking-tight">Face3D</h1>
              <p className="text-sm text-zinc-500">3D face reconstruction pipeline</p>
            </div>
          </div>
        </div>

        {/* New Scan — primary action */}
        <button
          onClick={onNewScan}
          className="w-full group p-6 rounded-2xl border border-indigo-500/20 bg-indigo-500/[0.03] hover:bg-indigo-500/[0.06] hover:border-indigo-500/30 transition-all duration-300 text-left active:scale-[0.99] mb-8"
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 rounded-xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center group-hover:bg-indigo-500/20 transition-colors">
                <Upload size={20} className="text-indigo-400" />
              </div>
              <div>
                <h3 className="text-base font-semibold text-zinc-100 group-hover:text-white transition-colors">
                  New Scan
                </h3>
                <p className="text-sm text-zinc-500 mt-0.5">
                  Import video, photos, and sensor data
                </p>
              </div>
            </div>
            <ArrowRight size={18} className="text-indigo-400/50 group-hover:text-indigo-400 group-hover:translate-x-1 transition-all" />
          </div>
        </button>

        {/* Sessions list */}
        {sessions.length > 0 && (
          <div>
            {/* Completed scans */}
            {completedSessions.length > 0 && (
              <div className="mb-6">
                <h3 className="text-xs font-semibold text-zinc-500 uppercase tracking-wider mb-3 flex items-center gap-2">
                  <Sparkles size={12} className="text-emerald-400" />
                  Completed ({completedSessions.length})
                </h3>
                <div className="space-y-1">
                  {completedSessions.map((session) => (
                    <button
                      key={session.id}
                      onClick={() => onSelectSession(session)}
                      className="w-full flex items-center gap-3 py-3 px-4 rounded-xl text-left hover:bg-zinc-800/40 transition-all group"
                    >
                      <CheckCircle2 size={16} className="text-emerald-500 shrink-0" />
                      <div className="flex-1 min-w-0">
                        <span className="text-sm font-medium text-zinc-200 group-hover:text-white transition-colors">
                          {session.name}
                        </span>
                        <div className="flex items-center gap-2 mt-0.5">
                          {session.has_gaussians && <span className="text-[11px] text-emerald-400/70">Splat</span>}
                          {session.has_mesh && <span className="text-[11px] text-blue-400/70">Mesh</span>}
                          {session.has_renders && <span className="text-[11px] text-purple-400/70">Renders</span>}
                          {session.quality_grade && (
                            <span className={`text-[11px] font-bold ${
                              session.quality_grade === 'A' ? 'text-emerald-400' :
                              session.quality_grade === 'B' ? 'text-blue-400' :
                              'text-zinc-500'
                            }`}>
                              Grade {session.quality_grade}
                            </span>
                          )}
                        </div>
                      </div>
                      <ArrowRight size={14} className="text-zinc-700 group-hover:text-zinc-400 shrink-0 transition-colors" />
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Pending scans */}
            {pendingSessions.length > 0 && (
              <div>
                <h3 className="text-xs font-semibold text-zinc-500 uppercase tracking-wider mb-3 flex items-center gap-2">
                  <Box size={12} className="text-zinc-500" />
                  In Progress ({pendingSessions.length})
                </h3>
                <div className="space-y-1">
                  {pendingSessions.map((session) => (
                    <button
                      key={session.id}
                      onClick={() => onSelectSession(session)}
                      className="w-full flex items-center gap-3 py-3 px-4 rounded-xl text-left hover:bg-zinc-800/40 transition-all group"
                    >
                      <div className="w-4 h-4 rounded-full border border-zinc-700 shrink-0" />
                      <span className="text-sm text-zinc-400 group-hover:text-zinc-200 transition-colors flex-1">
                        {session.name}
                      </span>
                      <ArrowRight size={14} className="text-zinc-700 group-hover:text-zinc-400 shrink-0 transition-colors" />
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Empty state */}
        {sessions.length === 0 && (
          <div className="text-center py-12">
            <p className="text-sm text-zinc-600">No sessions yet. Start a new scan to begin.</p>
          </div>
        )}

      </div>
    </div>
  );
}
