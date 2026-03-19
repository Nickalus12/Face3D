import React, { useEffect, useState } from "react";
import { CheckCircle2, XCircle, Info, AlertTriangle, X } from "lucide-react";
import useToastStore, { type Toast as ToastData } from "../store/toastStore";

const ICON_MAP = {
  success: CheckCircle2,
  error: XCircle,
  info: Info,
  warning: AlertTriangle,
} as const;

const COLOR_MAP = {
  success: {
    bg: "bg-emerald-500/10",
    border: "border-emerald-500/30",
    text: "text-emerald-400",
    icon: "text-emerald-400",
  },
  error: {
    bg: "bg-red-500/10",
    border: "border-red-500/30",
    text: "text-red-300",
    icon: "text-red-400",
  },
  info: {
    bg: "bg-blue-500/10",
    border: "border-blue-500/30",
    text: "text-blue-300",
    icon: "text-blue-400",
  },
  warning: {
    bg: "bg-amber-500/10",
    border: "border-amber-500/30",
    text: "text-amber-300",
    icon: "text-amber-400",
  },
} as const;

interface ToastItemProps {
  toast: ToastData;
  onRemove: (id: string) => void;
}

const ToastItem: React.FC<ToastItemProps> = ({ toast, onRemove }) => {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    // Trigger slide-in on mount
    const raf = requestAnimationFrame(() => setVisible(true));
    return () => cancelAnimationFrame(raf);
  }, []);

  const colors = COLOR_MAP[toast.type];
  const Icon = ICON_MAP[toast.type];

  const handleRemove = () => {
    setVisible(false);
    setTimeout(() => onRemove(toast.id), 200);
  };

  return (
    <div
      className={`flex items-start gap-3 px-4 py-3 rounded-lg border backdrop-blur-md shadow-xl transition-all duration-200 ease-out ${colors.bg} ${colors.border} ${
        visible ? "translate-x-0 opacity-100" : "translate-x-8 opacity-0"
      }`}
      style={{ minWidth: 280, maxWidth: 400 }}
    >
      <Icon size={18} className={`${colors.icon} shrink-0 mt-0.5`} />
      <p className={`text-sm flex-1 ${colors.text}`}>{toast.message}</p>
      <button
        onClick={handleRemove}
        className="text-zinc-500 hover:text-zinc-300 transition-colors shrink-0 mt-0.5"
      >
        <X size={14} />
      </button>
    </div>
  );
};

const ToastContainer: React.FC = () => {
  const { toasts, removeToast } = useToastStore();

  if (toasts.length === 0) return null;

  return (
    <div className="fixed bottom-6 right-6 z-[100] flex flex-col-reverse gap-2 pointer-events-auto">
      {toasts.map((toast) => (
        <ToastItem key={toast.id} toast={toast} onRemove={removeToast} />
      ))}
    </div>
  );
};

export default ToastContainer;
