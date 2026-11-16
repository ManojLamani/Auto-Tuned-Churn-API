export const pct = (v, digits = 1) => `${(v * 100).toFixed(digits)}%`;

export function Card({ title, subtitle, action, children, className = "" }) {
  return (
    <section className={`rounded-2xl border border-slate-200 bg-white p-5 shadow-sm ${className}`}>
      {(title || action) && (
        <header className="mb-4 flex items-start justify-between gap-3">
          <div>
            <h2 className="text-sm font-semibold text-slate-900">{title}</h2>
            {subtitle && <p className="mt-0.5 text-xs text-slate-500">{subtitle}</p>}
          </div>
          {action}
        </header>
      )}
      {children}
    </section>
  );
}

export function Kpi({ label, value, hint, accent = false }) {
  return (
    <div
      className={`rounded-2xl border p-5 shadow-sm ${
        accent ? "border-indigo-500 bg-indigo-600 text-white" : "border-slate-200 bg-white"
      }`}
    >
      <p className={`text-xs font-medium uppercase tracking-wide ${accent ? "text-indigo-100" : "text-slate-500"}`}>
        {label}
      </p>
      <p className="mt-2 text-3xl font-semibold tabular-nums">{value}</p>
      {hint && <p className={`mt-1 text-xs ${accent ? "text-indigo-100" : "text-slate-500"}`}>{hint}</p>}
    </div>
  );
}

export const TIERS = {
  HIGH: { icon: "▲", label: "High risk", badge: "bg-rose-50 text-rose-700 ring-rose-200", bar: "bg-rose-500" },
  MEDIUM: { icon: "●", label: "Medium risk", badge: "bg-amber-50 text-amber-700 ring-amber-200", bar: "bg-amber-400" },
  LOW: { icon: "▼", label: "Low risk", badge: "bg-emerald-50 text-emerald-700 ring-emerald-200", bar: "bg-emerald-500" },
};

export function TierBadge({ tier, large = false }) {
  const t = TIERS[tier];
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full font-semibold ring-1 ring-inset ${t.badge} ${
        large ? "px-3 py-1 text-sm" : "px-2 py-0.5 text-xs"
      }`}
    >
      <span aria-hidden="true">{t.icon}</span>
      {t.label}
    </span>
  );
}

export function BarRow({ label, value, note, color = "bg-indigo-500", strong = false }) {
  return (
    <div>
      <div className="mb-1 flex justify-between text-sm">
        <span className={strong ? "font-semibold text-slate-900" : "text-slate-600"}>{label}</span>
        <span className="font-medium tabular-nums text-slate-900">
          {value.toFixed(3)}
          {note && <span className="ml-1 font-normal text-slate-400">{note}</span>}
        </span>
      </div>
      <div className="h-2.5 rounded-full bg-slate-100">
        <div className={`h-2.5 rounded-full ${color}`} style={{ width: `${value * 100}%` }} />
      </div>
    </div>
  );
}

export function Spinner() {
  return <span className="inline-block size-4 animate-spin rounded-full border-2 border-current border-r-transparent" />;
}
