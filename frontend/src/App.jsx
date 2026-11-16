import { useCallback, useEffect, useState } from "react";
import { api } from "./api";
import { Spinner } from "./ui";
import Overview from "./pages/Overview";
import Score from "./pages/Score";
import Experiments from "./pages/Experiments";
import ModelCard from "./pages/ModelCard";

const PAGES = [
  { id: "overview", label: "Overview", icon: "M3 13h4v8H3zM10 8h4v13h-4zM17 3h4v18h-4z", component: Overview },
  { id: "score", label: "Score customer", icon: "M12 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8zm-8 9a8 8 0 0 1 16 0", component: Score },
  { id: "experiments", label: "Experiments", icon: "M9 3h6M10 3v6l-5 9a2 2 0 0 0 2 3h10a2 2 0 0 0 2-3l-5-9V3", component: Experiments },
  { id: "model-card", label: "Model card", icon: "M6 3h9l4 4v14H6zM14 3v5h5M9 13h7M9 17h7", component: ModelCard },
];

const pageFromHash = () => PAGES.find((p) => `#${p.id}` === window.location.hash)?.id || "overview";

export default function App() {
  const [page, setPage] = useState(pageFromHash);
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    setError(null);
    setData(null);
    try {
      const [model, experiments, card] = await Promise.all([api.model(), api.experiments(), api.modelCard()]);
      setData({ model, experiments, card });
    } catch (e) {
      setError(e.message);
    }
  }, []);

  useEffect(() => {
    load();
    const onHash = () => setPage(pageFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, [load]);

  const Page = PAGES.find((p) => p.id === page).component;

  return (
    <div className="min-h-screen lg:flex">
      <aside className="border-b border-slate-200 bg-white lg:fixed lg:inset-y-0 lg:w-64 lg:border-r lg:border-b-0">
        <div className="flex items-center gap-3 px-5 py-4 lg:py-6">
          <div className="grid size-9 place-items-center rounded-xl bg-indigo-600 text-sm font-bold text-white">AC</div>
          <div>
            <p className="text-sm font-semibold text-slate-900">Auto-Tuned Churn</p>
            <p className="text-xs text-slate-500">Retention decision support</p>
          </div>
        </div>
        <nav className="flex gap-1 overflow-x-auto px-3 pb-3 lg:flex-col lg:pb-0">
          {PAGES.map((p) => (
            <a
              key={p.id}
              href={`#${p.id}`}
              aria-current={page === p.id ? "page" : undefined}
              className={`flex shrink-0 items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition ${
                page === p.id ? "bg-indigo-50 text-indigo-700" : "text-slate-600 hover:bg-slate-100 hover:text-slate-900"
              }`}
            >
              <svg viewBox="0 0 24 24" className="size-5" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <path d={p.icon} />
              </svg>
              {p.label}
            </a>
          ))}
        </nav>
      </aside>

      <div className="flex-1 lg:pl-64">
        <header className="sticky top-0 z-10 flex items-center justify-between border-b border-slate-200 bg-white/80 px-6 py-3 backdrop-blur">
          <h1 className="text-base font-semibold text-slate-900">{PAGES.find((p) => p.id === page).label}</h1>
          <ApiStatus data={data} error={error} />
        </header>
        <main className="mx-auto max-w-7xl p-4 sm:p-6">
          {error ? (
            <ErrorState message={error} onRetry={load} />
          ) : !data ? (
            <div className="flex items-center justify-center gap-3 py-32 text-slate-500">
              <Spinner /> Loading model data…
            </div>
          ) : (
            <Page {...data} />
          )}
        </main>
      </div>
    </div>
  );
}

function ApiStatus({ data, error }) {
  const [dot, text] = error
    ? ["bg-rose-500", "API offline"]
    : data
      ? ["bg-emerald-500", `API online · ${data.model.model_version}`]
      : ["bg-slate-300", "Connecting…"];
  return (
    <span className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-medium text-slate-600">
      <span className={`size-2 rounded-full ${dot}`} />
      {text}
    </span>
  );
}

function ErrorState({ message, onRetry }) {
  return (
    <div role="alert" className="mx-auto mt-16 max-w-md rounded-2xl border border-rose-200 bg-white p-8 text-center shadow-sm">
      <div className="mx-auto mb-4 grid size-12 place-items-center rounded-full bg-rose-50 text-xl text-rose-600">!</div>
      <h2 className="text-lg font-semibold text-slate-900">Service unavailable</h2>
      <p className="mt-2 text-sm text-slate-600">{message}</p>
      <button
        onClick={onRetry}
        className="mt-6 rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white hover:bg-indigo-700"
      >
        Retry
      </button>
    </div>
  );
}
