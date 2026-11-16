import { Card } from "../ui";

const TYPE = {
  baseline: ["Baseline", "bg-slate-100 text-slate-700"],
  nested_cv_random_search: ["Nested CV + Random Search", "bg-indigo-50 text-indigo-700"],
  final_random_search: ["Final fit", "bg-emerald-50 text-emerald-700"],
};

export default function Experiments({ experiments, model }) {
  const final = experiments.find((r) => r.type === "final_random_search");
  const nested = experiments.filter((r) => r.type === "nested_cv_random_search");

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-semibold text-slate-900">Experiment log</h2>
        <p className="mt-1 text-sm text-slate-500">
          Dataset {model.dataset_version} · split seed {model.search_config.seed} · scoring F1 · inner{" "}
          {model.cv_config.inner}, outer {model.cv_config.outer}
        </p>
      </div>

      <Card title="Runs" subtitle="Every run records its configuration and scores">
        <div className="-mx-5 overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="border-b border-slate-200 text-xs uppercase tracking-wide text-slate-500">
              <tr>
                <th className="px-5 py-2 font-medium">Run</th>
                <th className="px-5 py-2 font-medium">Model</th>
                <th className="px-5 py-2 font-medium">Type</th>
                <th className="px-5 py-2 text-right font-medium">CV F1</th>
                <th className="px-5 py-2 text-right font-medium">Runtime</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {experiments.map((r) => (
                <tr key={r.run_id} className="hover:bg-slate-50">
                  <td className="px-5 py-3 font-mono text-xs text-slate-600">{r.run_id}</td>
                  <td className="px-5 py-3 font-medium text-slate-900">{r.model}</td>
                  <td className="px-5 py-3">
                    <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${TYPE[r.type][1]}`}>{TYPE[r.type][0]}</span>
                  </td>
                  <td className="px-5 py-3 text-right tabular-nums">
                    {r.cv_f1_mean.toFixed(3)}
                    {r.cv_f1_std != null && <span className="text-slate-400"> ± {r.cv_f1_std.toFixed(3)}</span>}
                  </td>
                  <td className="px-5 py-3 text-right tabular-nums text-slate-500">
                    {r.runtime_sec != null ? `${r.runtime_sec}s` : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Outer-fold F1 (nested CV)" subtitle="Each dot is one outer fold the search never saw">
          <div className="space-y-5">
            {nested.map((r) => (
              <div key={r.run_id}>
                <div className="mb-2 flex justify-between text-sm">
                  <span className="font-medium text-slate-900">{r.model}</span>
                  <span className="tabular-nums text-slate-500">mean {r.cv_f1_mean.toFixed(3)}</span>
                </div>
                <FoldStrip folds={r.outer_fold_f1} mean={r.cv_f1_mean} />
              </div>
            ))}
            <div className="flex justify-between text-xs text-slate-400">
              <span>0.55</span>
              <span>0.60</span>
              <span>0.65</span>
            </div>
          </div>
        </Card>

        <Card title="Selected configuration" subtitle={`${final.model} · best of ${final.n_iter} sampled configurations`}>
          <dl className="grid grid-cols-2 gap-3">
            {Object.entries(final.best_params).map(([k, v]) => (
              <div key={k} className="rounded-lg bg-slate-50 px-3 py-2">
                <dt className="text-xs text-slate-500">{k}</dt>
                <dd className="font-mono text-sm text-slate-900">{v === null ? "None" : String(v)}</dd>
              </div>
            ))}
            <div className="rounded-lg bg-indigo-50 px-3 py-2">
              <dt className="text-xs text-indigo-600">threshold</dt>
              <dd className="font-mono text-sm text-indigo-900">{final.threshold}</dd>
            </div>
          </dl>
        </Card>
      </div>

      <Card title="Search spaces" subtitle="Distributions sampled by Random Search (Bergstra & Bengio, 2012)">
        <div className="grid gap-6 md:grid-cols-2">
          {nested.map((r) => (
            <div key={r.run_id}>
              <p className="mb-2 text-sm font-medium text-slate-900">{r.model}</p>
              <ul className="space-y-1.5 text-sm">
                {Object.entries(r.search_space).map(([k, v]) => (
                  <li key={k} className="flex justify-between gap-4">
                    <span className="font-mono text-xs text-slate-600">{k}</span>
                    <span className="text-right font-mono text-xs text-slate-900">{Array.isArray(v) ? v.join(", ") : v}</span>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

function FoldStrip({ folds, mean }) {
  const pos = (v) => `${Math.min(Math.max((v - 0.55) / 0.1, 0), 1) * 100}%`;
  return (
    <div className="relative h-6 rounded-full bg-slate-100">
      <div className="absolute inset-y-0 w-0.5 bg-indigo-300" style={{ left: pos(mean) }} />
      {folds.map((f, i) => (
        <span
          key={i}
          title={`Fold ${i + 1}: ${f}`}
          className="absolute top-1/2 size-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white bg-indigo-600 shadow"
          style={{ left: pos(f) }}
        />
      ))}
    </div>
  );
}
