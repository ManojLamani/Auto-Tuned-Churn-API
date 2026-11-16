import { Card } from "../ui";

const EVAL_LABELS = {
  baselineCvF1: "Baseline CV F1",
  nestedCvF1: "Nested CV F1",
  nestedCvF1Std: "Nested CV F1 std",
  testF1: "Test F1",
  testBaselineF1: "Test baseline F1",
  testPrecision: "Test precision",
  testRecall: "Test recall",
};

export default function ModelCard({ card }) {
  const d = card.modelDetails;
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-semibold text-slate-900">Model card</h2>
        <p className="mt-1 text-sm text-slate-500">
          {d.name} · {d.version} · trained {new Date(d.trainedAt).toLocaleString()}
        </p>
      </div>

      <div role="note" className="flex gap-3 rounded-2xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900">
        <span aria-hidden="true" className="text-lg leading-none">⚠</span>
        <p>
          <b>Decision support only.</b> {card.ethicalConsiderations}
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Intended use">
          <p className="text-sm leading-relaxed text-slate-700">{card.intendedUse}</p>
        </Card>
        <Card title="Out of scope">
          <List items={card.outOfScope} icon="✕" color="text-rose-500" />
        </Card>
        <Card title="Data">
          <p className="text-sm leading-relaxed text-slate-700">{card.data}</p>
        </Card>
        <Card title="Split and validation">
          <p className="text-sm leading-relaxed text-slate-700">{card.splitAndValidation}</p>
        </Card>
        <Card title="Evaluation">
          <dl className="grid grid-cols-2 gap-3 sm:grid-cols-3">
            {Object.entries(card.evaluation).map(([k, v]) => (
              <div key={k} className="rounded-lg bg-slate-50 px-3 py-2">
                <dt className="text-xs text-slate-500">{EVAL_LABELS[k] || k}</dt>
                <dd className="text-lg font-semibold tabular-nums text-slate-900">{v.toFixed(3)}</dd>
              </div>
            ))}
          </dl>
        </Card>
        <Card title="Hyperparameters" subtitle={`Operating threshold ${d.threshold}`}>
          <dl className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm">
            {Object.entries(d.hyperparameters).map(([k, v]) => (
              <div key={k} className="flex justify-between border-b border-slate-100 py-1">
                <dt className="font-mono text-xs text-slate-500">{k}</dt>
                <dd className="font-mono text-xs text-slate-900">{v === null ? "None" : String(v)}</dd>
              </div>
            ))}
          </dl>
        </Card>
      </div>

      <Card title="Limitations" subtitle="Read these before acting on a score">
        <List items={card.limitations} icon="!" color="text-amber-500" />
      </Card>
      <Card title="Monitoring">
        <p className="text-sm leading-relaxed text-slate-700">{card.monitoring}</p>
      </Card>
    </div>
  );
}

function List({ items, icon, color }) {
  return (
    <ul className="space-y-2">
      {items.map((item) => (
        <li key={item} className="flex gap-3 text-sm text-slate-700">
          <span aria-hidden="true" className={`font-bold ${color}`}>{icon}</span>
          {item}
        </li>
      ))}
    </ul>
  );
}
