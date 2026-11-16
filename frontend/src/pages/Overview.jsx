import { BarRow, Card, Kpi, TIERS, pct } from "../ui";

const METRICS = [
  ["precision", "Precision"],
  ["recall", "Recall"],
  ["f1", "F1"],
  ["roc_auc", "ROC-AUC"],
  ["pr_auc", "PR-AUC"],
];

export default function Overview({ model, experiments }) {
  const m = model.metrics;
  const gain = m.test.f1 - m.test_baseline.f1;
  const cvRuns = experiments.filter((r) => r.type !== "final_random_search");
  const best = Math.max(...cvRuns.map((r) => r.cv_f1_mean));

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-semibold text-slate-900">Model performance</h2>
        <p className="mt-1 text-sm text-slate-500">
          {model.model_name} · {model.model_version} · trained {new Date(model.training_timestamp).toLocaleDateString()} ·
          threshold {model.threshold}
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Kpi accent label="Tuned F1 (test)" value={m.test.f1.toFixed(3)} hint={`${model.model_name}, Random Search`} />
        <Kpi label="Baseline F1 (test)" value={m.test_baseline.f1.toFixed(3)} hint="Default Logistic Regression" />
        <Kpi
          label="F1 improvement"
          value={`+${gain.toFixed(3)}`}
          hint={`${pct(gain / m.test_baseline.f1)} better than baseline`}
        />
        <Kpi
          label="Nested CV F1"
          value={m.nested_cv_f1_mean.toFixed(3)}
          hint={`± ${m.nested_cv_f1_std.toFixed(3)} across ${model.cv_config.outer.match(/\d+/)} outer folds`}
        />
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Model comparison" subtitle="CV F1 on the training set; tuned models scored with nested CV">
          <div className="space-y-4">
            {cvRuns.map((r) => (
              <BarRow
                key={r.run_id}
                label={r.type === "baseline" ? r.model : `${r.model} (tuned)`}
                value={r.cv_f1_mean}
                note={`± ${r.cv_f1_std.toFixed(3)}`}
                strong={r.cv_f1_mean === best}
                color={r.cv_f1_mean === best ? "bg-indigo-600" : r.type === "baseline" ? "bg-slate-300" : "bg-indigo-300"}
              />
            ))}
          </div>
        </Card>

        <Card
          title="Baseline vs tuned on the test set"
          subtitle="Evaluated once, after all choices were fixed"
          action={
            <div className="flex gap-3 text-xs text-slate-500">
              <Legend color="bg-slate-300" label="Baseline" />
              <Legend color="bg-indigo-600" label="Tuned" />
            </div>
          }
        >
          <div className="space-y-3">
            {METRICS.map(([key, label]) => (
              <div key={key} className="grid grid-cols-[72px_1fr] items-center gap-3 text-sm">
                <span className="text-slate-600">{label}</span>
                <div className="space-y-1">
                  <MiniBar value={m.test_baseline[key]} color="bg-slate-300" />
                  <MiniBar value={m.test[key]} color="bg-indigo-600" />
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <ConfusionMatrix cm={m.test.confusion_matrix} threshold={model.threshold} />
        <TierDistribution tiers={m.test_risk_tiers} riskTiers={model.risk_tiers} />
      </div>
    </div>
  );
}

function Legend({ color, label }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className={`size-2.5 rounded-sm ${color}`} />
      {label}
    </span>
  );
}

function MiniBar({ value, color }) {
  return (
    <div className="flex items-center gap-2">
      <div className="h-2 flex-1 rounded-full bg-slate-100">
        <div className={`h-2 rounded-full ${color}`} style={{ width: `${value * 100}%` }} />
      </div>
      <span className="w-11 text-right text-xs tabular-nums text-slate-700">{value.toFixed(3)}</span>
    </div>
  );
}

function ConfusionMatrix({ cm, threshold }) {
  const [[tn, fp], [fn, tp]] = cm;
  const max = Math.max(tn, fp, fn, tp);
  const cell = (value, label, good) => (
    <div
      className={`rounded-xl p-4 text-center ${good ? "text-indigo-950" : "text-rose-950"}`}
      style={{ backgroundColor: good ? `rgba(79,70,229,${0.08 + (value / max) * 0.35})` : `rgba(244,63,94,${0.08 + (value / max) * 0.35})` }}
    >
      <p className="text-2xl font-semibold tabular-nums">{value}</p>
      <p className="text-xs font-medium opacity-80">{label}</p>
    </div>
  );

  return (
    <Card title="Confusion matrix" subtitle={`Test set (${tn + fp + fn + tp} customers), threshold ${threshold}`}>
      <div className="grid grid-cols-[auto_1fr_1fr] gap-2 text-xs text-slate-500">
        <span />
        <span className="text-center font-medium">Predicted: stays</span>
        <span className="text-center font-medium">Predicted: churns</span>
        <span className="self-center font-medium [writing-mode:vertical-rl] rotate-180">Actual: stays</span>
        {cell(tn, "Correctly not flagged", true)}
        {cell(fp, "Extra outreach", false)}
        <span className="self-center font-medium [writing-mode:vertical-rl] rotate-180">Actual: churns</span>
        {cell(fn, "Missed churners", false)}
        {cell(tp, "Churners caught", true)}
      </div>
      <p className="mt-4 text-sm text-slate-600">
        Catches <b>{tp}</b> of {tp + fn} churners ({pct(tp / (tp + fn), 0)}). {pct(fp / (tp + fp), 0)} of flagged customers
        would have stayed: that is the outreach cost of high recall.
      </p>
    </Card>
  );
}

function TierDistribution({ tiers, riskTiers }) {
  const total = Object.values(tiers).reduce((a, b) => a + b, 0);
  const rule = {
    HIGH: `≥ ${pct(riskTiers.HIGH, 0)}`,
    MEDIUM: `${pct(riskTiers.MEDIUM, 0)} – ${pct(riskTiers.HIGH, 0)}`,
    LOW: `< ${pct(riskTiers.MEDIUM, 0)}`,
  };
  return (
    <Card title="Risk tier distribution" subtitle="Share of test-set customers in each tier">
      <div className="flex h-4 overflow-hidden rounded-full">
        {Object.entries(tiers).map(([tier, n]) => (
          <div key={tier} className={TIERS[tier].bar} style={{ width: `${(n / total) * 100}%` }} title={`${tier}: ${n}`} />
        ))}
      </div>
      <ul className="mt-5 divide-y divide-slate-100">
        {Object.entries(tiers).map(([tier, n]) => (
          <li key={tier} className="flex items-center justify-between py-3 text-sm">
            <span className="flex items-center gap-2">
              <span className={`size-2.5 rounded-full ${TIERS[tier].bar}`} />
              <span className="font-medium text-slate-900">{TIERS[tier].label}</span>
              <span className="text-slate-400">probability {rule[tier]}</span>
            </span>
            <span className="tabular-nums text-slate-700">
              {n} <span className="text-slate-400">({pct(n / total, 0)})</span>
            </span>
          </li>
        ))}
      </ul>
    </Card>
  );
}
