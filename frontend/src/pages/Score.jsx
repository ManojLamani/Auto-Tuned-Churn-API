import { useState } from "react";
import { api } from "../api";
import { Card, Spinner, TIERS, TierBadge, pct } from "../ui";

const GROUPS = [
  ["Customer", ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure"]],
  ["Services", ["PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]],
  ["Billing", ["Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"]],
];

const LABELS = {
  gender: "Gender", SeniorCitizen: "Senior citizen", Partner: "Has partner", Dependents: "Has dependents",
  tenure: "Tenure (months)", PhoneService: "Phone service", MultipleLines: "Multiple lines",
  InternetService: "Internet service", OnlineSecurity: "Online security", OnlineBackup: "Online backup",
  DeviceProtection: "Device protection", TechSupport: "Tech support", StreamingTV: "Streaming TV",
  StreamingMovies: "Streaming movies", Contract: "Contract", PaperlessBilling: "Paperless billing",
  PaymentMethod: "Payment method", MonthlyCharges: "Monthly charges ($)", TotalCharges: "Total charges ($)",
};

const EXAMPLES = {
  "High-risk example": {
    gender: "Female", SeniorCitizen: "1", Partner: "No", Dependents: "No", tenure: "2", PhoneService: "Yes",
    MultipleLines: "No", InternetService: "Fiber optic", OnlineSecurity: "No", OnlineBackup: "No",
    DeviceProtection: "No", TechSupport: "No", StreamingTV: "Yes", StreamingMovies: "Yes",
    Contract: "Month-to-month", PaperlessBilling: "Yes", PaymentMethod: "Electronic check",
    MonthlyCharges: "95.50", TotalCharges: "190.00",
  },
  "Low-risk example": {
    gender: "Male", SeniorCitizen: "0", Partner: "Yes", Dependents: "Yes", tenure: "60", PhoneService: "Yes",
    MultipleLines: "Yes", InternetService: "DSL", OnlineSecurity: "Yes", OnlineBackup: "Yes",
    DeviceProtection: "Yes", TechSupport: "Yes", StreamingTV: "No", StreamingMovies: "No",
    Contract: "Two year", PaperlessBilling: "No", PaymentMethod: "Bank transfer (automatic)",
    MonthlyCharges: "65.00", TotalCharges: "3900.00",
  },
};

const EMPTY = Object.fromEntries(Object.keys(LABELS).map((k) => [k, ""]));
const inputClass =
  "mt-1 block w-full rounded-lg border bg-white px-3 py-2 text-sm text-slate-900 shadow-xs focus:outline-none focus:ring-2 focus:ring-indigo-500";

export default function Score({ model }) {
  const schema = model.feature_schema;
  const [form, setForm] = useState(EMPTY);
  const [errors, setErrors] = useState({});
  const [state, setState] = useState({ status: "idle" });
  const [history, setHistory] = useState([]);

  const options = (field) =>
    field === "SeniorCitizen" ? [["0", "No"], ["1", "Yes"]] : schema.categorical[field]?.map((v) => [v, v]);

  const set = (field, value) => {
    setForm((f) => ({ ...f, [field]: value }));
    setErrors((e) => ({ ...e, [field]: undefined }));
  };

  const validate = () => {
    const found = {};
    for (const [field, value] of Object.entries(form)) {
      if (value === "") found[field] = "Required";
      else if (!options(field) && (isNaN(value) || Number(value) < 0)) found[field] = "Enter a number ≥ 0";
    }
    setErrors(found);
    return Object.keys(found).length === 0;
  };

  const submit = async (e) => {
    e.preventDefault();
    if (!validate()) return;
    const customer = Object.fromEntries(
      Object.entries(form).map(([k, v]) => [k, options(k) && k !== "SeniorCitizen" ? v : Number(v)]),
    );
    setState({ status: "loading" });
    try {
      const result = await api.predict(customer);
      setState({ status: "done", result });
      setHistory((h) => [result, ...h].slice(0, 5));
    } catch (err) {
      if (err.details) setErrors(Object.fromEntries(err.details.map((d) => [d.field, d.message])));
      setState({ status: "error", message: err.message });
    }
  };

  const reset = () => {
    setForm(EMPTY);
    setErrors({});
    setState({ status: "idle" });
  };

  return (
    <div className="grid gap-6 lg:grid-cols-[1fr_380px]">
      <form onSubmit={submit} noValidate className="space-y-6">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-2xl font-semibold text-slate-900">Score a customer</h2>
            <p className="mt-1 text-sm text-slate-500">All fields are required. Inputs are not stored.</p>
          </div>
          <div className="flex gap-2">
            {Object.keys(EXAMPLES).map((name) => (
              <button
                key={name}
                type="button"
                onClick={() => { setForm(EXAMPLES[name]); setErrors({}); }}
                className="rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-50"
              >
                {name}
              </button>
            ))}
          </div>
        </div>

        {GROUPS.map(([group, fields]) => (
          <Card key={group} title={group}>
            <fieldset className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
              <legend className="sr-only">{group}</legend>
              {fields.map((field) => (
                <label key={field} className="block text-sm font-medium text-slate-700">
                  {LABELS[field]}
                  {options(field) ? (
                    <select
                      value={form[field]}
                      onChange={(e) => set(field, e.target.value)}
                      aria-invalid={!!errors[field]}
                      className={`${inputClass} ${errors[field] ? "border-rose-400" : "border-slate-300"}`}
                    >
                      <option value="">Select…</option>
                      {options(field).map(([value, text]) => (
                        <option key={value} value={value}>{text}</option>
                      ))}
                    </select>
                  ) : (
                    <input
                      type="number"
                      min="0"
                      step={field === "tenure" ? "1" : "0.01"}
                      value={form[field]}
                      onChange={(e) => set(field, e.target.value)}
                      aria-invalid={!!errors[field]}
                      className={`${inputClass} ${errors[field] ? "border-rose-400" : "border-slate-300"}`}
                    />
                  )}
                  {errors[field] && <span className="mt-1 block text-xs font-normal text-rose-600">{errors[field]}</span>}
                </label>
              ))}
            </fieldset>
          </Card>
        ))}

        <div className="flex gap-3">
          <button
            type="submit"
            disabled={state.status === "loading"}
            className="inline-flex items-center gap-2 rounded-lg bg-indigo-600 px-5 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-indigo-700 disabled:opacity-60"
          >
            {state.status === "loading" && <Spinner />} Score customer
          </button>
          <button
            type="button"
            onClick={reset}
            className="rounded-lg border border-slate-300 bg-white px-5 py-2.5 text-sm font-semibold text-slate-700 hover:bg-slate-50"
          >
            Reset
          </button>
        </div>
      </form>

      <div className="space-y-6 lg:sticky lg:top-20 lg:self-start">
        <Result state={state} model={model} />
        {history.length > 0 && (
          <Card title="This session" subtitle="Last 5 scores (kept in this tab only)">
            <ul className="divide-y divide-slate-100">
              {history.map((h) => (
                <li key={h.predictionId} className="flex items-center justify-between py-2 text-sm">
                  <span className="tabular-nums text-slate-500">{new Date(h.timestamp).toLocaleTimeString()}</span>
                  <span className="font-semibold tabular-nums">{pct(h.churnProbability)}</span>
                  <TierBadge tier={h.riskTier} />
                </li>
              ))}
            </ul>
          </Card>
        )}
      </div>
    </div>
  );
}

function Result({ state, model }) {
  if (state.status === "idle" || state.status === "loading") {
    return (
      <Card title="Prediction">
        <div className="py-10 text-center text-sm text-slate-500">
          {state.status === "loading" ? (
            <span className="inline-flex items-center gap-2"><Spinner /> Scoring…</span>
          ) : (
            "Fill in the form or load an example, then press Score customer."
          )}
        </div>
      </Card>
    );
  }
  if (state.status === "error") {
    return (
      <Card title="Prediction">
        <div role="alert" className="rounded-lg bg-rose-50 p-4 text-sm text-rose-700">
          <p className="font-semibold">Could not score this customer</p>
          <p className="mt-1">{state.message}</p>
        </div>
      </Card>
    );
  }

  const r = state.result;
  const { HIGH, MEDIUM } = model.risk_tiers;
  const advice = {
    HIGH: "Above the operating threshold: recommended for proactive outreach.",
    MEDIUM: "Below the threshold but elevated: add to the watch list.",
    LOW: "Well below the threshold: no action suggested.",
  }[r.riskTier];

  return (
    <Card title="Prediction" action={<TierBadge tier={r.riskTier} large />}>
      <p className="text-xs font-medium uppercase tracking-wide text-slate-500">Estimated churn probability</p>
      <p className="mt-1 text-5xl font-semibold tabular-nums text-slate-900">{pct(r.churnProbability)}</p>

      <div className="relative mt-6">
        <div className="flex h-3 overflow-hidden rounded-full">
          <div className="bg-emerald-200" style={{ width: `${MEDIUM * 100}%` }} />
          <div className="bg-amber-200" style={{ width: `${(HIGH - MEDIUM) * 100}%` }} />
          <div className="flex-1 bg-rose-200" />
        </div>
        <div
          className={`absolute -top-1 size-5 -translate-x-1/2 rounded-full border-2 border-white shadow ${TIERS[r.riskTier].bar}`}
          style={{ left: `${r.churnProbability * 100}%` }}
        />
        <div className="absolute -bottom-5 h-6 border-l-2 border-dashed border-slate-500" style={{ left: `${HIGH * 100}%` }} />
      </div>
      <div className="mt-6 flex justify-between text-xs text-slate-500">
        <span>0%</span>
        <span>threshold {pct(HIGH, 0)}</span>
        <span>100%</span>
      </div>

      <p className="mt-5 rounded-lg bg-slate-50 p-3 text-sm text-slate-700">{advice}</p>

      <dl className="mt-5 grid grid-cols-2 gap-3 text-xs">
        <div><dt className="text-slate-500">Model</dt><dd className="font-medium text-slate-900">{r.modelVersion}</dd></div>
        <div><dt className="text-slate-500">Preprocessing</dt><dd className="font-medium text-slate-900">{r.preprocessingVersion}</dd></div>
        <div><dt className="text-slate-500">Scored at</dt><dd className="font-medium text-slate-900">{new Date(r.timestamp).toLocaleString()}</dd></div>
        <div><dt className="text-slate-500">Prediction ID</dt><dd className="font-mono text-slate-900">{r.predictionId}</dd></div>
      </dl>

      {r.modelVersion !== model.model_version && (
        <p role="alert" className="mt-4 rounded-lg bg-amber-50 p-3 text-xs text-amber-900">
          Scored by {r.modelVersion}, but this page shows metrics for {model.model_version}. Reload the page.
        </p>
      )}

      <p className="mt-5 border-t border-slate-100 pt-4 text-xs leading-relaxed text-slate-500">
        This is a statistical estimate to help prioritise outreach, not a certainty. Use it with human judgement and
        never to penalise or deny service to a customer.
      </p>
    </Card>
  );
}
