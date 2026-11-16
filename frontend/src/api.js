const BASE = "/api/v1";

async function request(path, options) {
  let res;
  try {
    res = await fetch(BASE + path, options);
  } catch {
    throw new Error("Cannot reach the API. Is the backend running?");
  }
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    const err = new Error(body?.error?.message || `Request failed (${res.status})`);
    err.details = body?.error?.details;
    throw err;
  }
  return body;
}

export const api = {
  health: () => request("/health"),
  model: () => request("/model"),
  experiments: () => request("/experiments"),
  modelCard: () => request("/model-card"),
  predict: (customer) =>
    request("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(customer),
    }),
};
