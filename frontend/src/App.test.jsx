// PRD TEST-011: when the backend is unavailable, the UI shows a recoverable error state.
import { afterEach, expect, test, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import App from "./App";
import Experiments from "./pages/Experiments";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

test("experiment page handles an empty log", () => {
  render(<Experiments experiments={[]} model={{}} />);
  expect(screen.getByText("No experiments are available for this model.")).toBeTruthy();
});

test("experiment page handles logs without a final fit", () => {
  render(<Experiments experiments={[{ run_id: "baseline", model: "Baseline", type: "baseline", cv_f1_mean: 0.5 }]}
    model={{ search_config: { seed: 42 }, cv_config: { inner: "3", outer: "5" }, threshold: 0.5 }} />);
  expect(screen.getByText("No final configuration recorded")).toBeTruthy();
});

test("shows a retryable error when the API is down", async () => {
  const fetchMock = vi.spyOn(globalThis, "fetch").mockRejectedValue(new TypeError("Failed to fetch"));
  render(<App />);

  expect(await screen.findByText("Service unavailable")).toBeTruthy();
  expect(screen.getByText("API offline")).toBeTruthy();

  const calls = fetchMock.mock.calls.length;
  fireEvent.click(screen.getByRole("button", { name: "Retry" }));
  expect(fetchMock.mock.calls.length).toBeGreaterThan(calls);
});

test("warns when the API serves a different model than the page shows", async () => {
  const model = { model_version: "churn-rf-v1", model_name: "Random Forest" };
  const responses = {
    "/api/v1/model": model,
    "/api/v1/experiments": [],
    "/api/v1/model-card": { modelDetails: { hyperparameters: {} }, outOfScope: [], evaluation: {}, limitations: [] },
    "/api/v1/health": { status: "ok", modelVersion: "churn-rf-v2" },
  };
  vi.spyOn(globalThis, "fetch").mockImplementation(async (url) => ({
    ok: true,
    json: async () => responses[url],
  }));
  window.location.hash = "#model-card";
  render(<App />);

  expect(await screen.findByText(/Model metadata is out of date/)).toBeTruthy();
  expect(screen.getByRole("button", { name: "Reload data" })).toBeTruthy();
});

test("recovers automatically once the API comes back", async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  const card = { modelDetails: { hyperparameters: {} }, outOfScope: [], evaluation: {}, limitations: [] };
  const responses = {
    "/api/v1/model": { model_version: "churn-rf-v1" },
    "/api/v1/experiments": [],
    "/api/v1/model-card": card,
    "/api/v1/health": { status: "ok", modelVersion: "churn-rf-v1" },
  };
  const fetchMock = vi.spyOn(globalThis, "fetch").mockRejectedValue(new TypeError("Failed to fetch"));
  window.location.hash = "#model-card";
  render(<App />);
  expect(await screen.findByText("Service unavailable")).toBeTruthy();

  fetchMock.mockImplementation(async (url) => ({ ok: true, json: async () => responses[url] }));
  await vi.advanceTimersByTimeAsync(5000);

  expect(await screen.findByText("Intended use")).toBeTruthy();
  expect(screen.queryByText("Service unavailable")).toBeNull();
  vi.useRealTimers();
});
