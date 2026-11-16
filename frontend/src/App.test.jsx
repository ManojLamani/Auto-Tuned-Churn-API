// PRD TEST-011: when the backend is unavailable, the UI shows a recoverable error state.
import { afterEach, expect, test, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import App from "./App";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
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
