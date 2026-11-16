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
