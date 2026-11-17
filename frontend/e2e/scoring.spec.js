import { test, expect } from "@playwright/test";

test("validates, scores through the real API, and retains result history across navigation", async ({ page }) => {
  await page.goto("/#score");
  await expect(page.getByRole("heading", { name: "Score a customer" })).toBeVisible();
  await page.getByRole("button", { name: "Score customer", exact: true }).click();
  await expect(page.getByText("Required", { exact: true })).toHaveCount(19);
  await page.getByRole("button", { name: "High-risk example" }).click();
  const responsePromise = page.waitForResponse((r) => r.url().endsWith("/api/v1/predict") && r.status() === 200);
  await page.getByRole("button", { name: "Score customer", exact: true }).click();
  const result = await (await responsePromise).json();
  await expect(page.getByText("Estimated churn probability")).toBeVisible();
  await expect(page.getByText(`${(result.churnProbability * 100).toFixed(1)}%`, { exact: true }).first()).toBeVisible();
  await expect(page.getByText(result.predictionId, { exact: true })).toBeVisible();
  await expect(page.getByText(result.thresholdVersion, { exact: true })).toBeVisible();
  await page.getByRole("link", { name: "Experiments" }).click();
  await expect(page.getByRole("heading", { name: "Experiment log" })).toBeVisible();
  await page.getByRole("link", { name: "Score customer", exact: true }).click();
  await expect(page.getByRole("heading", { name: "This session" })).toBeVisible();
  await page.getByRole("button", { name: "Reset", exact: true }).focus();
  await page.keyboard.press("Enter");
  await expect(page.getByLabel("Tenure (months)")).toHaveValue("");
});

test("shows API validation and server errors without losing the form", async ({ page }) => {
  await page.goto("/#score");
  await page.getByRole("button", { name: "High-risk example" }).click();
  await page.getByLabel("Tenure (months)").fill("121");
  await page.getByRole("button", { name: "Score customer", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText("Invalid customer data");
  await expect(page.getByLabel("Tenure (months)")).toHaveAttribute("aria-invalid", "true");
  await page.getByLabel("Tenure (months)").fill("2");
  await page.route("**/api/v1/predict", (route) => route.fulfill({
    status: 503, contentType: "application/json",
    body: JSON.stringify({ error: { code: "MODEL_UNAVAILABLE", message: "Model is not loaded" } }),
  }));
  await page.getByRole("button", { name: "Score customer", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText("Model is not loaded");
  await expect(page.getByLabel("Tenure (months)")).toHaveValue("2");
});

test("scoring stays usable at a mobile viewport", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/#score");
  await page.getByRole("button", { name: "Low-risk example" }).click();
  await page.getByRole("button", { name: "Score customer", exact: true }).click();
  await expect(page.getByText("Estimated churn probability")).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
});
