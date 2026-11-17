import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  use: { baseURL: "http://127.0.0.1:18762", trace: "retain-on-failure" },
  webServer: [
    {
      command: `"${process.env.E2E_PYTHON || "python"}" -m uvicorn backend.app.main:app --host 127.0.0.1 --port 18761`,
      cwd: "..",
      url: "http://127.0.0.1:18761/api/v1/health",
      reuseExistingServer: false,
      timeout: 60000,
    },
    {
      command: "npm run dev -- --host 127.0.0.1 --port 18762 --strictPort",
      env: { API_URL: "http://127.0.0.1:18761" },
      url: "http://127.0.0.1:18762",
      reuseExistingServer: false,
    },
  ],
});
