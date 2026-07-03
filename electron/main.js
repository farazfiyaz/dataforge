/**
 * DataForge — Electron Main Process
 * Copyright (C) 2026 Mohammed Farazuddin
 * License: AGPL-3.0
 *
 * Responsibilities:
 *  1. Spawn the FastAPI backend (Python) as a child process
 *  2. Wait until the backend is ready (polls /health)
 *  3. Open a native BrowserWindow pointing to http://localhost:8000
 *  4. Kill the backend cleanly on app quit
 */

const { app, BrowserWindow, shell, Menu } = require("electron");
const path  = require("path");
const http  = require("http");
const { spawn } = require("child_process");

const PORT       = 8000;
const BACKEND_URL = `http://localhost:${PORT}`;
const MAX_WAIT_MS = 30_000;   // 30 seconds before giving up
const POLL_MS     = 500;

let mainWindow   = null;
let backendProc  = null;

// ── 1. Start FastAPI backend ──────────────────────────────────────────────────
function startBackend() {
  // Resolve paths — works both in dev (project root) and packaged (extraResources)
  const isPackaged = app.isPackaged;
  const root = isPackaged
    ? path.join(process.resourcesPath)
    : path.join(__dirname, "..");

  const backendDir = path.join(root, "backend");
  const venvPython = process.platform === "win32"
    ? path.join(backendDir, "venv", "Scripts", "python.exe")
    : path.join(backendDir, "venv", "bin", "python");

  const pythonBin = require("fs").existsSync(venvPython) ? venvPython : "python";
  const appScript = path.join(backendDir, "app.py");

  console.log("[DataForge] Starting backend:", pythonBin, appScript);

  backendProc = spawn(pythonBin, [appScript], {
    cwd: backendDir,
    env: { ...process.env, PYTHONUNBUFFERED: "1" },
    stdio: ["ignore", "pipe", "pipe"],
  });

  backendProc.stdout.on("data", d => console.log("[backend]", d.toString().trim()));
  backendProc.stderr.on("data", d => console.error("[backend]", d.toString().trim()));
  backendProc.on("exit", (code) => {
    console.log("[DataForge] Backend exited with code", code);
  });
}

// ── 2. Poll until backend is ready ────────────────────────────────────────────
function waitForBackend() {
  return new Promise((resolve, reject) => {
    const start = Date.now();
    const check = () => {
      http.get(`${BACKEND_URL}/health`, res => {
        if (res.statusCode === 200) return resolve();
        retry();
      }).on("error", retry);
    };
    const retry = () => {
      if (Date.now() - start > MAX_WAIT_MS) return reject(new Error("Backend did not start in time."));
      setTimeout(check, POLL_MS);
    };
    check();
  });
}

// ── 3. Create the window ──────────────────────────────────────────────────────
function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 800,
    minWidth: 900,
    minHeight: 600,
    title: "DataForge",
    backgroundColor: "#0f1117",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
    // Remove default menu bar (feels more like an app)
    autoHideMenuBar: true,
  });

  mainWindow.loadURL(BACKEND_URL);

  // Open external links in the system browser, not inside the app
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: "deny" };
  });

  mainWindow.on("closed", () => { mainWindow = null; });
}

// Build a minimal app menu (File / Edit / View)
function buildMenu() {
  const template = [
    {
      label: "File",
      submenu: [
        { label: "Quit DataForge", accelerator: "CmdOrCtrl+Q", click: () => app.quit() },
      ],
    },
    {
      label: "Edit",
      submenu: [
        { role: "undo" }, { role: "redo" }, { type: "separator" },
        { role: "cut" }, { role: "copy" }, { role: "paste" },
      ],
    },
    {
      label: "View",
      submenu: [
        { role: "reload" },
        { role: "forceReload" },
        { type: "separator" },
        { role: "zoomIn" }, { role: "zoomOut" }, { role: "resetZoom" },
        { type: "separator" },
        { role: "togglefullscreen" },
        { label: "Developer Tools", accelerator: "CmdOrCtrl+Shift+I", click: () => mainWindow?.webContents.openDevTools() },
      ],
    },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

// ── 4. App lifecycle ──────────────────────────────────────────────────────────
app.whenReady().then(async () => {
  buildMenu();
  startBackend();

  try {
    await waitForBackend();
    createWindow();
  } catch (err) {
    console.error("[DataForge] Fatal:", err.message);
    app.quit();
  }

  app.on("activate", () => {
    // macOS: re-open window when dock icon is clicked
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on("will-quit", () => {
  if (backendProc && !backendProc.killed) {
    console.log("[DataForge] Shutting down backend...");
    backendProc.kill("SIGTERM");
  }
});
