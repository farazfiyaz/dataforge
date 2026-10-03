/**
 * DataForge — Electron Main Process
 * Copyright (C) 2026 Mohammed Farazuddin
 * License: AGPL-3.0
 *
 * Responsibilities:
 *  1. Spawn Ollama (if not already running) as a child process
 *  2. Spawn the FastAPI backend (Python) as a child process
 *  3. Wait until the backend is ready (polls /health)
 *  4. Open a native BrowserWindow on the backend (a free port, signed in with a per-launch token)
 *  5. Kill the backend + any Ollama instance we started, cleanly, on app quit
 */

const { app, BrowserWindow, shell, Menu, nativeTheme, session } = require("electron");
const path   = require("path");
const http   = require("http");
const net    = require("net");
const os     = require("os");
const crypto = require("crypto");
const { spawn } = require("child_process");

// Set in startBackend: a free port and a secret, new every launch. The backend
// refuses /api requests without the secret (other programs on this computer
// don't have it), and proves it knows it on /health, so a program squatting on
// the port can't pose as DataForge.
let PORT        = 0;
let BACKEND_URL = "";
const TOKEN     = crypto.randomBytes(32).toString("base64url");
const OLLAMA_URL  = "http://localhost:11434";
const MAX_WAIT_MS = 30_000;   // 30 seconds before giving up
const POLL_MS     = 500;

// Pick a model size that fits the machine's RAM, so a low-end laptop isn't
// handed a 7B model it can barely load. Every tier is a qwen2.5-coder variant
// (same family/behavior, Ollama supports tool calling on all of them) — only
// the size changes. Names must be real Ollama tags: `ollama pull <name>`.
const MODEL_TIERS = [
  { minGB: 12, model: "qwen2.5-coder:7b" },
  { minGB: 6,  model: "qwen2.5-coder:3b" },
  { minGB: 0,  model: "qwen2.5-coder:1.5b" },
];
function pickModel() {
  const totalGB = os.totalmem() / (1024 ** 3);
  const tier = MODEL_TIERS.find(t => totalGB >= t.minGB);
  console.log(`[DataForge] Detected ~${totalGB.toFixed(1)}GB RAM — selecting model ${tier.model}.`);
  return tier.model;
}
const MODEL = pickModel();   // also passed to the backend as DATAFORGE_MODEL (see startBackend)

let mainWindow   = null;
let backendProc  = null;
let ollamaProc   = null;
let ollamaStartedByUs = false;
let lastModelStatus = { state: "checking", model: MODEL };

// Tell the renderer (and remember for when the window loads later) what's
// happening with the model — so a first-time multi-minute download doesn't
// just look like the app hanging.
function broadcastModelStatus(status) {
  lastModelStatus = status;
  mainWindow?.webContents.send("df:model-status", status);
}

// ── 0. Start Ollama (skip if the user already has it running) ────────────────
function isOllamaRunning() {
  return new Promise(resolve => {
    http.get(`${OLLAMA_URL}/api/version`, res => resolve(res.statusCode === 200))
        .on("error", () => resolve(false));
  });
}

async function startOllama() {
  if (await isOllamaRunning()) {
    console.log("[DataForge] Ollama is already running — leaving it alone.");
    return;
  }

  console.log("[DataForge] Starting Ollama...");
  try {
    ollamaProc = spawn("ollama", ["serve"], {
      env: { ...process.env },
      stdio: ["ignore", "pipe", "pipe"],
    });
    ollamaStartedByUs = true;

    ollamaProc.stdout.on("data", d => console.log("[ollama]", d.toString().trim()));
    ollamaProc.stderr.on("data", d => console.error("[ollama]", d.toString().trim()));
    ollamaProc.on("error", err => {
      console.error("[DataForge] Failed to start Ollama (is it installed and on PATH?):", err.message);
      ollamaProc = null;
    });
    ollamaProc.on("exit", code => console.log("[DataForge] Ollama exited with code", code));
  } catch (err) {
    console.error("[DataForge] Could not spawn Ollama:", err.message);
  }
}

// Poll Ollama's version endpoint; non-fatal if it never comes up — chat
// features will just fail until the user starts Ollama themselves.
function waitForOllama() {
  return new Promise(resolve => {
    const start = Date.now();
    const check = () => {
      http.get(`${OLLAMA_URL}/api/version`, res => {
        if (res.statusCode === 200) return resolve(true);
        retry();
      }).on("error", retry);
    };
    const retry = () => {
      if (Date.now() - start > MAX_WAIT_MS) {
        console.warn("[DataForge] Ollama did not respond in time — continuing without waiting further.");
        return resolve(false);
      }
      setTimeout(check, POLL_MS);
    };
    check();
  });
}

// Load the model into memory right away (Ollama otherwise only loads it on the
// first real chat request, which makes that first question feel like it hung).
// Fire-and-forget: if it fails, the first real request just pays the load cost itself.
function warmModel() {
  const body = JSON.stringify({ model: MODEL, prompt: "", keep_alive: "30m" });
  const req = http.request(`${OLLAMA_URL}/api/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "Content-Length": Buffer.byteLength(body) },
  }, res => {
    res.on("data", () => {});
    res.on("end", () => console.log(`[DataForge] Model ${MODEL} warmed up (status ${res.statusCode}).`));
  });
  req.on("error", err => console.warn("[DataForge] Model warm-up failed (will load on first use instead):", err.message));
  req.write(body);
  req.end();
}

function isModelInstalled(model) {
  return new Promise(resolve => {
    http.get(`${OLLAMA_URL}/api/tags`, res => {
      let raw = "";
      res.on("data", d => raw += d);
      res.on("end", () => {
        try {
          const models = JSON.parse(raw).models || [];
          resolve(models.some(m => m.name === model));
        } catch {
          resolve(false);
        }
      });
    }).on("error", () => resolve(false));
  });
}

// Download the model via the Ollama CLI, relaying progress lines so the UI
// can show something better than a frozen screen on a first run.
function pullModel(model) {
  return new Promise(resolve => {
    console.log(`[DataForge] Model ${model} not installed — pulling (first run only)...`);
    const proc = spawn("ollama", ["pull", model], { stdio: ["ignore", "pipe", "pipe"] });
    const relay = d => {
      const text = d.toString().trim();
      if (text) {
        console.log("[ollama pull]", text);
        broadcastModelStatus({ state: "pulling", model, detail: text.split("\n").pop() });
      }
    };
    proc.stdout.on("data", relay);
    proc.stderr.on("data", relay);
    proc.on("error", err => {
      console.error("[DataForge] Could not run `ollama pull`:", err.message);
      resolve(false);
    });
    proc.on("exit", code => resolve(code === 0));
  });
}

// Make sure the picked model is actually available, downloading it if this
// is the first time this machine has run DataForge, then warm it into memory.
async function ensureModelReady() {
  broadcastModelStatus({ state: "checking", model: MODEL });
  let ok = await isModelInstalled(MODEL);
  if (!ok) {
    broadcastModelStatus({ state: "pulling", model: MODEL });
    ok = await pullModel(MODEL);
  }
  broadcastModelStatus({ state: ok ? "ready" : "error", model: MODEL });
  if (ok) warmModel();
}

// ── 1. Start FastAPI backend ──────────────────────────────────────────────────
// Ask the OS for a port nobody is using right now
function getFreePort() {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.unref();
    srv.on("error", reject);
    srv.listen(0, "127.0.0.1", () => {
      const { port } = srv.address();
      srv.close(() => resolve(port));
    });
  });
}

async function startBackend() {
  PORT = await getFreePort();
  BACKEND_URL = `http://127.0.0.1:${PORT}`;
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
    env: {
      ...process.env, PYTHONUNBUFFERED: "1", DATAFORGE_MODEL: MODEL,
      DATAFORGE_PORT: String(PORT), DATAFORGE_TOKEN: TOKEN,
    },
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
      // Only our backend knows TOKEN, so only it can answer the challenge
      const nonce = crypto.randomBytes(16).toString("hex");
      const expected = crypto.createHmac("sha256", TOKEN).update(nonce).digest("hex");
      http.get(`${BACKEND_URL}/health?nonce=${nonce}`, res => {
        let raw = "";
        res.on("data", d => raw += d);
        res.on("end", () => {
          let proof = "";
          try { proof = JSON.parse(raw).proof || ""; } catch {}
          if (res.statusCode === 200 && proof.length === expected.length &&
              crypto.timingSafeEqual(Buffer.from(proof), Buffer.from(expected))) return resolve();
          if (res.statusCode === 200) console.error("[DataForge] Something else answered on the backend port; waiting for ours.");
          retry();
        });
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
async function createWindow() {
  // The window's ticket to the API. HttpOnly: page script never sees it.
  // No expiry: it lives in memory only and dies with the app.
  await session.defaultSession.cookies.set({
    url: BACKEND_URL, name: "df_session", value: TOKEN, httpOnly: true, sameSite: "strict",
  });

  mainWindow = new BrowserWindow({
    width: 1280,
    height: 800,
    minWidth: 900,
    minHeight: 600,
    title: "DataForge",
    backgroundColor: nativeTheme.shouldUseDarkColors ? "#161618" : "#f4f4f1",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      webSecurity: true,
    },
    // Remove default menu bar (feels more like an app)
    autoHideMenuBar: true,
  });

  mainWindow.loadURL(BACKEND_URL);

  // The renderer's listener only attaches after its own script runs, which is
  // after this — so resend whatever we know once the page is actually ready.
  mainWindow.webContents.on("did-finish-load", () => {
    mainWindow.webContents.send("df:model-status", lastModelStatus);
  });

  // Open external web links in the system browser, not inside the app. Only
  // http(s): links come from web search results, and other schemes (file:,
  // ms-*, custom protocol handlers) can launch programs.
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https?:\/\//i.test(url)) shell.openExternal(url);
    return { action: "deny" };
  });

  // The window only ever shows DataForge itself
  const stayHome = (event, url) => {
    if (new URL(url).origin !== BACKEND_URL) event.preventDefault();
  };
  mainWindow.webContents.on("will-navigate", stayHome);
  mainWindow.webContents.on("will-redirect", stayHome);

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
        // A dev tool, not for installed builds: its console can call the API as the window
        ...(app.isPackaged ? [] : [
          { label: "Developer Tools", accelerator: "CmdOrCtrl+Shift+I", click: () => mainWindow?.webContents.openDevTools() },
        ]),
      ],
    },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

// ── 4. App lifecycle ──────────────────────────────────────────────────────────
app.whenReady().then(async () => {
  buildMenu();
  await startOllama();
  // Don't block backend/window startup on this — the UI shows a status banner
  // (via df:model-status) while it settles, possibly including a first-run download.
  waitForOllama().then(ok => { if (ok) ensureModelReady(); });
  try {
    await startBackend();
    await waitForBackend();
    await createWindow();
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
  if (ollamaStartedByUs && ollamaProc && !ollamaProc.killed) {
    console.log("[DataForge] Shutting down Ollama...");
    ollamaProc.kill("SIGTERM");
  }
});
