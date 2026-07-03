/**
 * DataForge — Electron Preload Script
 * Copyright (C) 2026 Mohammed Farazuddin
 *
 * Runs in a privileged context before the renderer page loads.
 * Exposes only what the frontend needs — nothing else.
 * contextIsolation: true means the renderer cannot access Node.js directly.
 */

const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("dataforge", {
  // App version from package.json
  version: process.env.npm_package_version ?? "0.1.0",

  // Platform info (useful for UI tweaks)
  platform: process.platform,

  // Future: add IPC calls here as the app grows
  // e.g. openFilePicker: () => ipcRenderer.invoke("open-file-picker"),
});
