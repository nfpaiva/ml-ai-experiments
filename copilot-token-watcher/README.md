# 🔭 Copilot Token Watcher

A lightweight, zero-dependency VS Code extension that monitors **GitHub Copilot Chat token usage** and **AI credit cost** per prompt — directly in your status bar.

> **Why?** As of June 2026, GitHub Copilot switched to token-based AI Credits billing (1 credit ≈ $0.01). VS Code does **not** show a per-prompt token counter natively. This extension fills that gap by reading Copilot's own local session files.

---

## 📁 Repository structure

This folder contains the **full, inspectable source code** of the extension. You can read every file before packaging it into a `.vsix`:

- `extension.js` — ~300 lines of vanilla JavaScript. Core logic: file watcher, parser, status bar, webview panel.
- `package.json` — VS Code extension manifest (activation events, commands, devDependencies).
- `.vscodeignore` — Excludes build artefacts and editor settings from the final `.vsix`.
- `README.md` — This file.

No transpiling. No bundlers. No hidden build steps. What you see is what runs.

---

## 🛠️ How it works

The extension reads VS Code's local Copilot Chat session data:

```
%APPDATA%\Code\User\workspaceStorage\*\chatSessions\*.jsonl
```

Each `.jsonl` line is a JSON event. The billing event contains `promptTokens`, `outputTokens`, `resolvedModel`, and a `details` string with credit cost. The extension:

1. **Scans** all today's `.jsonl` files on startup.
2. **Polls** every 5 seconds using efficient byte-offset tracking (no full re-reads).
3. **Parses** token counts and credits from each new request.
4. **Displays** last-prompt stats in the status bar, and a full history panel on click.

### Architecture note
The extension runs in the **VS Code extension host on Windows**, so it correctly reads `%APPDATA%` even when VS Code itself is connected via SSH to a Linux VM. This is intentional and by design.

---

## ✅ Features

| Feature | Description |
|---------|-------------|
| **Status bar** (bottom-right) | Shows last prompt tokens in → out and credits in real time |
| **Click to open history** | Full webview panel with today's session breakdown |
| **Summary cards** | Total prompts, tokens in/out, credits used, dollar cost |
| **Per-prompt table** | Time, prompt snippet, **session title**, **workspace**, model, tokens, input/output ratio, credits (colour-coded) |
| **Session & workspace columns** | Each row shows the VS Code chat session name (`customTitle`) and the workspace folder name, so multi-project days are easy to filter at a glance |
| **Insight line** | % of tokens that were invisible context overhead vs. actual response |
| **Context breakdown panel** | Click any row to see a per-section token estimate (prompt, attachments, tool schemas, conversation history) |
| **Reducible% column** | Highlights how many input tokens could be saved by disabling unused MCP servers or extensions |
| **Date picker** | Browse any day within the current month |
| **Auto-refresh** | New prompts detected within 5 seconds without re-scanning entire files |

---

## ⚙️ Prerequisites

- [Node.js](https://nodejs.org/) (LTS recommended) and `npm`
- [VS Code](https://code.visualstudio.com/) 1.80+
- `vsce` — the VS Code extension packaging tool

Install `vsce` globally if you haven't:

```bash
npm install -g @vscode/vsce
```

---

## 📦 Build & install

All commands are run from **inside** this folder:

```bash
# 1. Enter the extension source folder
cd copilot-token-watcher

# 2. Install devDependencies
npm install

# 3. Package into a .vsix file
npx vsce package --no-dependencies --allow-missing-repository
```

This produces a file like `copilot-token-watcher-0.1.0.vsix` in this same folder.

### Install into VS Code

Because this extension is declared as a **UI extension** (`"extensionKind": ["ui"]`), it must be installed on the **local Windows machine** where VS Code is running — not on a remote server.

> If you built the `.vsix` on a remote machine (SSH/WSL), you must first **download the file to your local PC** before installing.

**To download** (if built remotely):  
In VS Code's Explorer panel, right-click `copilot-token-watcher-0.1.0.vsix` → **Download…**

**To install locally:**  
Open VS Code → Extensions view (`Ctrl+Shift+X`) → `…` menu → **Install from VSIX…** → browse to the downloaded file.

> ⚠️ Running `code --install-extension` from a remote terminal will fail with "extension is declared to not run in this setup" — this is expected. Always install via the Extensions UI on the local machine.

---

## 🚀 Usage

1. Start VS Code.
2. Look at the **bottom-right status bar** — you'll see something like:
   ```
   🧪 17,331→5 | Claude Sonnet 4.6 • 3.3 credits
   ```
3. **Click** the status bar item to open the full history panel.

The panel shows:
- Summary cards (prompts, tokens in/out, credits, cost)
- An insight line explaining context overhead
- A per-prompt table with colour-coded credit usage

---

## 🐛 Known limitations

| Limitation | Details |
|---|---|
| **New thread detection delay** | When starting a brand-new conversation, a new `.jsonl` file is created. Detection happens on the next 5-second poll cycle, not instantly. |
| **Encoding artifacts** | The `details` field from Copilot uses multi-byte characters (`•` shows as `â€¢`). Credits are parsed robustly via regex after stripping non-ASCII. |
| **Windows-only runtime** | Reads from `%APPDATA%`. The extension installs on any OS, but will only display data when running on Windows. |
| **File list cache TTL** | The list of today's session files is refreshed from disk at most once per minute (not every poll). A brand-new chat thread created within the last minute may be missed until the cache expires. |
| **Midnight boundary** | State resets automatically when the calendar day changes (detected on the next poll cycle). The final few seconds before midnight may technically be attributed to the new day. |
| **Timestamp accuracy** | Event times are read from `requestTime` or `timestamp` fields in the JSONL data. If neither is present, the extension falls back to the time the entry was parsed (typically within 5 seconds of the actual request). |

---

## 🧹 Operational notes / maintenance

**Nothing to clean up.** This extension is purely read-only:
- It **creates no files**, writes no databases, and appends to nothing.
- All session data it displays is sourced from Copilot's own `.jsonl` files in `%APPDATA%\Code\User\workspaceStorage\`. Those files are managed entirely by VS Code/Copilot and are unaffected by this extension.

**Startup scan performance.**  
On activation, the extension scans every folder inside `workspaceStorage` looking for today's session files. If you've accumulated hundreds of old workspace folders over months/years, this scan gets slower. VS Code allows you to prune stale workspace storage safely:
> `Ctrl+Shift+P` → **Developer: Remove Stale Workspace Storage Entries**

This removes orphaned folders whose workspace no longer exists and speeds up the startup scan.

**Polling impact.**  
The extension polls every **5 seconds**. Each poll reads at most a few kilobytes of new data per file (byte-offset tracking). The file-system folder listing (`readdirSync`) is cached and only refreshed once per minute. CPU and I/O impact is negligible on any modern machine.

**Clean uninstall.**  
Uninstalling the extension removes all impact immediately. No residual data, no background processes. Copilot's session files remain untouched.

---

## 🔧 Development

Since the source is plain JavaScript, you can debug it directly:

1. Open this folder in VS Code (`File → Open Folder…`).
2. Press `F5` to launch the **Extension Development Host**.
3. The extension will activate automatically; check the status bar.

---

## 📜 License

Personal / experimental. Not published to the VS Code Marketplace. Install via `.vsix` only.
