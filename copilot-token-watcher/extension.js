/**
 * Copilot Token Watcher — extension.js
 *
 * Reads GitHub Copilot Chat session files written locally by VS Code on Windows,
 * parses per-prompt token counts and AI credit costs, and displays them in the
 * status bar and a webview history panel.
 *
 * IMPORTANT: This extension must run as a UI extension (see "extensionKind": ["ui"]
 * in package.json). When VS Code is connected to a remote (SSH/WSL/Dev Container),
 * the extension still executes on the LOCAL Windows machine so it can read
 * %APPDATA%\Code\User\workspaceStorage\*\chatSessions\*.jsonl.
 */

'use strict';

const vscode = require('vscode');
const fs = require('fs');
const path = require('path');
const os = require('os');

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Root folder where VS Code writes per-workspace chat session files on Windows. */
const APPDATA = process.env.APPDATA || path.join(os.homedir(), 'AppData', 'Roaming');
const SESSION_BASE = path.join(APPDATA, 'Code', 'User', 'workspaceStorage');

/** How often (ms) to poll for new requests. 5 s is responsive without hammering disk. */
const POLL_INTERVAL_MS = 5_000;

/** How long (ms) to cache the list of today's session files before rescanning. */
const FILE_LIST_CACHE_TTL_MS = 60_000;

// ---------------------------------------------------------------------------
// Module-level state
// ---------------------------------------------------------------------------

let statusBar;
let panel;

/** All parsed requests for the current calendar day. Reset at midnight. */
let todayRequests = [];

/** Byte offsets per file — lets us read only new bytes, not the full file each poll. */
let lastSeenSize = {};

/** Cached list of session file paths for today. Refreshed every FILE_LIST_CACHE_TTL_MS. */
let cachedTodayFiles = [];
let fileListCachedAt = 0;

/** Date string (e.g. "Mon Jun 01 2026") used to detect midnight rollover. */
let currentDay = new Date().toDateString();

/** Monotonically increasing counter assigned to each parsed request. Guarantees
 *  stable display order even when no real timestamp is available in the JSONL. */
let nextSeq = 0;

let watcherInterval;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Escapes special HTML characters so user-supplied strings can be safely
 * inserted into HTML content or attributes without risk of XSS.
 *
 * @param {string} str - Raw string to escape.
 * @returns {string} HTML-safe string.
 */
function escapeHtml(str) {
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

// ---------------------------------------------------------------------------
// File discovery
// ---------------------------------------------------------------------------

/**
 * Returns the list of `.jsonl` chat session files that were modified today.
 * Results are cached for FILE_LIST_CACHE_TTL_MS milliseconds to avoid
 * repeatedly scanning all workspace folders on every poll cycle.
 *
 * @param {boolean} [forceRefresh=false] - Bypass cache and rescan immediately.
 * @returns {string[]} Absolute paths to today's session files.
 */
function getTodaySessionFiles(forceRefresh = false) {
    const now = Date.now();
    if (!forceRefresh && now - fileListCachedAt < FILE_LIST_CACHE_TTL_MS) {
        return cachedTodayFiles;
    }

    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const files = [];
    try {
        const workspaceFolders = fs.readdirSync(SESSION_BASE);
        for (const folder of workspaceFolders) {
            const chatDir = path.join(SESSION_BASE, folder, 'chatSessions');
            if (!fs.existsSync(chatDir)) continue;
            const jsonlFiles = fs.readdirSync(chatDir).filter(f => f.endsWith('.jsonl'));
            for (const file of jsonlFiles) {
                const filePath = path.join(chatDir, file);
                const stat = fs.statSync(filePath);
                if (stat.mtime >= today) files.push(filePath);
            }
        }
    } catch { /* SESSION_BASE may not exist on non-Windows machines; silently skip. */ }

    cachedTodayFiles = files;
    fileListCachedAt = now;
    return files;
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

/**
 * Parses new Copilot billing events from a session file, reading only the bytes
 * added since `fromByte` (byte-offset tracking avoids full file re-reads).
 *
 * Each JSONL line that contains `promptTokens` is treated as a billing event.
 * The function extracts:
 * - Actual request timestamp (from `j.v.requestTime` or `j.timestamp`; falls back to now)
 * - First 60 chars of the user's prompt (from the `<userRequest>…</userRequest>` block)
 * - Resolved model name
 * - Prompt and output token counts
 * - Credit cost (parsed from the `details` field after stripping multi-byte chars)
 *
 * @param {string} filePath - Absolute path to the `.jsonl` session file.
 * @param {number} [fromByte=0] - Byte offset to start reading from.
 * @returns {{ requests: object[], newSize: number }} Parsed requests and updated file size.
 */
function parseRequests(filePath, fromByte = 0) {
    const requests = [];
    try {
        const stat = fs.statSync(filePath);
        if (stat.size <= fromByte) return { requests, newSize: fromByte };

        // Read only the new bytes since the last check.
        const fd = fs.openSync(filePath, 'r');
        const buf = Buffer.alloc(stat.size - fromByte);
        fs.readSync(fd, buf, 0, buf.length, fromByte);
        fs.closeSync(fd);

        const lines = buf.toString('utf8').split('\n').filter(Boolean);

        for (const line of lines) {
            try {
                // Quick pre-filter — skip lines that can't be billing events.
                if (!line.includes('promptTokens')) continue;

                const j = JSON.parse(line);
                const meta = j?.v?.metadata;
                const details = j?.v?.details || '';
                if (!meta?.promptTokens) continue;

                // Resolve actual event time. Try all known field locations:
                //   j.v.requestTime  — Windows chatSessions format
                //   j.timestamp      — ISO string top-level
                //   j.ts             — Unix-ms top-level (debug-log format)
                // If none are present, time is null — the seq field guarantees order.
                const rawTime = j?.v?.requestTime ?? j?.timestamp ?? j?.ts;
                const time = rawTime ? new Date(rawTime) : null;

                // The rendered user message is an array of content blocks; the first
                // text block contains the full prompt wrapped in an XML-like tag.
                const userMsgBlock = meta?.renderedUserMessage?.[0]?.text || '';
                const userRequestMatch = userMsgBlock.match(/<userRequest>\s*([\s\S]*?)\s*<\/userRequest>/);
                const prompt = userRequestMatch
                    ? userRequestMatch[1].trim().substring(0, 60)
                    : '(no prompt text)';

                // The `details` field uses multi-byte Unicode (e.g. bullet • → â€¢).
                // Strip non-ASCII before applying the credits regex to avoid false negatives.
                const detailsAscii = details.replace(/[^\x00-\x7F]/g, '');
                const creditsMatch = detailsAscii.match(/([\d.]+)\s*credits/i);

                requests.push({
                    time,
                    prompt,
                    model: (meta.resolvedModel || '').replace('claude-', '').replace(/-/g, ' '),
                    promptTokens: meta.promptTokens,
                    outputTokens: meta.outputTokens,
                    credits: creditsMatch ? parseFloat(creditsMatch[1]) : null,
                    creditsLabel: detailsAscii.replace(/\s+/g, ' ').trim()
                });
            } catch { /* Malformed line — skip silently. */ }
        }
        return { requests, newSize: stat.size };
    } catch {
        return { requests, newSize: fromByte };
    }
}

// ---------------------------------------------------------------------------
// State management
// ---------------------------------------------------------------------------

/**
 * (Re-)loads all of today's request history from scratch.
 * Called once on activation and again automatically at midnight.
 * Resets `todayRequests`, `lastSeenSize`, and the file-list cache.
 */
function loadTodayHistory() {
    todayRequests = [];
    lastSeenSize = {};  // Prunes stale byte-offset entries from the previous day.
    nextSeq = 0;
    currentDay = new Date().toDateString();
    const files = getTodaySessionFiles(/* forceRefresh */ true);
    for (const f of files) {
        const { requests, newSize } = parseRequests(f, 0);
        requests.forEach(r => { r.seq = nextSeq++; });
        todayRequests.push(...requests);
        lastSeenSize[f] = newSize;
    }
    todayRequests.sort((a, b) => a.seq - b.seq);
    updateStatusBar();
}

/**
 * Polls for new data appended to today's session files since the last check.
 * Also detects midnight rollover and triggers a full reload for the new day.
 * Called on a fixed interval (`POLL_INTERVAL_MS`).
 */
function checkForNewRequests() {
    // Midnight rollover — reset all state for the new calendar day.
    if (new Date().toDateString() !== currentDay) {
        loadTodayHistory();
        if (panel) renderPanel();
        return;
    }

    // File list is re-fetched from disk at most once per FILE_LIST_CACHE_TTL_MS.
    const files = getTodaySessionFiles();
    let hasNew = false;
    for (const f of files) {
        const fromByte = lastSeenSize[f] || 0;
        const { requests, newSize } = parseRequests(f, fromByte);
        if (requests.length > 0) {
            requests.forEach(r => { r.seq = nextSeq++; });
            todayRequests.push(...requests);
            hasNew = true;
        }
        lastSeenSize[f] = newSize;
    }
    if (hasNew) {
        updateStatusBar();
        if (panel) renderPanel();
    }
}

// ---------------------------------------------------------------------------
// UI
// ---------------------------------------------------------------------------

/**
 * Updates the status bar item with stats from the most recent request.
 * Shows token counts (in → out) and credit cost for the last prompt,
 * plus aggregate totals in the tooltip.
 */
function updateStatusBar() {
    if (!statusBar) return;
    if (todayRequests.length === 0) {
        statusBar.text = '$(beaker) Copilot: no prompts yet';
        statusBar.tooltip = 'No Copilot prompts today. Click to open history.';
        return;
    }
    const last = todayRequests[todayRequests.length - 1];
    const totalCredits = todayRequests.reduce((s, r) => s + (r.credits || 0), 0);
    const totalIn = todayRequests.reduce((s, r) => s + (r.promptTokens || 0), 0);
    statusBar.text = `$(beaker) ${last.promptTokens.toLocaleString()}→${last.outputTokens} | ${last.creditsLabel || (last.credits + ' credits')}`;
    statusBar.tooltip = `Last prompt: ${last.promptTokens.toLocaleString()} in / ${last.outputTokens} out\nToday total: ${todayRequests.length} prompts | ${totalIn.toLocaleString()} tokens in | ${totalCredits.toFixed(1)} credits\nClick to see full history`;
}

/**
 * Renders (or re-renders) the full history webview panel.
 * All user-supplied strings are HTML-escaped before insertion to prevent XSS.
 * Called on panel open and whenever new requests are detected.
 */
function renderPanel() {
    if (!panel) return;
    const totalCredits = todayRequests.reduce((s, r) => s + (r.credits || 0), 0);
    const totalIn = todayRequests.reduce((s, r) => s + (r.promptTokens || 0), 0);
    const totalOut = todayRequests.reduce((s, r) => s + (r.outputTokens || 0), 0);

    const rows = [...todayRequests].reverse().map(r => {
        const ratio = r.outputTokens > 0 ? Math.round(r.promptTokens / r.outputTokens) : '∞';
        const creditsStr = r.credits != null ? r.credits.toFixed(1) : '?';
        const timeStr = r.time
            ? r.time.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
            : '--:--:--';
        const safePrompt = escapeHtml(r.prompt);
        return `<tr>
            <td class="seq">#${r.seq + 1}</td>
            <td class="time">${timeStr}</td>
            <td class="prompt" title="${safePrompt}">${r.prompt.length >= 60 ? safePrompt + '…' : safePrompt}</td>
            <td class="model">${escapeHtml(r.model)}</td>
            <td class="num">${r.promptTokens.toLocaleString()}</td>
            <td class="num">${r.outputTokens.toLocaleString()}</td>
            <td class="ratio">${ratio}x</td>
            <td class="credits ${r.credits > 10 ? 'high' : r.credits > 5 ? 'med' : 'low'}">${creditsStr}</td>
        </tr>`;
    }).join('');

    panel.webview.html = `<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; font-size: 13px; background: var(--vscode-editor-background); color: var(--vscode-editor-foreground); padding: 20px; }
  h1 { font-size: 16px; font-weight: 600; margin-bottom: 4px; color: var(--vscode-editor-foreground); }
  .subtitle { font-size: 12px; color: var(--vscode-descriptionForeground); margin-bottom: 20px; }
  .summary { display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }
  .card { background: var(--vscode-editorWidget-background); border: 1px solid var(--vscode-editorWidget-border, #444); border-radius: 6px; padding: 12px 18px; min-width: 140px; }
  .card .label { font-size: 11px; color: var(--vscode-descriptionForeground); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; }
  .card .value { font-size: 22px; font-weight: 700; color: var(--vscode-editor-foreground); }
  .card .unit { font-size: 11px; color: var(--vscode-descriptionForeground); margin-left: 3px; }
  .insight { background: var(--vscode-editorInfo-background, rgba(0,122,204,0.1)); border-left: 3px solid var(--vscode-editorInfo-foreground, #3794ff); border-radius: 0 4px 4px 0; padding: 10px 14px; margin-bottom: 20px; font-size: 12px; color: var(--vscode-editor-foreground); line-height: 1.6; }
  table { width: 100%; border-collapse: collapse; }
  thead th { font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--vscode-descriptionForeground); padding: 6px 10px; text-align: left; border-bottom: 1px solid var(--vscode-editorWidget-border, #444); }
  tbody tr:hover { background: var(--vscode-list-hoverBackground); }
  td { padding: 7px 10px; border-bottom: 1px solid var(--vscode-editorWidget-border, rgba(128,128,128,0.2)); vertical-align: middle; }
  td.seq { font-family: monospace; font-size: 11px; color: var(--vscode-descriptionForeground); text-align: right; white-space: nowrap; min-width: 32px; }
  td.time { font-family: monospace; font-size: 12px; color: var(--vscode-descriptionForeground); white-space: nowrap; }
  td.prompt { max-width: 280px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--vscode-editor-foreground); }
  td.model { font-size: 11px; color: var(--vscode-descriptionForeground); white-space: nowrap; }
  td.num { font-family: monospace; text-align: right; color: var(--vscode-editor-foreground); }
  td.ratio { font-family: monospace; text-align: right; color: var(--vscode-descriptionForeground); font-size: 11px; }
  td.credits { font-family: monospace; font-weight: 600; text-align: right; }
  td.credits.low { color: #4caf50; }
  td.credits.med { color: #ff9800; }
  td.credits.high { color: #f44336; }
  .empty { text-align: center; padding: 40px; color: var(--vscode-descriptionForeground); }
</style>
</head>
<body>
<h1>Copilot Token Watcher</h1>
<p class="subtitle">Today's session — ${new Date().toLocaleDateString('en-GB', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}</p>

<div class="summary">
  <div class="card"><div class="label">Prompts</div><div class="value">${todayRequests.length}</div></div>
  <div class="card"><div class="label">Tokens in</div><div class="value">${(totalIn / 1000).toFixed(0)}<span class="unit">k</span></div></div>
  <div class="card"><div class="label">Tokens out</div><div class="value">${(totalOut / 1000).toFixed(1)}<span class="unit">k</span></div></div>
  <div class="card"><div class="label">Credits used</div><div class="value">${totalCredits.toFixed(1)}<span class="unit">cr</span></div></div>
  <div class="card"><div class="label">Cost</div><div class="value">$${(totalCredits * 0.01).toFixed(2)}</div></div>
</div>

${totalIn > 0 ? `<div class="insight">
  💡 ${((totalIn / (totalIn + totalOut)) * 100).toFixed(1)}% of all tokens today were <strong>input context overhead</strong> — not your actual responses.
  Average input per prompt: <strong>${Math.round(totalIn / todayRequests.length).toLocaleString()} tokens</strong>.
</div>` : ''}

${todayRequests.length === 0 ? '<div class="empty">No Copilot prompts recorded today yet.</div>' : `
<table>
  <thead><tr>
    <th>#</th><th>Time</th><th>Prompt</th><th>Model</th>
    <th style="text-align:right">In</th>
    <th style="text-align:right">Out</th>
    <th style="text-align:right">Ratio</th>
    <th style="text-align:right">Credits</th>
  </tr></thead>
  <tbody>${rows}</tbody>
</table>`}
</body>
</html>`;
}

// ---------------------------------------------------------------------------
// Extension lifecycle
// ---------------------------------------------------------------------------

/**
 * Called by VS Code when the extension activates (on startup, after the UI is ready).
 * Registers the status bar item and the history panel command, loads today's history,
 * and starts the polling interval.
 *
 * @param {vscode.ExtensionContext} context - Extension context for registering disposables.
 */
function activate(context) {
    statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
    statusBar.command = 'copilotTokenWatcher.showHistory';
    statusBar.show();
    context.subscriptions.push(statusBar);

    context.subscriptions.push(
        vscode.commands.registerCommand('copilotTokenWatcher.showHistory', () => {
            if (!panel) {
                panel = vscode.window.createWebviewPanel(
                    'copilotTokenHistory',
                    'Copilot Token Watcher',
                    vscode.ViewColumn.Beside,
                    { enableScripts: true }
                );
                panel.onDidDispose(() => { panel = null; });
            }
            renderPanel();
            panel.reveal();
        })
    );

    loadTodayHistory();
    // Poll for new requests at a fixed interval. The interval handle is registered
    // as a disposable so it is automatically cleared when the extension deactivates.
    watcherInterval = setInterval(checkForNewRequests, POLL_INTERVAL_MS);
    context.subscriptions.push({ dispose: () => clearInterval(watcherInterval) });
}

/**
 * Called by VS Code when the extension deactivates (window close, reload, uninstall).
 * Clears the polling interval as a safety net (subscriptions also handle this).
 */
function deactivate() {
    clearInterval(watcherInterval);
}

module.exports = { activate, deactivate };
