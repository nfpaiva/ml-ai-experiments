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

/** YYYY-MM-DD string of the date currently shown in the panel. Defaults to today. */
let selectedDateStr = new Date().toISOString().split('T')[0];

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
 * - Actual request timestamp (from `j.v.requestTime`, `j.timestamp`, or `j.ts`)
 * - First 60 chars of the user's prompt via a three-step fallback chain:
 *     1. `<userRequest>` tag in `meta.renderedUserMessage`
 *     2. Matching `kind:0` session header message at the same billing-event index
 *     3. `(agent loop)` — internal automated turn with no user message
 * - Resolved model name
 * - Prompt and output token counts
 * - Credit cost (parsed from the `details` field after stripping multi-byte chars)
 *
 * @param {string} filePath - Absolute path to the `.jsonl` session file.
 * @param {number} [fromByte=0] - Byte offset to start reading from.
 * @returns {{ requests: object[], newSize: number }} Parsed requests and updated file size.
 */
function parseRequests(filePath, fromByte = 0, fallbackTime = null) {
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

        // First pass: collect kind:0 session header messages (and their timestamps) in
        // file order. These carry the raw user message text and optionally the request
        // time for each turn. Pushed unconditionally (null slots preserved) so that
        // kindZeroMessages[i] and kindZeroTimes[i] stay in sync with billingEventIndex.
        // Field paths are best-effort assumptions; silently ignored if wrong.
        const kindZeroMessages = [];
        const kindZeroTimes = [];
        for (const line of lines) {
            try {
                if (!line.includes('"kind"')) continue;
                const j = JSON.parse(line);
                const kind = j?.kind ?? j?.v?.kind;
                if (kind !== 0) continue;
                const text = j?.message?.text
                    ?? j?.v?.message?.text
                    ?? j?.messages?.[0]?.text
                    ?? j?.v?.messages?.[0]?.text
                    ?? null;
                kindZeroMessages.push(text ? String(text).trim().substring(0, 60) : null);
                const rawT = j?.v?.requestTime ?? j?.v?.timestamp ?? j?.v?.time ?? j?.v?.createdAt
                    ?? j?.requestTime ?? j?.timestamp ?? j?.ts ?? j?.time ?? j?.createdAt ?? null;
                kindZeroTimes.push(rawT ? new Date(rawT) : null);
            } catch { /* skip malformed lines */ }
        }

        let billingEventIndex = 0;

        for (const line of lines) {
            try {
                // Quick pre-filter — skip lines that can't be billing events.
                if (!line.includes('promptTokens')) continue;

                const j = JSON.parse(line);
                const meta = j?.v?.metadata;
                const details = j?.v?.details || '';
                if (!meta?.promptTokens) continue;

                // Resolve actual event time. Try all known field locations across
                // different VS Code / Copilot JSONL schema variants, then fall back
                // to the matching kind:0 header timestamp collected in the first pass.
                // If nothing is found, use fallbackTime (set to now() when called from
                // checkForNewRequests, so live-detected records always get a real time).
                const rawTime = j?.v?.requestTime ?? j?.v?.timestamp ?? j?.v?.time ?? j?.v?.createdAt
                    ?? j?.requestTime ?? j?.timestamp ?? j?.ts ?? j?.time ?? j?.createdAt ?? null;
                const time = rawTime ? new Date(rawTime) : (kindZeroTimes[billingEventIndex] ?? fallbackTime);

                // Three-step prompt extraction:
                //   1. <userRequest> tag in renderedUserMessage (primary, most reliable)
                //   2. kind:0 header message at the same billing-event index (fallback)
                //   3. '(agent loop)' — internal automated turn with no real user message
                const userMsgBlock = meta?.renderedUserMessage?.[0]?.text || '';
                const userRequestMatch = userMsgBlock.match(/<userRequest>\s*([\s\S]*?)\s*<\/userRequest>/);
                let prompt;
                let isAgentLoop = false;
                if (userRequestMatch) {
                    prompt = userRequestMatch[1].trim().substring(0, 60);
                } else if (kindZeroMessages[billingEventIndex] != null) {
                    prompt = kindZeroMessages[billingEventIndex];
                } else {
                    prompt = '(agent loop)';
                    isAgentLoop = true;
                }

                // The `details` field uses multi-byte Unicode (e.g. bullet • → â€¢).
                // Strip non-ASCII before applying the credits regex to avoid false negatives.
                const detailsAscii = details.replace(/[^\x00-\x7F]/g, '');
                const creditsMatch = detailsAscii.match(/([\d.]+)\s*credits/i);

                // Build context breakdown: each XML section in renderedUserMessage
                // gets a { label, charCount } entry. Token estimates (charCount/4)
                // and percentages are computed at render time, not stored here.
                const breakdown = [];
                const extractSection = (tag, label) => {
                    const m = userMsgBlock.match(new RegExp(`<${tag}(?:[^>]*)>([\\s\\S]*?)<\\/${tag}>`, ''));
                    if (m) breakdown.push({ label, charCount: m[1].length });
                };
                // Named single-instance tags
                extractSection('userRequest',          'Your prompt');
                extractSection('workspace_info',       'Workspace tree');
                extractSection('availableDeferredTools','Tool schemas (MCP)');
                extractSection('editorContext',        'Open file / notebook');
                extractSection('reminderInstructions', 'System instructions');
                // Attachments — multiple per message, each with an id attribute
                for (const am of userMsgBlock.matchAll(/<attachment\s+id="([^"]*)"[^>]*>([\s\S]*?)<\/attachment>/g)) {
                    breakdown.push({ label: `File: ${am[1]}`, charCount: am[2].length });
                }
                // contentReferences array — auto-injected instruction files
                if (Array.isArray(meta.contentReferences)) {
                    for (const ref of meta.contentReferences) {
                        const refLabel = ref.uri ?? ref.name ?? ref.path ?? JSON.stringify(ref);
                        breakdown.push({ label: `Instructions: ${String(refLabel).split('/').pop()}`, charCount: JSON.stringify(ref).length });
                    }
                }
                // Remainder — chars not accounted for by tagged sections
                const taggedChars = breakdown.reduce((s, b) => s + b.charCount, 0);
                const remainder = userMsgBlock.length - taggedChars;
                if (remainder > 50) breakdown.push({ label: 'Other / untagged', charCount: remainder });

                requests.push({
                    time,
                    prompt,
                    isAgentLoop,
                    model: (meta.resolvedModel || '').replace('claude-', '').replace(/-/g, ' '),
                    promptTokens: meta.promptTokens,
                    outputTokens: meta.outputTokens,
                    credits: creditsMatch ? parseFloat(creditsMatch[1]) : null,
                    creditsLabel: detailsAscii.replace(/\s+/g, ' ').trim(),
                    breakdown
                });

                billingEventIndex++;
            } catch { /* Malformed line — skip silently. */ }
        }
        return { requests, newSize: stat.size };
    } catch {
        return { requests, newSize: fromByte };
    }
}

/**
 * Groups consecutive `(agent loop)` requests under the preceding user-initiated
 * request. Returns an array of group objects, each with a `leader` (the user
 * prompt) and zero or more `children` (agent loop turns that immediately followed).
 *
 * Groups with no children represent ordinary single-turn user prompts and are
 * rendered as flat rows identical to the pre-grouping layout.
 *
 * @param {object[]} requests - Request objects in display order.
 * @returns {{ leader: object, children: object[] }[]} Array of groups.
 */
function buildGroups(requests) {
    // requests must be in CHRONOLOGICAL order (oldest first).
    // Agent-loop turns are internal calls that fire before their parent billing
    // event and therefore appear at a lower seq. Buffer them and attach to the
    // NEXT non-agent-loop turn so they're shown as children of the user prompt
    // that triggered them.
    // Trailing agent loops with no following user prompt are attached to the
    // last group (or become a standalone group if the list is entirely agent loops).
    const groups = [];
    let pending = [];   // buffered agent-loop turns waiting for the next leader
    for (const r of requests) {
        if (r.isAgentLoop) {
            pending.push(r);
        } else {
            groups.push({ leader: r, children: pending });
            pending = [];
        }
    }
    // Flush any trailing agent-loop turns.
    if (pending.length > 0) {
        if (groups.length > 0) {
            groups[groups.length - 1].children.push(...pending);
        } else {
            // Edge case: every entry in the file is an agent loop.
            for (const r of pending) groups.push({ leader: r, children: [] });
        }
    }
    return groups;
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
    selectedDateStr = new Date().toISOString().split('T')[0];
    const files = getTodaySessionFiles(/* forceRefresh */ true);
    for (const f of files) {
        // Use the file's last-modified time as fallback for records with no
        // embedded timestamp — better than --:--:-- for startup history.
        const fileMtime = (() => { try { return fs.statSync(f).mtime; } catch { return null; } })();
        const { requests, newSize } = parseRequests(f, 0, fileMtime);
        requests.forEach(r => { r.seq = nextSeq++; });
        todayRequests.push(...requests);
        lastSeenSize[f] = newSize;
    }
    todayRequests.sort((a, b) => a.seq - b.seq);
    updateStatusBar();
}

/**
 * Loads request history for a specific calendar day (YYYY-MM-DD).
 * Scans all session files with mtime on or after that day, parses them fully,
 * then filters to only requests whose timestamp falls within that exact day.
 * Called when the user selects a date from the panel date picker.
 *
 * @param {string} dateStr - ISO date string, e.g. "2026-06-01".
 */
function loadHistoryForDate(dateStr) {
    selectedDateStr = dateStr;
    const dayStart = new Date(dateStr + 'T00:00:00');
    const dayEnd = new Date(dayStart.getTime() + 24 * 60 * 60 * 1000);
    todayRequests = [];
    lastSeenSize = {};
    nextSeq = 0;
    const files = [];
    try {
        const workspaceFolders = fs.readdirSync(SESSION_BASE);
        for (const folder of workspaceFolders) {
            const chatDir = path.join(SESSION_BASE, folder, 'chatSessions');
            if (!fs.existsSync(chatDir)) continue;
            const jsonlFiles = fs.readdirSync(chatDir).filter(f => f.endsWith('.jsonl'));
            for (const file of jsonlFiles) {
                const fp = path.join(chatDir, file);
                try {
                    if (fs.statSync(fp).mtime >= dayStart) files.push(fp);
                } catch { /* unreadable — skip */ }
            }
        }
    } catch { /* SESSION_BASE may not exist; silently skip. */ }
    for (const f of files) {
        const fileMtime = (() => { try { return fs.statSync(f).mtime; } catch { return null; } })();
        const { requests, newSize } = parseRequests(f, 0, fileMtime);
        const filtered = requests.filter(r => r.time && r.time >= dayStart && r.time < dayEnd);
        filtered.forEach(r => { r.seq = nextSeq++; });
        todayRequests.push(...filtered);
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

    // Only live-update when the panel is showing today's data.
    const _todayStr = new Date().toISOString().split('T')[0];
    if (selectedDateStr !== _todayStr) return;

    // File list is re-fetched from disk at most once per FILE_LIST_CACHE_TTL_MS.
    const files = getTodaySessionFiles();
    let hasNew = false;
    const detectionTime = new Date();
    for (const f of files) {
        const fromByte = lastSeenSize[f] || 0;
        const { requests, newSize } = parseRequests(f, fromByte, detectionTime);
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

    const _now = new Date();
    const _toDs = d => `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
    const minDate = _toDs(new Date(_now.getFullYear(), _now.getMonth(), 1));
    const maxDate = _toDs(new Date(_now.getFullYear(), _now.getMonth() + 1, 0));
    const isToday = selectedDateStr === _toDs(_now);
    const dispDate = new Date(selectedDateStr + 'T00:00:00');
    const dateLabel = dispDate.toLocaleDateString('en-GB', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' });

    // Build groups in chronological order, then reverse for newest-first display.
    // Agent loops are buffered and attached to the NEXT user prompt that follows
    // them in time, so reversing after grouping keeps children with their parent.
    const groups = buildGroups([...todayRequests]).reverse();

    // Helper: compute HIST% cell HTML for a single request row.
    // histPct = (billed promptTokens - estimated tagged chars/4) / promptTokens.
    // Represents conversation history (prior turns) as % of total input context.
    const histCellHtml = (r) => {
        const te = Math.round((r.breakdown || []).reduce((s, b) => s + b.charCount, 0) / 4);
        const hp = r.promptTokens > 0 ? (r.promptTokens - te) / r.promptTokens * 100 : 0;
        const cls = hp >= 90 ? 'hist-high' : hp >= 60 ? 'hist-mid' : 'hist-low';
        const warn = hp >= 90 ? '<span title="Conversation history &gt;90% of context \u2014 start a new chat to reduce costs.">\u26a0</span>\u202f' : '';
        return `<td class="hist ${cls}">${warn}${hp.toFixed(1)}%</td>`;
    };

    const rows = groups.map(({ leader, children }) => {
        const timeStr = leader.time
            ? leader.time.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
            : '--:--:--';
        const safePrompt = escapeHtml(leader.prompt);
        const promptDisplay = leader.prompt.length >= 60 ? safePrompt + '…' : safePrompt;

        if (children.length === 0) {
            // Ordinary single-turn row — same layout as before grouping.
            const ratio = leader.outputTokens > 0 ? Math.round(leader.promptTokens / leader.outputTokens) : '∞';
            const creditsStr = leader.credits != null ? leader.credits.toFixed(1) : '?';
            const creditClass = (leader.credits || 0) > 10 ? 'high' : (leader.credits || 0) > 5 ? 'med' : 'low';
            return `<tr class="data-row" data-seq="${leader.seq}" data-ptokens="${leader.promptTokens}" data-time="${leader.time ? leader.time.getTime() : 0}" onclick="rowClick(event,this)">
            <td class="seq">#${leader.seq + 1}</td>
            <td class="time">${timeStr}</td>
            <td class="prompt" title="${safePrompt}">${promptDisplay}</td>
            <td class="model">${escapeHtml(leader.model)}</td>
            <td class="num">${leader.promptTokens.toLocaleString()}</td>
            <td class="num">${leader.outputTokens.toLocaleString()}</td>
            <td class="ratio">${ratio}x</td>
            <td class="credits ${creditClass}">${creditsStr}</td>
            ${histCellHtml(leader)}
        </tr>`;
        }

        // Group row: collapsed by default, showing combined totals.
        const all = [leader, ...children];
        const grpIn = all.reduce((s, r) => s + (r.promptTokens || 0), 0);
        const grpOut = all.reduce((s, r) => s + (r.outputTokens || 0), 0);
        const grpCredits = all.reduce((s, r) => s + (r.credits || 0), 0);
        const grpRatio = grpOut > 0 ? Math.round(grpIn / grpOut) : '∞';
        const grpCreditsStr = grpCredits > 0 ? grpCredits.toFixed(1) : '?';
        const grpCreditClass = grpCredits > 10 ? 'high' : grpCredits > 5 ? 'med' : 'low';
        const gid = leader.seq;

        const headerRow = `<tr class="group-header data-row" data-seq="${leader.seq}" data-ptokens="${leader.promptTokens}" data-time="${leader.time ? leader.time.getTime() : 0}" onclick="rowClick(event,this,${gid})">
            <td class="seq">#${leader.seq + 1}</td>
            <td class="time">${timeStr}</td>
            <td class="prompt" title="${safePrompt}"><span class="toggle" data-gid="${gid}">▶</span> ${promptDisplay} <span class="badge">${all.length} calls</span></td>
            <td class="model">${escapeHtml(leader.model)}</td>
            <td class="num">${grpIn.toLocaleString()}</td>
            <td class="num">${grpOut.toLocaleString()}</td>
            <td class="ratio">${grpRatio}x</td>
            <td class="credits ${grpCreditClass}">${grpCreditsStr}</td>
            ${histCellHtml(leader)}
        </tr>`;

        const childRows = children.map(c => {
            const cTime = c.time
                ? c.time.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
                : '--:--:--';
            const cRatio = c.outputTokens > 0 ? Math.round(c.promptTokens / c.outputTokens) : '∞';
            const cCreditsStr = c.credits != null ? c.credits.toFixed(1) : '?';
            const cCreditClass = (c.credits || 0) > 10 ? 'high' : (c.credits || 0) > 5 ? 'med' : 'low';
            return `<tr class="group-child group-${gid} data-row" data-seq="${c.seq}" data-ptokens="${c.promptTokens}" data-time="${c.time ? c.time.getTime() : 0}" data-parent-seq="${gid}" onclick="rowClick(event,this)">
            <td class="seq">#${c.seq + 1}</td>
            <td class="time">${cTime}</td>
            <td class="prompt"><span class="indent">↳</span> <em>(agent loop)</em></td>
            <td class="model">${escapeHtml(c.model)}</td>
            <td class="num">${c.promptTokens.toLocaleString()}</td>
            <td class="num">${c.outputTokens.toLocaleString()}</td>
            <td class="ratio">${cRatio}x</td>
            <td class="credits ${cCreditClass}">${cCreditsStr}</td>
            ${histCellHtml(c)}
        </tr>`;
        }).join('');

        return headerRow + childRows;
    }).join('');

    panel.webview.html = `<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; font-size: 13px; background: var(--vscode-editor-background); color: var(--vscode-editor-foreground); padding: 20px; }
  h1 { font-size: 16px; font-weight: 600; margin-bottom: 4px; color: var(--vscode-editor-foreground); }
  .subtitle { font-size: 12px; color: var(--vscode-descriptionForeground); margin-bottom: 0; }
  .date-bar { display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px; }
  input[type="date"] { background: var(--vscode-input-background); color: var(--vscode-input-foreground); border: 1px solid var(--vscode-input-border, #555); border-radius: 4px; padding: 4px 8px; font-size: 12px; font-family: inherit; cursor: pointer; color-scheme: dark; }
  input[type="date"]:focus { outline: 1px solid var(--vscode-focusBorder, #007acc); }
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
  tr.data-row { cursor: pointer; }
  tr.group-header { background: var(--vscode-editorWidget-background); }
  tr.group-header td.prompt { font-weight: 600; }
  tr.group-child { display: none; }
  tr.group-child td.prompt { color: var(--vscode-descriptionForeground); font-style: italic; }
  .toggle { display: inline-block; font-size: 10px; margin-right: 4px; min-width: 10px; }
  .badge { font-size: 10px; background: var(--vscode-badge-background, rgba(128,128,128,0.3)); color: var(--vscode-badge-foreground); border-radius: 8px; padding: 1px 6px; margin-left: 6px; vertical-align: middle; font-weight: normal; }
  .indent { color: var(--vscode-descriptionForeground); margin-right: 4px; }
  #breakdown-panel { display: none; position: sticky; bottom: 0; background: var(--vscode-editorWidget-background); border: 1px solid var(--vscode-editorWidget-border, #444); border-radius: 6px; padding: 14px 16px; margin-top: 20px; }
  .bk-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
  .bk-title { font-size: 13px; font-weight: 600; color: var(--vscode-editor-foreground); }
  .bk-close { background: none; border: none; color: var(--vscode-descriptionForeground); font-size: 16px; cursor: pointer; padding: 0 4px; line-height: 1; }
  .bk-close:hover { color: var(--vscode-editor-foreground); }
  .bk-note { font-size: 11px; color: var(--vscode-descriptionForeground); margin-bottom: 10px; }
  #bk-table { font-size: 12px; }
  #bk-table th { font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--vscode-descriptionForeground); padding: 4px 8px; text-align: left; border-bottom: 1px solid var(--vscode-editorWidget-border, #444); }
  #bk-table td { padding: 5px 8px; border-bottom: 1px solid var(--vscode-editorWidget-border, rgba(128,128,128,0.15)); }
  #bk-table td.bk-num { font-family: monospace; text-align: right; }
  #bk-table td.bk-pct { font-family: monospace; text-align: right; min-width: 52px; }
  #bk-table td.bk-bar-cell { width: 120px; padding-right: 12px; }
  .bk-bar-bg { background: var(--vscode-editorWidget-border, rgba(128,128,128,0.2)); border-radius: 3px; height: 6px; width: 100%; }
  .bk-bar-fill { height: 6px; border-radius: 3px; background: var(--vscode-progressBar-background, #3794ff); }
  tr.bk-amber { background: rgba(255,152,0,0.12); }
  tr.bk-red { background: rgba(244,67,54,0.14); }
  tr.bk-history td:first-child { color: var(--vscode-descriptionForeground); }
  tr.bk-history { border-top: 1px dashed var(--vscode-editorWidget-border, #666); }
  .bk-footer { font-size: 11px; color: var(--vscode-descriptionForeground); margin-top: 8px; border-top: 1px solid var(--vscode-editorWidget-border, rgba(128,128,128,0.2)); padding-top: 6px; }
  th.sortable { cursor: pointer; user-select: none; white-space: nowrap; }
  th.sortable:hover { color: var(--vscode-editor-foreground); }
  .sort-ind { font-size: 9px; margin-left: 2px; }
  td.hist { font-family: monospace; text-align: right; font-size: 11px; white-space: nowrap; }
  td.hist-low { color: #4caf50; }
  td.hist-mid { color: #ff9800; }
  td.hist-high { color: #f44336; font-weight: 600; }
</style>
</head>
<body>
<h1>Copilot Token Watcher</h1>
<div class="date-bar">
  <p class="subtitle">${isToday ? 'Today\u2019s session \u2014 ' : ''}${escapeHtml(dateLabel)}</p>
  <input type="date" id="date-picker" min="${minDate}" max="${maxDate}" value="${selectedDateStr}">
</div>

<div class="summary">
  <div class="card"><div class="label">Prompts</div><div class="value">${todayRequests.length}</div></div>
  <div class="card"><div class="label">Tokens in</div><div class="value">${(totalIn / 1000).toFixed(0)}<span class="unit">k</span></div></div>
  <div class="card"><div class="label">Tokens out</div><div class="value">${(totalOut / 1000).toFixed(1)}<span class="unit">k</span></div></div>
  <div class="card"><div class="label">Credits used</div><div class="value">${totalCredits.toFixed(1)}<span class="unit">cr</span></div></div>
  <div class="card"><div class="label">Cost</div><div class="value">$${(totalCredits * 0.01).toFixed(2)}</div></div>
</div>

${totalIn > 0 ? `<div class="insight">
  💡 ${((totalIn / (totalIn + totalOut)) * 100).toFixed(1)}% of all tokens ${isToday ? 'today' : 'on this day'} were <strong>input context overhead</strong> — not your actual responses.
  Average input per prompt: <strong>${Math.round(totalIn / todayRequests.length).toLocaleString()} tokens</strong>.
</div>` : ''}

${todayRequests.length === 0 ? `<div class="empty">No Copilot prompts recorded ${isToday ? 'today' : 'on this date'} yet.</div>` : `
<table>
  <thead><tr>
    <th class="sortable" onclick="sortBy('seq')"># <span class="sort-ind" id="si-seq"></span></th>
    <th class="sortable" onclick="sortBy('time')">Time <span class="sort-ind" id="si-time"></span></th>
    <th>Prompt</th><th>Model</th>
    <th style="text-align:right">In</th>
    <th style="text-align:right">Out</th>
    <th style="text-align:right">Ratio</th>
    <th style="text-align:right">Credits</th>
    <th style="text-align:right">Hist %</th>
  </tr></thead>
  <tbody>${rows}</tbody>
</table>

<div id="breakdown-panel">
  <div class="bk-header">
    <span class="bk-title">Context breakdown &mdash; <span id="bk-seq"></span></span>
    <button class="bk-close" onclick="closeBreakdown()">&#x2715;</button>
  </div>
  <div class="bk-note">Token estimates are approximate (chars &divide; 4). Click any row to inspect its context.</div>
  <table id="bk-table">
    <thead><tr><th>Section</th><th style="text-align:right">Est. tokens</th><th style="text-align:right">% of input</th><th>Usage</th></tr></thead>
    <tbody id="bk-body"></tbody>
  </table>
  <div class="bk-footer" id="bk-footer"></div>
</div>`}
<script>
const BREAKDOWNS = ${JSON.stringify(
    todayRequests.reduce((m, r) => { m[r.seq] = r.breakdown || []; return m; }, {})
)};
var _sortKey = '';
var _sortDir = -1;
function sortBy(key) {
    if (_sortKey === key) { _sortDir *= -1; } else { _sortKey = key; _sortDir = -1; }
    var tbody = document.querySelector('table:not(#bk-table) tbody');
    if (!tbody) return;
    var allRows = Array.from(tbody.querySelectorAll('tr'));
    var units = [];
    allRows.forEach(function(tr) {
        var ps = tr.dataset.parentSeq;
        if (ps !== undefined) {
            var parent = units.find(function(u) { return String(u.leader.dataset.seq) === ps; });
            if (parent) { parent.children.push(tr); return; }
        }
        units.push({ leader: tr, children: [] });
    });
    units.sort(function(a, b) {
        var av = parseInt(a.leader.dataset[key] || 0, 10);
        var bv = parseInt(b.leader.dataset[key] || 0, 10);
        if (av !== bv) return (av - bv) * _sortDir;
        return (parseInt(b.leader.dataset.seq, 10) - parseInt(a.leader.dataset.seq, 10));
    });
    units.forEach(function(u) {
        tbody.appendChild(u.leader);
        u.children.forEach(function(c) { tbody.appendChild(c); });
    });
    ['seq', 'time'].forEach(function(k) {
        var el = document.getElementById('si-' + k);
        if (el) el.textContent = k === _sortKey ? (_sortDir === -1 ? '\u25bc' : '\u25b2') : '';
    });
}
function toggleGroup(gid) {
    var children = document.querySelectorAll('.group-' + gid);
    var toggle = document.querySelector('[data-gid="' + gid + '"]');
    if (!children.length) return;
    var isVisible = window.getComputedStyle(children[0]).display !== 'none';
    children.forEach(function(el) { el.style.display = isVisible ? 'none' : 'table-row'; });
    if (toggle) toggle.textContent = isVisible ? '\u25b6' : '\u25bc';
}
function rowClick(event, tr, gid) {
    if (gid !== undefined) toggleGroup(gid);
    var seq = parseInt(tr.dataset.seq, 10);
    var ptokens = parseInt(tr.dataset.ptokens, 10);
    showBreakdown(seq, ptokens);
}
function showBreakdown(seq, promptTokens) {
    var items = BREAKDOWNS[seq];
    if (!items || !items.length) {
        document.getElementById('breakdown-panel').style.display = 'none';
        return;
    }
    document.getElementById('bk-seq').textContent = '#' + (seq + 1);
    var totalEst = 0;
    var bodyHtml = items.map(function(item) {
        var est = Math.round(item.charCount / 4);
        totalEst += est;
        var pct = promptTokens > 0 ? (est / promptTokens * 100) : 0;
        var cls = pct > 40 ? 'bk-red' : pct > 20 ? 'bk-amber' : '';
        var barPct = Math.min(pct, 100).toFixed(1);
        return '<tr class="' + cls + '"><td>' + item.label + '</td>' +
            '<td class="bk-num">' + est.toLocaleString() + '</td>' +
            '<td class="bk-pct">' + pct.toFixed(1) + '%</td>' +
            '<td class="bk-bar-cell"><div class="bk-bar-bg"><div class="bk-bar-fill" style="width:' + barPct + '%"></div></div></td></tr>';
    }).join('');
    // Conversation history: the gap between billed tokens and what renderedUserMessage accounts for.
    // This is the dominant cost on long sessions — all prior turns sent to the model on every request.
    var history = promptTokens - totalEst;
    if (history > 0) {
        var histPct = (history / promptTokens * 100);
        var histCls = histPct > 40 ? 'bk-red' : histPct > 20 ? 'bk-amber' : '';
        var histBarPct = Math.min(histPct, 100).toFixed(1);
        bodyHtml += '<tr class="' + histCls + ' bk-history"><td><em>Conversation history (prior turns)</em></td>' +
            '<td class="bk-num">' + history.toLocaleString() + '</td>' +
            '<td class="bk-pct">' + histPct.toFixed(1) + '%</td>' +
            '<td class="bk-bar-cell"><div class="bk-bar-bg"><div class="bk-bar-fill" style="width:' + histBarPct + '%"></div></div></td></tr>';
    }
    document.getElementById('bk-body').innerHTML = bodyHtml;
    document.getElementById('bk-footer').textContent =
        'The IN column (' + promptTokens.toLocaleString() + ' tokens) is the full context sent to the model. ' +
        'Conversation history is billed tokens minus what renderedUserMessage accounts for (' + totalEst.toLocaleString() + ' tokens estimated).';
    document.getElementById('breakdown-panel').style.display = 'block';
}
function closeBreakdown() {
    document.getElementById('breakdown-panel').style.display = 'none';
}
sortBy('time');
const _vscode = acquireVsCodeApi();
var _datePicker = document.getElementById('date-picker');
if (_datePicker) {
    _datePicker.addEventListener('change', function() {
        _vscode.postMessage({ command: 'selectDate', date: this.value });
    });
}
</script>
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
                panel.webview.onDidReceiveMessage(msg => {
                    if (msg.command === 'selectDate' && typeof msg.date === 'string') {
                        if (!/^\d{4}-\d{2}-\d{2}$/.test(msg.date)) return;
                        const d = new Date(msg.date + 'T00:00:00');
                        if (isNaN(d.getTime())) return;
                        const _now = new Date();
                        const _min = new Date(_now.getFullYear(), _now.getMonth(), 1);
                        const _max = new Date(_now.getFullYear(), _now.getMonth() + 1, 0);
                        _max.setHours(23, 59, 59, 999);
                        if (d < _min || d > _max) return;
                        loadHistoryForDate(msg.date);
                        renderPanel();
                    }
                });
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
