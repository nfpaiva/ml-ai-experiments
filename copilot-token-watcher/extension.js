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

// ---------------------------------------------------------------------------
// Tool group classifier
// ---------------------------------------------------------------------------

/**
 * Static lookup table for VS Code built-in / extension tools.
 * Key = exact tool name as it appears in <availableDeferredTools>.
 * Value = { group, actionable } where actionable is null for irreducible tools.
 * Any tool name NOT in this table falls to 'VS Code core' (safe default).
 * MCP tools (prefix mcp_<server>_) are auto-classified by server name — no entry needed here.
 */
const TOOL_GROUPS = {
    // ── GitHub Pull Requests extension (vscode-pull-request-github) ──────────
    'github-pull-request_currentActivePullRequest': { group: 'GitHub PR ext',    actionable: 'disable ext when not reviewing PRs' },
    'github-pull-request_issue_fetch':              { group: 'GitHub PR ext',    actionable: 'disable ext when not reviewing PRs' },
    'github-pull-request_doSearch':                 { group: 'GitHub PR ext',    actionable: 'disable ext when not reviewing PRs' },
    'github-pull-request_labels_fetch':             { group: 'GitHub PR ext',    actionable: 'disable ext when not reviewing PRs' },
    'github-pull-request_notification_fetch':       { group: 'GitHub PR ext',    actionable: 'disable ext when not reviewing PRs' },
    'github-pull-request_pullRequestInViewport':    { group: 'GitHub PR ext',    actionable: 'disable ext when not reviewing PRs' },
    'github-pull-request_pullRequestStatusChecks':  { group: 'GitHub PR ext',    actionable: 'disable ext when not reviewing PRs' },
    'github-pull-request_resolveReviewThread':      { group: 'GitHub PR ext',    actionable: 'disable ext when not reviewing PRs' },
    'github-pull-request_create_pull_request':      { group: 'GitHub PR ext',    actionable: 'disable ext when not reviewing PRs' },
    'github_repo':                                  { group: 'GitHub PR ext',    actionable: 'disable ext when not reviewing PRs' },
    'github_text_search':                           { group: 'GitHub PR ext',    actionable: 'disable ext when not reviewing PRs' },
    // ── Jupyter / notebook extension ─────────────────────────────────────────
    'copilot_getNotebookSummary':                   { group: 'Jupyter ext',      actionable: 'disable ext when not using notebooks' },
    'configure_non_python_notebook':                { group: 'Jupyter ext',      actionable: 'disable ext when not using notebooks' },
    'configure_python_notebook':                    { group: 'Jupyter ext',      actionable: 'disable ext when not using notebooks' },
    'configure_notebook':                           { group: 'Jupyter ext',      actionable: 'disable ext when not using notebooks' },
    'edit_notebook_file':                           { group: 'Jupyter ext',      actionable: 'disable ext when not using notebooks' },
    'read_notebook_cell_output':                    { group: 'Jupyter ext',      actionable: 'disable ext when not using notebooks' },
    'restart_notebook_kernel':                      { group: 'Jupyter ext',      actionable: 'disable ext when not using notebooks' },
    'run_notebook_cell':                            { group: 'Jupyter ext',      actionable: 'disable ext when not using notebooks' },
    'notebook_install_packages':                    { group: 'Jupyter ext',      actionable: 'disable ext when not using notebooks' },
    'notebook_list_packages':                       { group: 'Jupyter ext',      actionable: 'disable ext when not using notebooks' },
    'create_new_jupyter_notebook':                  { group: 'Jupyter ext',      actionable: 'disable ext when not using notebooks' },
    // ── Browser / Playwright tools ────────────────────────────────────────────
    'open_browser_page':                            { group: 'Browser tools',    actionable: 'disable if not doing web/UI work' },
    'click_element':                                { group: 'Browser tools',    actionable: 'disable if not doing web/UI work' },
    'drag_element':                                 { group: 'Browser tools',    actionable: 'disable if not doing web/UI work' },
    'hover_element':                                { group: 'Browser tools',    actionable: 'disable if not doing web/UI work' },
    'navigate_page':                                { group: 'Browser tools',    actionable: 'disable if not doing web/UI work' },
    'screenshot_page':                              { group: 'Browser tools',    actionable: 'disable if not doing web/UI work' },
    'read_page':                                    { group: 'Browser tools',    actionable: 'disable if not doing web/UI work' },
    'type_in_page':                                 { group: 'Browser tools',    actionable: 'disable if not doing web/UI work' },
    'handle_dialog':                                { group: 'Browser tools',    actionable: 'disable if not doing web/UI work' },
    'run_playwright_code':                          { group: 'Browser tools',    actionable: 'disable if not doing web/UI work' },
    // ── Python environment tools ──────────────────────────────────────────────
    'configure_python_environment':                 { group: 'Python env tools', actionable: 'disable if not doing Python work' },
    'get_python_environment_details':               { group: 'Python env tools', actionable: 'disable if not doing Python work' },
    'get_python_executable_details':                { group: 'Python env tools', actionable: 'disable if not doing Python work' },
    'install_python_packages':                      { group: 'Python env tools', actionable: 'disable if not doing Python work' },
    // ── VS Code core — always present, cannot be removed ─────────────────────
    // Listed here only for documentation. Unknown tools also fall to 'VS Code core'.
    'create_directory':          { group: 'VS Code core', actionable: null },
    'create_new_workspace':      { group: 'VS Code core', actionable: null },
    'create_and_run_task':       { group: 'VS Code core', actionable: null },
    'get_task_output':           { group: 'VS Code core', actionable: null },
    'get_vscode_api':            { group: 'VS Code core', actionable: null },
    'install_extension':         { group: 'VS Code core', actionable: null },
    'resolve_memory_file_uri':   { group: 'VS Code core', actionable: null },
    'run_vscode_command':        { group: 'VS Code core', actionable: null },
    'terminal_last_command':     { group: 'VS Code core', actionable: null },
    'terminal_selection':        { group: 'VS Code core', actionable: null },
    'testFailure':               { group: 'VS Code core', actionable: null },
    'vscode_searchExtensions_internal': { group: 'VS Code core', actionable: null },
};

/**
 * Classifies a tool name into a group label and actionable hint.
 * MCP tools are auto-grouped by server name (mcp_<server>_<tool>).
 * Built-in tools are looked up in TOOL_GROUPS; unknowns fall to 'VS Code core'.
 *
 * @param {string} name - Tool name from availableDeferredTools.
 * @returns {{ group: string, actionable: string|null, isMcp: boolean }}
 */
function classifyTool(name) {
    if (name.startsWith('mcp_')) {
        const rest = name.slice(4);
        const idx = rest.indexOf('_');
        const server = idx >= 0 ? rest.slice(0, idx) : rest;
        return { group: 'MCP: ' + server, actionable: 'Disable "' + server + '" MCP server', isMcp: true };
    }
    const entry = TOOL_GROUPS[name];
    return {
        group: entry ? entry.group : 'VS Code core',
        actionable: entry ? entry.actionable : null,
        isMcp: false
    };
}

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

/** Session title per file (from kind:0 customTitle or kind:1 customTitle update).
 *  Populated on the initial full parse; reused for incremental updates. */
let fileSessionTitle = {};

/** Workspace basename per file (from <workspace_info> in renderedGlobalContext).
 *  Populated on the initial full parse; reused for incremental updates. */
let fileWorkspace = {};

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

        // Pre-pass A: extract session title from this file's chunk.
        // Only needed on the initial full parse (fromByte === 0); incremental chunks
        // won't contain kind:1 customTitle updates (they appear at the top of the file).
        // Two patterns observed in the wild:
        //   - kind:0 snapshot header: j.v.customTitle (string)
        //   - kind:1 live update:     {"kind":1,"k":["customTitle"],"v":"…"}
        // We scan all lines and keep the last match so a kind:1 update wins over kind:0.
        if (fromByte === 0) {
            let title = '';
            for (const rawLine of lines) {
                if (!rawLine.includes('"customTitle"')) continue;
                try {
                    const j = JSON.parse(rawLine);
                    if (j?.kind === 0 && typeof j?.v?.customTitle === 'string') {
                        title = j.v.customTitle;
                    } else if (j?.kind === 1 && j?.k?.[0] === 'customTitle' && typeof j?.v === 'string') {
                        title = j.v;
                    }
                } catch { /* skip */ }
            }
            fileSessionTitle[filePath] = title;
        }

        // Pre-pass B: extract workspace basename from the first billing event's
        // renderedGlobalContext. Only needed on the initial full parse.
        // Three JSONL schema variants observed:
        //   1. Synthetic/expanded per-request:  j.v.metadata.renderedGlobalContext
        //   2. Snapshot kind:0 with requests[]: j.v.requests[i].result.metadata.renderedGlobalContext
        //   3. Live kind:2 with v as array:     j.v[i].result.metadata.renderedGlobalContext
        if (fromByte === 0 && !fileWorkspace[filePath]) {
            for (const rawLine of lines) {
                if (!rawLine.includes('workspace_info') || !rawLine.includes('promptTokens')) continue;
                try {
                    const j = JSON.parse(rawLine);
                    let gc =
                        // Variant 1: synthetic per-request or already-expanded
                        j?.v?.metadata?.renderedGlobalContext
                        // Variant 2: snapshot kind:0 with requests array
                        ?? (() => { if (!Array.isArray(j?.v?.requests)) return undefined;
                                    for (const req of j.v.requests) { const g = req?.result?.metadata?.renderedGlobalContext; if (g) return g; } })()
                        // Variant 3: kind:2 where j.v is an array of updates
                        ?? (() => { if (!Array.isArray(j?.v)) return undefined;
                                    for (const req of j.v) { const g = req?.result?.metadata?.renderedGlobalContext; if (g) return g; } })();
                    if (!gc) continue;
                    const gb = gc.map(x => x?.text || '').join('\n');
                    const wsMatch = gb.match(/<workspace_info>[\s\S]*?-\s+([^\n\\]+)/);
                    if (wsMatch) {
                        fileWorkspace[filePath] = path.basename(wsMatch[1].trim());
                        break;
                    }
                } catch { /* skip */ }
            }
        }

        // Normalize JSONL lines: expand session-header (kind:0 with embedded requests)
        // and kind:2 update lines into synthetic per-request standard-format lines so
        // that both the first pass (kindZeroMessages) and second pass (billing events)
        // treat every billing entry uniformly. The original session-header / kind:2 lines
        // are replaced — not duplicated — to avoid double-counting or index skew.
        //
        // IMPORTANT: kind:0 session-header expansion is only done when there are NO kind:2
        // billing lines in the file. In live/ongoing sessions VS Code writes both a kind:0
        // checkpoint (containing all prior requests) AND kind:2 per-request updates.
        // Expanding both would double-count every billing event. For snapshot-only files
        // (e.g. sessions written entirely as kind:0) there are no kind:2 billing lines,
        // so expansion is the only way to extract billing data.
        const hasKind2Billing = lines.some(line => {
            if (!line.includes('"kind"') || !line.includes('promptTokens')) return false;
            try {
                const j = JSON.parse(line);
                return j?.kind === 2 && Array.isArray(j?.v) &&
                       j.v.some(r => r?.result?.metadata?.promptTokens);
            } catch { return false; }
        });

        const expandedLines = [];
        for (const rawLine of lines) {
            try {
                if (!rawLine.includes('"kind"')) { expandedLines.push(rawLine); continue; }
                const jl = JSON.parse(rawLine);
                if (!hasKind2Billing && jl?.kind === 0 && Array.isArray(jl?.v?.requests) &&
                        jl.v.requests.some(r => r?.result?.metadata?.promptTokens)) {
                    // Snapshot-only session-header: replace with one synthetic line per turn.
                    for (const req of jl.v.requests) {
                        if (!req?.result?.metadata?.promptTokens) continue;
                        expandedLines.push(JSON.stringify({
                            kind: 0,
                            v: { ...req.result, requestTime: req.timestamp ?? req.timeSpentWaiting,
                                 message: req.message, variableData: req.variableData }
                        }));
                    }
                } else if (jl?.kind === 2 && Array.isArray(jl?.v) &&
                        jl.v.some(r => r?.result?.metadata?.promptTokens)) {
                    // kind:2 update: replace with one synthetic per-request line per turn.
                    for (const req of jl.v) {
                        if (!req?.result?.metadata?.promptTokens) continue;
                        expandedLines.push(JSON.stringify({
                            kind: 0,
                            v: { ...req.result, requestTime: req.timestamp,
                                 message: req.message, variableData: req.variableData }
                        }));
                    }
                } else {
                    expandedLines.push(rawLine);
                }
            } catch {
                expandedLines.push(rawLine);
            }
        }

        // First pass: collect kind:0 session header messages, timestamps, and variableData
        // in file order. Handles two formats:
        //   - Session-header: one kind:0 line with v.requests[] containing all turns
        //   - Per-request: one kind:0 line per turn
        // All three arrays stay in sync with billingEventIndex.
        const kindZeroMessages = [];
        const kindZeroTimes = [];
        const kindZeroVariables = []; // promptFile/workspace variables per request turn
        for (const line of expandedLines) {
            try {
                if (!line.includes('"kind"')) continue;
                const j = JSON.parse(line);
                const kind = j?.kind ?? j?.v?.kind;
                if (kind !== 0) continue;
                if (!hasKind2Billing && Array.isArray(j?.v?.requests)) {
                    // Session-header format: one record containing all requests
                    for (const req of j.v.requests) {
                        const text = req?.message?.text ?? null;
                        kindZeroMessages.push(text ? String(text).replace(/\\n/g, ' ').replace(/\s+/g, ' ').trim().substring(0, 60) : null);
                        const rawT = req?.timestamp ?? req?.timeSpentWaiting ?? null;
                        kindZeroTimes.push(rawT ? new Date(rawT) : null);
                        kindZeroVariables.push(req?.variableData?.variables ?? []);
                    }
                } else {
                    // Per-request format: one kind:0 line per turn
                    const text = j?.message?.text
                        ?? j?.v?.message?.text
                        ?? j?.messages?.[0]?.text
                        ?? j?.v?.messages?.[0]?.text
                        ?? null;
                    kindZeroMessages.push(text ? String(text).replace(/\\n/g, ' ').replace(/\s+/g, ' ').trim().substring(0, 60) : null);
                    const rawT = j?.v?.requestTime ?? j?.v?.timestamp ?? j?.v?.time ?? j?.v?.createdAt
                        ?? j?.requestTime ?? j?.timestamp ?? j?.ts ?? j?.time ?? j?.createdAt ?? null;
                    kindZeroTimes.push(rawT ? new Date(rawT) : null);
                    kindZeroVariables.push(
                        j?.v?.variableData?.variables ?? j?.variableData?.variables ?? []
                    );
                }
            } catch { /* skip malformed lines */ }
        }

        let billingEventIndex = 0;

        // Second pre-scan: find the most complete tool list seen anywhere in the session.
        // Tool schemas are loaded for EVERY turn, but <availableDeferredTools> only appears
        // in some events and sometimes only lists deferred tools (not MCP / always-on tools).
        // By finding the richest list once, we can apply it as a fallback to all other turns.
        let sessionBestToolNames = [];
        for (const line of expandedLines) {
            if (!line.includes('promptTokens')) continue;
            try {
                const j = JSON.parse(line);
                const meta = j?.v?.metadata;
                if (!meta?.promptTokens) continue;
                const gb = (meta.renderedGlobalContext || []).map(x => x?.text || '').join('\n');
                const ub = (meta.renderedUserMessage   || []).map(x => x?.text || '').join('\n');
                const m  = (gb || ub).match(/<availableDeferredTools>([\s\S]*?)<\/availableDeferredTools>/);
                if (!m) continue;
                let c = m[1];
                if (!c.includes('\n') && c.includes('\\n')) c = c.replace(/\\n/g, '\n');
                const names = c.trim().split('\n').map(l => l.trim()).filter(Boolean)
                    .filter(l => !l.startsWith('Available'));
                if (names.length > sessionBestToolNames.length) sessionBestToolNames = names;
            } catch { /* skip */ }
        }

        for (const line of expandedLines) {
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
                //
                // renderedUserMessage is an array; real content may be at any index (not always [0]).
                // Concatenate all non-empty text items to ensure we capture all XML sections.
                const allUserMsgTexts = (meta?.renderedUserMessage || [])
                    .map(item => item?.text || '')
                    .filter(Boolean);
                const userMsgBlock = allUserMsgTexts.join('\n');
                const userRequestMatch = userMsgBlock.match(/<userRequest>\s*([\s\S]*?)\s*<\/userRequest>/);
                let prompt;
                let isAgentLoop = false;
                if (userRequestMatch) {
                    prompt = userRequestMatch[1]
                        .replace(/\\n/g, ' ')   // literal backslash-n sequences
                        .replace(/\s+/g, ' ')    // collapse actual newlines / whitespace
                        .trim().substring(0, 60);
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

                // Build context breakdown from both message fields.
                // renderedGlobalContext = structural overhead (workspace, tools, memories) — fixed per turn.
                // renderedUserMessage   = per-turn content (attachments, context, instructions, prompt).
                // Tool JSON schemas are NOT in the JSONL — only names are listed.
                // Schema token cost is estimated as: residual / toolCount × groupCount.
                const breakdown = [];
                const globalMsgBlock = meta?.renderedGlobalContext?.[0]?.text || '';

                // When renderedGlobalContext is absent (Variant B), structural sections are
                // merged into renderedUserMessage. Track chars extracted from userMsgBlock for
                // structural items so the per-turn remainder is computed correctly.
                let structFromUserChars = 0;

                const extractFrom = (block, tag, label, category) => {
                    const m = block.match(new RegExp(`<${tag}(?:[^>]*)>([\\s\\S]*?)<\\/${tag}>`, ''));
                    if (m) breakdown.push({ label, charCount: m[1].length, category });
                };

                // Extract a structural section, preferring globalMsgBlock; when absent,
                // fall back to userMsgBlock and track the consumed chars.
                const extractStructural = (tag, label) => {
                    if (globalMsgBlock) {
                        extractFrom(globalMsgBlock, tag, label, 'structural');
                    } else {
                        const m = userMsgBlock.match(new RegExp(`<${tag}(?:[^>]*)>([\\s\\S]*?)<\\/${tag}>`, ''));
                        if (m) {
                            breakdown.push({ label, charCount: m[1].length, category: 'structural' });
                            structFromUserChars += m[0].length; // full tag including tag overhead
                        }
                    }
                };

                // ── Structural sections ───────────────────────────────────────
                extractStructural('environment_info', 'OS / environment');
                extractStructural('workspace_info',   'Workspace tree');
                extractStructural('userMemory',       'User memory');
                extractStructural('sessionMemory',    'Session memory');
                extractStructural('repoMemory',       'Repo memory');
                extractStructural('modeInstructions', 'Mode instructions');
                extractStructural('instructions',     'Skills / agents');

                // ── renderedUserMessage sections (per-turn) ──────────────────
                extractFrom(userMsgBlock, 'userRequest',         'Your prompt',        'prompt');
                extractFrom(userMsgBlock, 'context',             'Date / context',     'per-turn');
                extractFrom(userMsgBlock, 'editorContext',       'Open file / notebook', 'per-turn');
                extractFrom(userMsgBlock, 'reminderInstructions','Reminder instructions', 'per-turn');

                // Attachments — multiple per message, each with an id attribute
                for (const am of userMsgBlock.matchAll(/<attachment\s+id="([^"]*)"[^>]*>([\s\S]*?)<\/attachment>/g)) {
                    breakdown.push({ label: `File: ${am[1]}`, charCount: am[2].length, category: 'per-turn' });
                }
                // contentReferences array — auto-injected instruction files
                if (Array.isArray(meta.contentReferences)) {
                    for (const ref of meta.contentReferences) {
                        const refLabel = ref.uri ?? ref.name ?? ref.path ?? JSON.stringify(ref);
                        breakdown.push({ label: `Instructions: ${String(refLabel).split('/').pop()}`, charCount: JSON.stringify(ref).length, category: 'per-turn' });
                    }
                }
                // Auto-attached prompt/instruction files from variableData (kind:0).
                for (const variable of (kindZeroVariables[billingEventIndex] ?? [])) {
                    if (variable?.kind === 'promptFile') {
                        const name = String(variable?.name ?? '').replace(/^prompt:/, '');
                        breakdown.push({ label: `Auto-instructions: ${name}`, charCount: 0, category: 'structural', inResidual: true });
                    }
                }

                // ── Tool schema estimation ────────────────────────────────────
                // Full tool JSON schemas are in the API tools[] payload but are NOT written
                // to the JSONL. We use two sources to estimate:
                //   Source A (preferred): <availableDeferredTools> in renderedGlobalContext
                //                        gives the deferred-tool name list with accurate count.
                //   Source B (fallback):  toolCallRounds[].toolCalls gives the tools actually
                //                        CALLED this turn. We use TOOL_GROUPS to expand each
                //                        called tool to its full group size (all group tools are
                //                        always loaded when any one of them is called), and
                //                        apply a calibrated 141 tokens/schema constant.
                //   When neither source provides data: schemaDataAvailable = false, show —.

                // Search both blocks: when globalMsgBlock is absent, availableDeferredTools
                // is embedded in userMsgBlock (Variant B / merged format).
                const toolsSearchBlock = globalMsgBlock || userMsgBlock;
                const toolsBlockM = toolsSearchBlock.match(/<availableDeferredTools>([\s\S]*?)<\/availableDeferredTools>/);
                let toolNames = [];
                let useCalibrated = false; // true when using Source B (fixed tok/schema constant)

                if (toolsBlockM) {
                    // Source A: extract names from the XML block.
                    // Some JSONL events store tool names with literal \n (double-encoded)
                    // instead of real newlines — unescape before splitting.
                    let toolContent = toolsBlockM[1];
                    if (!toolContent.includes('\n') && toolContent.includes('\\n')) {
                        toolContent = toolContent.replace(/\\n/g, '\n');
                    }
                    toolNames = toolContent.trim().split('\n')
                        .map(l => l.trim()).filter(Boolean)
                        .filter(l => !l.startsWith('Available'));
                    useCalibrated = toolNames.length < 3; // too few names = degraded format
                } else if (Array.isArray(meta.toolCallRounds) && meta.toolCallRounds.length) {
                    // Source B: called tools only — known subset
                    const calledNames = [...new Set(
                        meta.toolCallRounds
                            .flatMap(r => (r.toolCalls || []).map(tc => tc.name || tc.function?.name))
                            .filter(Boolean)
                    )];
                    toolNames = calledNames;
                    useCalibrated = true;
                }

                // Session-level fallback: if we couldn't find MCP / extension tools
                // for this turn (they're loaded every turn but only appear in some events),
                // substitute the best tool list seen anywhere in this session.
                // This ensures MCP schemas are counted even on turns that only called
                // VS Code core tools (read_file, grep_search, etc.).
                const hasMcp = toolNames.some(n => n.startsWith('mcp_'));
                if (!hasMcp && sessionBestToolNames.length > toolNames.length) {
                    toolNames = sessionBestToolNames;
                    useCalibrated = true; // calibrated constant; mark as session estimate
                }

                const schemaDataAvailable = toolNames.length > 0;

                // Always use calibrated constant — dividing schemaResidual by tool count
                // inflates per-tool estimates on turns beyond turn 1 because the residual
                // also includes accumulated conversation history.
                const TOK_PER_SCHEMA = 141;
                const tokPerTool = TOK_PER_SCHEMA;

                // Group tools by classifier and accumulate per-group counts.
                // For Source B: expand each called tool to its full TOOL_GROUPS group size
                // (all tools in a group are loaded whenever any one of them appears).
                const toolGroupMap = {}; // { groupLabel: { count, actionable, isMcp } }
                if (useCalibrated) {
                    // Source B: group by name, then use the full known group size as count
                    const seenGroups = new Set();
                    for (const tname of toolNames) {
                        const { group, actionable, isMcp } = classifyTool(tname);
                        if (!seenGroups.has(group)) {
                            seenGroups.add(group);
                            // Full group size: count all entries in TOOL_GROUPS for this group
                            const fullCount = Object.values(TOOL_GROUPS).filter(v => v && v.group === group).length || 1;
                            toolGroupMap[group] = { count: fullCount, actionable, isMcp };
                        }
                    }
                } else {
                    for (const tname of toolNames) {
                        const { group, actionable, isMcp } = classifyTool(tname);
                        if (!toolGroupMap[group]) toolGroupMap[group] = { count: 0, actionable, isMcp };
                        toolGroupMap[group].count++;
                    }
                }
                // Sort: irreducible (VS Code core) last, MCP servers first by size
                const toolGroupEntries = Object.entries(toolGroupMap).sort((a, b) => {
                    if (a[0] === 'VS Code core') return 1;
                    if (b[0] === 'VS Code core') return -1;
                    return b[1].count - a[1].count;
                });
                let reducibleTokens = 0;
                for (const [grp, { count, actionable, isMcp }] of toolGroupEntries) {
                    const estTokens = Math.round(tokPerTool * count);
                    breakdown.push({
                        label: `${grp} (${count} tools)`,
                        charCount: estTokens * 4,
                        category: 'schema',
                        actionable,
                        estimated: true,
                        estimatedFromCalls: useCalibrated,
                        isMcp
                    });
                    if (actionable) reducibleTokens += estTokens;
                }

                // ── Remainder of userMsgBlock not accounted for by tagged sections ──
                // Subtract structural chars extracted from userMsgBlock to avoid double-counting.
                const taggedUserChars = breakdown
                    .filter(b => b.category === 'prompt' || b.category === 'per-turn')
                    .reduce((s, b) => s + b.charCount, 0);
                const userRemainder = userMsgBlock.length - taggedUserChars - structFromUserChars;
                if (userRemainder > 50) breakdown.push({ label: 'Other / untagged', charCount: userRemainder, category: 'per-turn' });

                // ── Residual (always-on tool schemas ± conversation history) ──────────────
                // After all tracked sections + estimated deferred-tool schema cost, the
                // remaining tokens come from:
                //   • Always-on tool schemas (read_file, grep_search, run_in_terminal, etc.)
                //     — these JSON schemas are NOT written to the JSONL, so we can never
                //     extract them directly. They are present on EVERY turn.
                //   • Accumulated conversation history — prior turns replayed in API messages[].
                //     Grows with each new turn; zero on the very first turn of a session.
                // When fromByte === 0 we read from file start, so billingEventIndex is the
                // absolute turn index within this session. Turn 0 has no prior history, so
                // the entire residual is always-on tool schemas on a fresh session.
                const totalAccountedTokens = Math.round(
                    breakdown.reduce((s, b) => s + (b.inResidual ? 0 : b.charCount), 0) / 4
                );
                const historyTokens = Math.max(0, meta.promptTokens - totalAccountedTokens);
                if (historyTokens > Math.round(meta.promptTokens * 0.05)) {
                    const isAbsoluteFirstTurn = fromByte === 0 && billingEventIndex === 0;
                    breakdown.push({
                        label: isAbsoluteFirstTurn
                            ? 'Always-on tool schemas (not in logs)'
                            : 'Always-on tool schemas + prior turns',
                        charCount: historyTokens * 4,
                        category: isAbsoluteFirstTurn ? 'schema' : 'history',
                        estimated: true
                    });
                }

                requests.push({
                    time,
                    prompt,
                    isAgentLoop,
                    model: (meta.resolvedModel || '').replace('claude-', '').replace(/-/g, ' '),
                    promptTokens: meta.promptTokens,
                    outputTokens: meta.outputTokens,
                    credits: creditsMatch ? parseFloat(creditsMatch[1]) : null,
                    creditsLabel: detailsAscii.replace(/\s+/g, ' ').trim(),
                    breakdown,
                    reducibleTokens,
                    schemaDataAvailable,
                    sessionTitle: fileSessionTitle[filePath] || '',
                    workspace: fileWorkspace[filePath] || ''
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
    fileSessionTitle = {};
    fileWorkspace = {};
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
    fileSessionTitle = {};
    fileWorkspace = {};
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

    // Helper: compute REDUCIBLE% cell HTML for a single request row.
    // rp = reducibleTokens / promptTokens — fraction of input that could be
    // eliminated by disabling unused MCP servers / extensions.
    const reducibleCellHtml = (r) => {
        if (!r.promptTokens) return `<td class="reducible red-low">—</td>`;
        if (!r.schemaDataAvailable) {
            return `<td class="reducible red-low" title="Schema data not in JSONL for this session. Open breakdown for per-turn sections.">—</td>`;
        }
        const rp = r.promptTokens > 0 ? (r.reducibleTokens || 0) / r.promptTokens * 100 : 0;
        const cls = rp >= 60 ? 'red-high' : rp >= 30 ? 'red-mid' : 'red-low';
        const warn = rp >= 60
            ? `<span title="⚠ ${rp.toFixed(1)}% of input tokens are from tools/extensions you could disable — open context breakdown for details.">\u26a0</span>\u202f`
            : '';
        return `<td class="reducible ${cls}">${warn}${rp.toFixed(1)}%</td>`;
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
            <td class="session" title="${escapeHtml(leader.sessionTitle)}">${escapeHtml(leader.sessionTitle.substring(0, 40))}${leader.sessionTitle.length > 40 ? '…' : ''}</td>
            <td class="workspace">${escapeHtml(leader.workspace)}</td>
            <td class="model">${escapeHtml(leader.model)}</td>
            <td class="num">${leader.promptTokens.toLocaleString()}</td>
            <td class="num">${leader.outputTokens.toLocaleString()}</td>
            <td class="ratio">${ratio}x</td>
            <td class="credits ${creditClass}">${creditsStr}</td>
            ${reducibleCellHtml(leader)}
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
            <td class="session" title="${escapeHtml(leader.sessionTitle)}">${escapeHtml(leader.sessionTitle.substring(0, 40))}${leader.sessionTitle.length > 40 ? '…' : ''}</td>
            <td class="workspace">${escapeHtml(leader.workspace)}</td>
            <td class="model">${escapeHtml(leader.model)}</td>
            <td class="num">${grpIn.toLocaleString()}</td>
            <td class="num">${grpOut.toLocaleString()}</td>
            <td class="ratio">${grpRatio}x</td>
            <td class="credits ${grpCreditClass}">${grpCreditsStr}</td>
            ${reducibleCellHtml(leader)}
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
            <td class="session" title="${escapeHtml(c.sessionTitle)}">${escapeHtml(c.sessionTitle.substring(0, 40))}${c.sessionTitle.length > 40 ? '…' : ''}</td>
            <td class="workspace">${escapeHtml(c.workspace)}</td>
            <td class="model">${escapeHtml(c.model)}</td>
            <td class="num">${c.promptTokens.toLocaleString()}</td>
            <td class="num">${c.outputTokens.toLocaleString()}</td>
            <td class="ratio">${cRatio}x</td>
            <td class="credits ${cCreditClass}">${cCreditsStr}</td>
            ${reducibleCellHtml(c)}
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
  tr.bk-residual-hint td { color: var(--vscode-descriptionForeground); font-style: italic; }
  .bk-residual-note { font-size: 0.85em; opacity: 0.7; text-align: left !important; padding-left: 0.5em; }
  .bk-footer { font-size: 11px; color: var(--vscode-descriptionForeground); margin-top: 8px; border-top: 1px solid var(--vscode-editorWidget-border, rgba(128,128,128,0.2)); padding-top: 6px; }
  th.sortable { cursor: pointer; user-select: none; white-space: nowrap; }
  th.sortable:hover { color: var(--vscode-editor-foreground); }
  .sort-ind { font-size: 9px; margin-left: 2px; }
  td.hist { font-family: monospace; text-align: right; font-size: 11px; white-space: nowrap; }
  td.hist-low { color: #4caf50; }
  td.hist-mid { color: #ff9800; }
  td.hist-high { color: #f44336; font-weight: 600; }
  td.reducible { font-family: monospace; text-align: right; font-size: 11px; white-space: nowrap; }
  td.red-low  { color: #4caf50; }
  td.red-mid  { color: #ff9800; }
  td.red-high { color: #f44336; font-weight: 600; }
  td.session { max-width: 180px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 11px; color: var(--vscode-descriptionForeground); }
  td.workspace { font-size: 11px; color: var(--vscode-descriptionForeground); white-space: nowrap; }
  /* Context breakdown panel — sections and new rows */
  tr.bk-section-header td { background: var(--vscode-editorGroupHeader-tabsBackground, rgba(128,128,128,0.12)); font-size: 11px; font-weight: 600; letter-spacing: 0.06em; padding: 4px 8px; color: var(--vscode-descriptionForeground); }
  tr.bk-estimated td:first-child { font-style: italic; color: var(--vscode-descriptionForeground); }
  .bk-lever { display: block; font-size: 10px; color: var(--vscode-descriptionForeground); margin-top: 1px; }
  .bk-savings-callout { margin-top: 10px; padding: 8px 12px; background: var(--vscode-editor-infoBackground, rgba(30,120,200,0.12)); border-left: 3px solid var(--vscode-editorInfo-foreground, #4fc3f7); border-radius: 3px; font-size: 12px; line-height: 1.5; }
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
    <th>Prompt</th><th>Session</th><th>Workspace</th><th>Model</th>
    <th style="text-align:right">In</th>
    <th style="text-align:right">Out</th>
    <th style="text-align:right">Ratio</th>
    <th style="text-align:right">Credits</th>
    <th style="text-align:right" title="Estimated % of input tokens from tools/extensions you could disable by turning off unused MCP servers or extensions">Reducible %</th>
  </tr></thead>
  <tbody>${rows}</tbody>
</table>

<div id="breakdown-panel">
  <div class="bk-header">
    <span class="bk-title">Context breakdown &mdash; <span id="bk-seq"></span></span>
    <button class="bk-close" onclick="closeBreakdown()">&#x2715;</button>
  </div>
  <div class="bk-note">Token estimates are approximate (chars &divide; 4). <em>~</em> prefix = estimated from schema residual.</div>
  <table id="bk-table">
    <thead><tr><th>Section</th><th style="text-align:right">Est. tokens</th><th style="text-align:right">% of input</th><th>Usage</th></tr></thead>
    <tbody id="bk-body"></tbody>
  </table>
  <div id="bk-savings"></div>
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

    // Partition items into sections
    var structural = items.filter(function(it) { return it.category === 'structural' && !it.inResidual; });
    var perTurn    = items.filter(function(it) { return it.category === 'prompt' || it.category === 'per-turn'; });
    var schemas    = items.filter(function(it) { return it.category === 'schema'; });
    var history    = items.filter(function(it) { return it.category === 'history'; });
    var inResidual = items.filter(function(it) { return it.inResidual; });

    function escH(s) {
        return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
    }

    function rowHtml(item) {
        var est = Math.round(item.charCount / 4);
        var pct = promptTokens > 0 ? (est / promptTokens * 100) : 0;
        var barPct = Math.min(pct, 100).toFixed(1);
        var estCls = item.estimated ? ' bk-estimated' : '';
        var colorCls = pct > 40 ? ' bk-red' : pct > 20 ? ' bk-amber' : '';
        var lever = item.actionable
            ? '<span class="bk-lever">→ ' + escH(item.actionable) + '</span>'
            : '';
        var prefix = item.estimated
            ? '<em title="' + (item.estimatedFromCalls ? 'Estimated from tool calls in this turn (×141 tok/schema)' : 'Estimated from schema residual') + '">~</em>&thinsp;'
            : '';
        return '<tr class="' + estCls + colorCls + '">' +
            '<td>' + prefix + escH(item.label) + lever + '</td>' +
            '<td class="bk-num">' + est.toLocaleString() + '</td>' +
            '<td class="bk-pct">' + pct.toFixed(1) + '%</td>' +
            '<td class="bk-bar-cell"><div class="bk-bar-bg"><div class="bk-bar-fill" style="width:' + barPct + '%"></div></div></td></tr>';
    }

    function sectionHtml(title, sectionItems, extraRows) {
        if (!sectionItems.length && !extraRows) return '';
        var rows = sectionItems.map(rowHtml).join('') + (extraRows || '');
        return '<tr class="bk-section-header"><td colspan="4">' + title + '</td></tr>' + rows;
    }

    // Structural section
    var structRows = sectionHtml('📐 Structural (fixed per turn)', structural,
        // list inResidual items as hints at the bottom of structural section
        inResidual.map(function(it) {
            return '<tr class="bk-residual-hint"><td><em>' + it.label + '</em></td>' +
                '<td class="bk-num bk-residual-note" colspan="3">size in residual ↓</td></tr>';
        }).join('')
    );

    // Per-turn section
    var perTurnRows = sectionHtml('💬 Per-turn content', perTurn);

    // Schema section
    var schemaRows = sectionHtml('🔧 Tool schemas (estimated)', schemas);

    // Savings callout
    var actionableSchemas = schemas.filter(function(it) { return it.actionable; });
    var calloutHtml = '';
    if (actionableSchemas.length) {
        var savingsTokens = actionableSchemas.reduce(function(s, it) { return s + Math.round(it.charCount / 4); }, 0);
        var savingsPct = promptTokens > 0 ? (savingsTokens / promptTokens * 100) : 0;
        var tipParts = actionableSchemas.slice(0, 3).map(function(it) {
            var pct = (Math.round(it.charCount / 4) / promptTokens * 100).toFixed(1);
            return '<strong>' + escH(it.label) + '</strong> (' + pct + '%)';
        });
        calloutHtml = '<div class="bk-savings-callout">💡 You could save ~<strong>' +
            savingsTokens.toLocaleString() + ' tokens/turn (' + savingsPct.toFixed(1) +
            '%)</strong> by disabling unused tools/extensions: ' + tipParts.join(', ') +
            (actionableSchemas.length > 3 ? ', …' : '') + '.</div>';
    }

    var historyRows = sectionHtml('📜 Conversation history (prior turns)', history);
    var fromCalls = items.some(function(it) { return it.estimatedFromCalls; });
    document.getElementById('bk-body').innerHTML = structRows + perTurnRows + schemaRows + historyRows;
    document.getElementById('bk-savings').innerHTML = calloutHtml;
    document.getElementById('bk-footer').textContent =
        'IN = ' + promptTokens.toLocaleString() + ' tokens total. ' +
        'Structural/per-turn sections: chars÷4 estimate. ' +
        (fromCalls
            ? 'Tool schemas: estimated from called tools × 141 tok/schema (schema list not in JSONL for this session). '
            : 'Tool schemas: estimated from tool list × 141 tok/schema (±15%). ') +
        'Always-on tool schemas (read_file, grep_search, etc.) are never in the JSONL — they appear as residual. Conversation history grows with each turn — start a new chat to reset.';
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
