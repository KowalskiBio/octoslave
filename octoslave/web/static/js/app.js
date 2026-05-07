/**
 * OctoSlave Web UI - Main Application
 */

console.log('[app.js] Module loaded');

import {
  WS_URL, connectWebSocket, sendMsg, applyConfig, populateModelSelects, onConfigUpdated
} from './websocket.js?v=20260429';
import { handleSlashCommand } from './slash-commands.js?v=20260429';
import {
  toggleHistory, browseDir, refreshHistory, refreshFileTree, viewFile,
  uploadFile, removeAttachment, clearChatMessages, appendChatInfo, appendChatError
} from './components.js?v=20260429';
import { scrollToBottom, autoResizeTextarea, renderMarkdown, esc } from './utils.js?v=20260429';

// Export functions to global scope for inline handlers
window.toggleHistory = toggleHistory;
window.browseDir = browseDir;
window.refreshHistory = refreshHistory;
window.refreshFileTree = refreshFileTree;
window.viewFile = viewFile;
window.uploadFile = uploadFile;
window.removeAttachment = removeAttachment;
window.appendChatInfo = appendChatInfo;
window.appendChatError = appendChatError;
window.clearChatMessages = clearChatMessages;
window.loadChat = (id) => { window.loadChatImpl && window.loadChatImpl(id); };
window.deleteChat = (id) => { window.deleteChatImpl && window.deleteChatImpl(id); };

// ──────────────────────────────────────────────────────────────
// Server message handler
// ──────────────────────────────────────────────────────────────
function handleServerMessage(msg) {
  console.log('[app] Received message:', msg.type, msg);

  // Parallel-mode routing: events tagged with a candidate_index belong to a
  // running multi-agent run and should update the live panel rather than the
  // single-conversation chat bubble.
  if (msg.type === 'parallel_start')   { onParallelStart(msg);   return; }
  if (msg.type === 'parallel_progress'){ onParallelProgress(msg);return; }
  if (msg.candidate_index !== undefined && parallelLive.active) {
    onParallelEvent(msg);
    return;
  }

  switch (msg.type) {
    case 'config':        applyConfig(msg.data); break;
    case 'config_updated': onConfigUpdated(msg); break;
    case 'models':        populateModelSelects(msg.list || []); break;
    case 'stream_start':  onStreamStart(); break;
    case 'token':         onToken(msg.text); break;
    case 'stream_end':    onStreamEnd(); break;
    case 'tool_call':     onToolCall(msg.name, msg.summary); break;
    case 'tool_result':   onToolResult(msg.name, msg.ok, msg.preview); break;
    case 'plan':          onPlan(msg.text); break;
    case 'done':          onDone(msg.iterations); break;
    case 'info':          appendChatInfo(msg.text); break;
    case 'error':         onServerError(msg.text); break;
    case 'cleared':       break;
    case 'chat_saved':
      if (msg.id) window.appState.currentChatId = msg.id;
      refreshHistory();
      break;
    case 'chat_loaded': onChatLoaded(msg); break;
    case 'research_start':    onResearchStart(msg); break;
    case 'round_start':       onRoundStart(msg); break;
    case 'round_done':        onRoundDone(msg); break;
    case 'agent_start':       onAgentStart(msg); break;
    case 'agent_done':        onAgentDone(msg); break;
    case 'research_complete':    onResearchComplete(msg); break;
    case 'permission_request':  onPermissionRequest(msg); break;
    case 'parallel_result':     onParallelResult(msg); break;
    default: break;
  }
}

// ──────────────────────────────────────────────────────────────
// Parallel run — live progress panel
// ──────────────────────────────────────────────────────────────

const parallelLive = {
  active: false,
  panel: null,        // root <div> in the chat stream
  cards: {},          // { [index]: { card, log, status, action, model, iters } }
  candidates: [],     // metadata pushed in parallel_start
};

function onParallelStart(msg) {
  // Tear down any leftover panel from a previous run
  if (parallelLive.panel && parallelLive.panel.parentElement) {
    parallelLive.panel.remove();
  }
  parallelLive.active = true;
  parallelLive.cards = {};
  parallelLive.candidates = msg.candidates || [];

  const container = document.getElementById('chat-messages');
  const root = document.createElement('div');
  root.className = 'msg msg-assistant parallel-live-msg';

  const cardsHtml = (msg.candidates || []).map(c => {
    const color = c.color || '#fab283';
    return `
      <div class="plive-card plive-running" data-idx="${c.index}" style="--c: ${color}">
        <div class="plive-head">
          <span class="plive-idx">#${c.index}</span>
          <span class="plive-model">${esc(c.model || '')}</span>
          <span class="plive-status">running…</span>
        </div>
        <div class="plive-meta">
          <span class="plive-profile">${esc(c.profile || '')}</span>
          <span class="plive-iter">0 iter</span>
        </div>
        <div class="plive-action plive-action-empty">waiting for first action…</div>
        <div class="plive-log"></div>
      </div>`;
  }).join('');

  root.innerHTML = `
    <div class="msg-bubble plive-bubble">
      <div class="plive-head-row">
        🐙×${(msg.candidates || []).length}
        <span class="plive-strategy">strategy=<strong>${esc(msg.strategy)}</strong></span>
        <span class="plive-overall">running…</span>
      </div>
      <div class="plive-grid">${cardsHtml}</div>
    </div>`;
  container.appendChild(root);

  parallelLive.panel = root;
  Array.from(root.querySelectorAll('.plive-card')).forEach(card => {
    const idx = parseInt(card.dataset.idx);
    parallelLive.cards[idx] = {
      card,
      log: card.querySelector('.plive-log'),
      status: card.querySelector('.plive-status'),
      action: card.querySelector('.plive-action'),
      iter: card.querySelector('.plive-iter'),
    };
  });
  scrollToBottom(container);
}

function onParallelProgress(msg) {
  if (!parallelLive.active) return;
  const slot = parallelLive.cards[msg.index];
  if (!slot) return;

  if (typeof msg.iter === 'number') {
    slot.iter.textContent = msg.iter + ' iter';
  }
  if (msg.action) {
    slot.action.textContent = msg.action;
    slot.action.classList.remove('plive-action-empty');
  }
  if (msg.status === 'done') {
    slot.card.classList.remove('plive-running');
    slot.card.classList.add('plive-done');
    slot.status.textContent = 'done';
  } else if (msg.status === 'error') {
    slot.card.classList.remove('plive-running');
    slot.card.classList.add('plive-failed');
    slot.status.textContent = 'failed';
  } else {
    slot.status.textContent = 'running…';
  }
}

function onParallelEvent(msg) {
  // Append a single line to the candidate's log feed.
  const slot = parallelLive.cards[msg.candidate_index];
  if (!slot) return;
  const line = document.createElement('div');
  line.className = 'plive-line';

  const t = msg.type;
  if (t === 'tool_call') {
    line.innerHTML = `<span class="plive-line-tool">${esc(msg.name || '')}</span> ` +
                     `<span class="plive-line-args">${esc(msg.summary || '')}</span>`;
  } else if (t === 'tool_result') {
    if (!msg.preview) return;  // silent results add no signal
    const cls = msg.ok ? 'ok' : 'fail';
    line.classList.add('plive-line-result', cls);
    line.textContent = (msg.ok ? '✓ ' : '✗ ') + (msg.preview || '').replace(/\n+/g, ' ').slice(0, 140);
  } else if (t === 'info') {
    if (!msg.text) return;
    line.classList.add('plive-line-info');
    line.textContent = 'ℹ ' + msg.text.slice(0, 200);
  } else if (t === 'error') {
    line.classList.add('plive-line-error');
    line.textContent = '✗ ' + (msg.text || '').slice(0, 200);
  } else {
    return;  // skip token / plan / stream noise
  }

  slot.log.appendChild(line);
  // Cap log growth so DOM doesn't balloon during long runs.
  while (slot.log.childElementCount > 60) slot.log.firstChild.remove();
  slot.log.scrollTop = slot.log.scrollHeight;
}

// ──────────────────────────────────────────────────────────────
// Parallel run result panel (final summary, after live run finishes)
// ──────────────────────────────────────────────────────────────

function onParallelResult(msg) {
  setChatRunning(false);
  // Mark the live panel as complete (don't tear it down — it's the run's log).
  if (parallelLive.panel) {
    const overall = parallelLive.panel.querySelector('.plive-overall');
    if (overall) {
      const winner = msg.winner;
      overall.textContent = winner !== null && winner !== undefined
        ? `winner: #${winner}` : 'complete';
      overall.classList.add('plive-overall-done');
    }
    // Highlight the winning card
    if (msg.winner !== null && msg.winner !== undefined) {
      const winSlot = parallelLive.cards[msg.winner];
      if (winSlot) winSlot.card.classList.add('plive-winner');
    }
  }
  parallelLive.active = false;

  const container = document.getElementById('chat-messages');
  const wrap = document.createElement('div');
  wrap.className = 'msg msg-assistant';

  const cards = (msg.candidates || []).map(c => {
    const isWinner = c.index === msg.winner;
    const head = `Candidate ${c.index} · ${esc(c.profile)} · ${esc(c.model || '')}`;
    const status = c.succeeded ? (isWinner ? '🏆 winner' : '✓') : '✗ failed';
    const body = c.succeeded ? esc(c.summary || '') : esc(c.error || 'failed');
    return `
      <div class="parallel-card${isWinner ? ' parallel-winner' : ''}${c.succeeded ? '' : ' parallel-fail'}">
        <div class="parallel-card-head">
          <span class="parallel-card-title">${head}</span>
          <span class="parallel-card-status">${status}</span>
        </div>
        <pre class="parallel-card-body">${body}</pre>
        <div class="parallel-card-foot"><code>${esc(c.workdir || '')}</code></div>
      </div>`;
  }).join('');

  let mergeBlock = '';
  if (msg.strategy === 'merge' && msg.merged_text) {
    mergeBlock = `
      <div class="parallel-merge">
        <div class="parallel-merge-head">📝 Synthesised result</div>
        <div class="parallel-merge-body">${renderMarkdown(msg.merged_text)}</div>
      </div>`;
  }

  wrap.innerHTML = `
    <div class="msg-bubble parallel-bubble">
      <div class="parallel-head">
        🐙×${(msg.candidates || []).length} parallel run · strategy=<strong>${esc(msg.strategy)}</strong>
        ${msg.winner !== null && msg.winner !== undefined ? `· winner=<strong>#${msg.winner}</strong>` : ''}
      </div>
      <div class="parallel-reason">${esc(msg.reason || '')}</div>
      <div class="parallel-grid">${cards}</div>
      ${mergeBlock}
    </div>`;
  container.appendChild(wrap);
  scrollToBottom(container);
}

// ──────────────────────────────────────────────────────────────
// Chat functions
// ──────────────────────────────────────────────────────────────

let currentAssistantBubble = null;
let currentToolCallsDiv = null;
let streamBuffer = '';

function sendChat() {
  const textarea = document.getElementById('chat-textarea');
  const text = textarea.value.trim();
  const hasFiles = window.appState.attachedFiles.length > 0;
  if ((!text && !hasFiles) || window.appState.running) return;

  // Check for slash commands first
  if (text.startsWith('/')) {
    const handled = handleSlashCommand(text);
    if (handled) {
      textarea.value = '';
      autoResizeTextarea(textarea);
      return;  // Don't send as regular message
    }
  }

  let fullText = text;
  if (hasFiles) {
    const paths = window.appState.attachedFiles.map(f => `- ${f.path}`).join('\n');
    fullText += (text ? '\n\n' : '') + `Attached files:\n${paths}`;
  }

  appendUserMessage(fullText);
  textarea.value = '';
  autoResizeTextarea(textarea);
  document.getElementById('chat-attachments').innerHTML = '';
  window.appState.attachedFiles = [];
  setChatRunning(true);

  const model = document.getElementById('chat-model-select').value.trim();
  const dir   = document.getElementById('chat-dir-input').value.trim();
  const profile = document.getElementById('chat-profile-select').value;
  const permMode = document.getElementById('chat-permission-select').value;

  // Parallel mode short-circuit: when the popover toggle is on, route to the
  // multi-agent handler with per-candidate model/profile selections.
  if (window.parallelConfig?.enabled) {
    const cfg = window.parallelConfig;
    const summary = cfg.models?.length
      ? `models=[${cfg.models.join(', ')}]`
      : `model=${model || '(default)'}`;
    appendChatInfo(`🐙×${cfg.n} Spawning ${cfg.n} agents · strategy=${cfg.strategy} · ${summary}`);
    sendMsg({
      type: 'chat_parallel',
      message: fullText,
      n: cfg.n,
      strategy: cfg.strategy,
      models: cfg.models?.length ? cfg.models : undefined,
      profiles: cfg.profiles?.length ? cfg.profiles : undefined,
      judge_model: cfg.judge || undefined,
      model,
      working_dir: dir,
      permission_mode: permMode,
    });
    return;
  }

  const type = window.appState.chatIsFirst ? 'chat' : 'chat_continue';
  window.appState.chatIsFirst = false;

  sendMsg({ type, message: fullText, model, working_dir: dir, prompt_profile: profile, permission_mode: permMode });
}

function appendUserMessage(text) {
  window.appState.messages.push({ role: 'user', content: text });
  const container = document.getElementById('chat-messages');
  const div = document.createElement('div');
  div.className = 'msg msg-user';
  div.innerHTML = `<div class="msg-bubble">${esc(text)}</div>`;
  container.appendChild(div);
  scrollToBottom(container);
}

function ensureAssistantBubble() {
  if (currentAssistantBubble) return;

  const container = document.getElementById('chat-messages');
  const wrap = document.createElement('div');
  wrap.className = 'msg msg-assistant';

  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble';

  const textDiv = document.createElement('div');
  textDiv.className = 'md-content streaming-cursor';
  textDiv.dataset.raw = '';

  const toolsDiv = document.createElement('div');
  toolsDiv.className = 'tool-calls';

  bubble.appendChild(textDiv);
  bubble.appendChild(toolsDiv);
  wrap.appendChild(bubble);
  container.appendChild(wrap);

  currentAssistantBubble = textDiv;
  currentToolCallsDiv    = toolsDiv;
  streamBuffer           = '';
  scrollToBottom(container);
}

function onStreamStart() {
  ensureAssistantBubble();
  if (currentAssistantBubble) {
    currentAssistantBubble.classList.remove('streaming-cursor');
    currentAssistantBubble.classList.add('waiting-for-response');
  }
}

function onToken(text) {
  // Transition from waiting indicator to live streaming on first token
  if (currentAssistantBubble && currentAssistantBubble.classList.contains('waiting-for-response')) {
    currentAssistantBubble.classList.remove('waiting-for-response');
    currentAssistantBubble.classList.add('streaming-cursor');
  }
  ensureAssistantBubble();
  streamBuffer += text;
  currentAssistantBubble.textContent = streamBuffer;
  scrollToBottom(document.getElementById('chat-messages'));
}

function onStreamEnd() {
  if (currentAssistantBubble) {
    currentAssistantBubble.classList.remove('streaming-cursor');
    currentAssistantBubble.classList.remove('waiting-for-response');
    currentAssistantBubble.innerHTML = renderMarkdown(streamBuffer);
  }
  currentAssistantBubble = null;
  currentToolCallsDiv    = null;
}

function onToolCall(name, summary) {
  ensureAssistantBubble();
  
  const icon = globalThis.TOOL_ICONS?.[name] || '🔧';
  const toolBlock = document.createElement('details');
  toolBlock.className = 'tool-block';
  toolBlock.innerHTML = `
    <summary>
      <span class="tool-icon">${icon}</span>
      <span class="tool-name">${name}</span>
      <span class="tool-summary">${esc(summary)}</span>
    </summary>
    <div class="tool-detail pending">Loading...</div>
  `;
  
  currentToolCallsDiv.appendChild(toolBlock);
  scrollToBottom(document.getElementById('chat-messages'));
  
  // Store reference for updating
  window.appState.pendingToolCall = { element: toolBlock, name };
}

function onToolResult(name, ok, preview) {
  if (!window.appState.pendingToolCall) return;
  
  const { element } = window.appState.pendingToolCall;
  const detail = element.querySelector('.tool-detail');
  if (detail) {
    detail.className = `tool-detail ${ok ? 'ok' : 'fail'}`;
    detail.textContent = preview || (ok ? 'Success' : 'Failed');
  }
  
  window.appState.pendingToolCall = null;
}

function onPlan(text) {
  const container = document.getElementById('chat-messages');
  const wrap = document.createElement('div');
  wrap.className = 'msg msg-plan';
  wrap.innerHTML = `
    <div class="plan-card">
      <div class="plan-header">
        <span class="plan-icon">◆</span>
        <span>Plan</span>
      </div>
      <div class="plan-body">${esc(text)}</div>
    </div>`;
  container.appendChild(wrap);
  scrollToBottom(container);
}

function onDone(iterations) {
  setChatRunning(false);
  appendChatInfo(`✓ Done (${iterations} iteration${iterations !== 1 ? 's' : ''})`);
}

function onServerError(text) {
  appendChatError(text);
  window.appState.researchRunning = false;
  setChatRunning(false);
}

function setChatRunning(running) {
  window.appState.running = running;
  const statusBadge = document.getElementById('chat-status');
  const sendBtn = document.getElementById('chat-send-btn');
  const startBtn = document.getElementById('research-start-btn');
  
  if (statusBadge) {
    statusBadge.textContent = running ? 'running' : 'idle';
    statusBadge.className = running ? 'badge badge-running' : 'badge badge-idle';
  }
  
  if (sendBtn) sendBtn.disabled = running;
  if (startBtn) startBtn.disabled = running;
}

// ──────────────────────────────────────────────────────────────
// Chat history
// ──────────────────────────────────────────────────────────────

function onChatLoaded(msg) {
  window.appState.messages = msg.messages || [];
  window.appState.model = msg.model || '';
  
  // Clear and rebuild chat UI
  const container = document.getElementById('chat-messages');
  container.innerHTML = '';
  
  msg.messages.forEach(m => {
    if (m.role === 'user') {
      const div = document.createElement('div');
      div.className = 'msg msg-user';
      div.innerHTML = `<div class="msg-bubble">${esc(m.content)}</div>`;
      container.appendChild(div);
    } else if (m.role === 'assistant') {
      const div = document.createElement('div');
      div.className = 'msg msg-assistant';
      div.innerHTML = `<div class="msg-bubble">${renderMarkdown(m.content)}</div>`;
      container.appendChild(div);
    }
  });
  
  scrollToBottom(container);
}

// ──────────────────────────────────────────────────────────────
// Research functions
// ──────────────────────────────────────────────────────────────

function onResearchStart(msg) {
  window.appState.researchRunning = true;
  setChatRunning(true);
  document.getElementById('pipeline-section').classList.add('show');
  document.getElementById('research-console').innerHTML = '';
  document.getElementById('completion-card').classList.remove('show');
  // Reset all pipeline boxes to pending state for the new run
  document.querySelectorAll('.pipeline-box').forEach(box => {
    box.className = 'pipeline-box pending';
    const m = box.querySelector('.p-model'); if (m) m.textContent = '';
    const e = box.querySelector('.p-elapsed'); if (e) e.textContent = '';
  });
}

function onRoundStart(msg) {
  const label = document.getElementById('round-progress-label');
  const fill = document.getElementById('round-progress-fill');
  if (label) label.textContent = `Round ${msg.round}/${window.appState.researchMaxRounds}`;
  if (fill) fill.style.width = '10%';
  
  appendToConsole(`<span class="console-round">═══ ROUND ${msg.round} ═══</span>`);
}

function onRoundDone(msg) {
  const fill = document.getElementById('round-progress-fill');
  if (fill) fill.style.width = `${Math.min(90, ((msg.round / window.appState.researchMaxRounds) * 100))}%`;
}

function onAgentStart(msg) {
  const box = document.querySelector(`.pipeline-box[data-role="${msg.role}"]`);
  if (box) {
    box.classList.remove('pending');
    box.classList.add('active');
    box.querySelector('.p-model').textContent = msg.model || '';
  }
  
  appendToConsole(`<span class="console-agent">▶ ${msg.role}</span> starting...`);
}

function onAgentDone(msg) {
  const box = document.querySelector(`.pipeline-box[data-role="${msg.role}"]`);
  if (box) {
    box.classList.remove('active');
    box.classList.add('done');
    box.querySelector('.p-elapsed').textContent = msg.elapsed || '';
  }
  
  appendToConsole(`<span class="console-agent">✓ ${msg.role}</span> done in ${msg.elapsed || '?'}`);
}

function onResearchComplete(msg) {
  window.appState.researchRunning = false;
  setChatRunning(false);
  document.getElementById('pipeline-section').classList.remove('show');
  document.getElementById('completion-card').classList.add('show');
  
  const reportPath = msg.report_path || 'research/final_report.html';
  const reportBtn = document.getElementById('comp-report-btn');
  if (reportBtn) reportBtn.href = `/api/files/view/${encodeURIComponent(reportPath)}`;
  
  appendToConsole('<span class="console-success">═════ RESEARCH COMPLETE ═════</span>');
}

function appendToConsole(text) {
  const consoleEl = document.getElementById('research-console');
  if (!consoleEl) return;
  
  const line = document.createElement('div');
  line.innerHTML = text;
  consoleEl.appendChild(line);
  consoleEl.scrollTop = consoleEl.scrollHeight;
}

// ──────────────────────────────────────────────────────────────
// Permission request UI
// ──────────────────────────────────────────────────────────────

function onPermissionRequest(msg) {
  const container = document.getElementById('chat-messages');
  const wrap = document.createElement('div');
  wrap.className = 'msg msg-permission';

  const modeLabel = msg.mode === 'supervised' ? 'Supervised' : 'Controlled';
  wrap.innerHTML = `
    <div class="perm-card">
      <div class="perm-header">
        <span class="perm-icon">⚠</span>
        <span>Permission Required</span>
        <span class="perm-mode-badge">${modeLabel} Mode</span>
      </div>
      <div class="perm-body">
        <span class="perm-tool">${esc(msg.tool)}</span>
        wants to: <strong>${esc(msg.desc)}</strong>
      </div>
      <div class="perm-dir">${esc(msg.working_dir)}</div>
      <div class="perm-actions">
        <button class="perm-btn perm-allow" onclick="window.resolvePermission(this, true)">✓ Allow</button>
        <button class="perm-btn perm-deny"  onclick="window.resolvePermission(this, false)">✗ Deny</button>
      </div>
    </div>`;

  container.appendChild(wrap);
  scrollToBottom(container);
}

window.resolvePermission = function(btn, allow) {
  sendMsg({ type: 'permission_response', allow });
  const actions = btn.closest('.perm-actions');
  if (actions) {
    actions.innerHTML = allow
      ? '<span class="perm-resolved perm-resolved-allow">✓ Allowed</span>'
      : '<span class="perm-resolved perm-resolved-deny">✗ Denied</span>';
  }
};

// ──────────────────────────────────────────────────────────────
// Initialization
// ──────────────────────────────────────────────────────────────

function fetchPromptProfiles() {
  console.log('[app] Fetching prompt profiles...');
  fetch('/api/profiles')
    .then(r => {
      if (!r.ok) {
        console.error('[app] Failed to fetch profiles:', r.status, r.statusText);
        return Promise.reject(new Error(`HTTP ${r.status}`));
      }
      return r.json();
    })
    .then(data => {
      console.log('[app] Profiles received:', data);
      populatePromptProfiles(data.profiles || []);
    })
    .catch(err => {
      console.error('[app] Profile fetch error:', err);
      populatePromptProfiles([]);
    });
}

function populatePromptProfiles(profiles) {
  const sel = document.getElementById('chat-profile-select');
  if (!sel) {
    console.error('[app] chat-profile-select element not found!');
    return;
  }
  const prev = sel.value;
  sel.innerHTML = '';

  if (!profiles || !profiles.length) {
    // No profiles from server, use fallback list
    const fallback = ['base', 'coder', 'analyst', 'biomedic', 'local'];
    fallback.forEach(p => {
      const o = document.createElement('option');
      o.value = p;
      o.textContent = p.charAt(0).toUpperCase() + p.slice(1);
      sel.appendChild(o);
    });
    console.log('[app] Using fallback profiles:', fallback);
  } else {
    profiles.sort();
    profiles.forEach(p => {
      const o = document.createElement('option');
      o.value = p;
      o.textContent = p.charAt(0).toUpperCase() + p.slice(1);
      sel.appendChild(o);
    });
    console.log('[app] Populated profiles from server:', profiles);
  }

  // Restore previous selection if still valid, otherwise fall back to config or first item
  const pref = prev || window.appState?.config?.prompt_profile || '';
  const allValues = Array.from(sel.options).map(o => o.value);
  if (pref && allValues.includes(pref)) {
    sel.value = pref;
    console.log('[app] Restored profile selection:', pref);
  } else if (allValues.length > 0) {
    sel.value = allValues[0];
    console.log('[app] Defaulted to first profile:', allValues[0]);
  }
}

function initApp() {
  // Tab switching
  document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const tab = btn.dataset.tab;
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
      document.getElementById('tab-' + tab).classList.add('active');
      if (tab === 'files') refreshFileTree();
    });
  });

  // Chat input
  const textarea = document.getElementById('chat-textarea');
  if (textarea) {
    textarea.addEventListener('keydown', (e) => {
      if (filePickerVisible() && handlePickerKey(e)) return;
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendChat();
      }
    });

    textarea.addEventListener('input', () => {
      autoResizeTextarea(textarea);
      maybeShowFilePicker(textarea);
    });

    // Close picker when caret leaves the @-token
    textarea.addEventListener('blur', () => setTimeout(hideFilePicker, 150));
  }

  // Expose for slash-commands.js
  window.setChatRunningExternal = setChatRunning;

  // Parallel-agents popover
  initParallelPopover();

  document.getElementById('chat-send-btn')?.addEventListener('click', sendChat);

  document.getElementById('chat-attach-btn')?.addEventListener('click', () => {
    document.getElementById('chat-file-input')?.click();
  });

  document.getElementById('chat-file-input')?.addEventListener('change', async (e) => {
    const files = Array.from(e.target.files);
    for (const file of files) {
      await uploadFile(file);
    }
    e.target.value = '';
  });

  document.getElementById('chat-new-btn')?.addEventListener('click', () => {
    if (window.appState.messages.length > 0 && !window.appState.currentChatId) {
      sendMsg({ type: 'save_chat', chat_id: '' });
    }
    sendMsg({ type: 'chat_clear' });
    clearChatMessages();
    refreshHistory();
  });

  // Backend select change handler — send switch_backend and refresh model list
  document.getElementById('backend-select')?.addEventListener('change', (e) => {
    const backend = e.target.value;
    e.target.dataset.backend = backend;
    const backendNames = { einfra: 'e-INFRA CZ', ollama: 'Local (Ollama)', nim: 'NVIDIA NIM' };
    appendChatInfo(`🔄 Switching to [bold]${backendNames[backend] || backend}[/bold] backend…`);
    sendMsg({ type: 'switch_backend', backend });
    setTimeout(() => sendMsg({ type: 'list_models' }), 600);
  });

  // Model select change handler - update the badge in the sidebar
  document.getElementById('chat-model-select')?.addEventListener('change', (e) => {
    const badge = document.getElementById('model-badge');
    if (badge) badge.textContent = e.target.value || '—';
  });

  // Profile and permission select change handlers
  document.getElementById('chat-profile-select')?.addEventListener('change', (e) => {
    const label = e.target.value ? (e.target.options[e.target.selectedIndex]?.textContent || e.target.value) : 'Default';
    appendChatInfo(`📝 Profile set to [bold]${label}[/bold]. Will apply to next task.`);
  });

  document.getElementById('chat-permission-select')?.addEventListener('change', (e) => {
    const modeNames = { autonomous: 'Autonomous', controlled: 'Controlled', supervised: 'Supervised' };
    appendChatInfo(`🛡️ Permission mode set to [bold]${modeNames[e.target.value]}[/bold]. Will apply to next tool execution.`);
  });

  // File refresh button
  document.getElementById('files-refresh-btn')?.addEventListener('click', refreshFileTree);

  // Settings refresh button
  document.getElementById('settings-refresh-btn')?.addEventListener('click', () => {
    sendMsg({ type: 'get_config' });
  });

  // Research start button
  document.getElementById('research-start-btn')?.addEventListener('click', () => {
    if (window.appState.running) return;
    const topic = document.getElementById('research-topic').value.trim();
    if (!topic) {
      appendChatError('⚠ Research topic is required.');
      return;
    }
    
    const rounds = parseInt(document.getElementById('research-rounds').value) || 3;
    const modelAll = document.getElementById('research-model-select').value || undefined;
    const resume = document.getElementById('research-resume').checked;
    const workingDir = document.getElementById('research-dir-input').value || '.';
    
    window.appState.researchMaxRounds = rounds;
    window.appState.researchDir = workingDir;
    
    sendMsg({ 
      type: 'research', 
      topic, 
      rounds, 
      model_all: modelAll, 
      resume,
      working_dir: workingDir
    });
  });

  // Completion card buttons
  document.getElementById('comp-files-btn')?.addEventListener('click', () => {
    document.querySelector('[data-tab="files"]').click();
  });

  // Fetch available prompt profiles dynamically
  fetchPromptProfiles();

  // History close button
  document.getElementById('history-close')?.addEventListener('click', toggleHistory);

  // Expose load/delete chat functions globally
  window.loadChatImpl = (id) => {
    sendMsg({ type: 'load_chat', chat_id: id });
    toggleHistory();
  };

  window.deleteChatImpl = async (id) => {
    if (!confirm('Delete this chat?')) return;
    try {
      await fetch(`/api/chats/${id}`, { method: 'DELETE' });
      refreshHistory();
    } catch (err) {
      console.error('Failed to delete chat:', err);
    }
  };

  // Initialize WebSocket connection
  connectWebSocket(
    () => {
      // On open - request config and models; reset any stuck running state
      sendMsg({ type: 'get_config' });
      sendMsg({ type: 'list_models' });
      if (window.appState.running) {
        window.appState.running = false;
        setChatRunning(false);
      }
    },
    () => {
      // On close - show error
      appendChatError('Disconnected from server. Reconnecting...');
    },
    handleServerMessage
  );

  console.log('OctoSlave Web UI initialized');
}

// ──────────────────────────────────────────────────────────────
// Parallel-agents popover
// ──────────────────────────────────────────────────────────────

window.parallelConfig = {
  enabled: false,
  n: 3,
  strategy: 'best',
  judge: '',
  models: [],
  profiles: [],
};

const PARALLEL_PROFILES = ['(inherit)', 'base', 'coder', 'analyst', 'biomedic', 'local'];

function modelOptionsHtml(includeBlank = true) {
  const sel = document.getElementById('chat-model-select');
  const opts = sel ? Array.from(sel.options).map(o => o.value).filter(Boolean) : [];
  let html = '';
  if (includeBlank) html += '<option value="">(inherit)</option>';
  opts.forEach(m => {
    html += `<option value="${esc(m)}">${esc(m)}</option>`;
  });
  return html;
}

function refreshParallelRows() {
  const host = document.getElementById('parallel-rows');
  if (!host) return;
  const n = window.parallelConfig.n;
  const cur = window.parallelConfig.models || [];
  const curP = window.parallelConfig.profiles || [];
  let html = '';
  for (let i = 0; i < n; i++) {
    const profOpts = PARALLEL_PROFILES.map(p =>
      `<option value="${p === '(inherit)' ? '' : p}"${(curP[i] || '') === (p === '(inherit)' ? '' : p) ? ' selected' : ''}>${p}</option>`
    ).join('');
    const modelOpts = modelOptionsHtml(true).replace(
      `value="${esc(cur[i] || '')}"`,
      `value="${esc(cur[i] || '')}" selected`
    );
    html += `
      <div class="parallel-cand-row">
        <span class="parallel-cand-idx">#${i}</span>
        <select class="parallel-cand-model" data-idx="${i}">${modelOpts}</select>
        <select class="parallel-cand-profile" data-idx="${i}">${profOpts}</select>
      </div>`;
  }
  host.innerHTML = html;
  // Re-attach listeners
  host.querySelectorAll('.parallel-cand-model').forEach(sel => {
    sel.addEventListener('change', e => {
      const i = parseInt(e.target.dataset.idx);
      window.parallelConfig.models[i] = e.target.value || '';
      // Trim trailing empties so backend gets a tidy array
      while (window.parallelConfig.models.length && !window.parallelConfig.models.at(-1)) {
        window.parallelConfig.models.pop();
      }
    });
  });
  host.querySelectorAll('.parallel-cand-profile').forEach(sel => {
    sel.addEventListener('change', e => {
      const i = parseInt(e.target.dataset.idx);
      window.parallelConfig.profiles[i] = e.target.value || '';
      while (window.parallelConfig.profiles.length && !window.parallelConfig.profiles.at(-1)) {
        window.parallelConfig.profiles.pop();
      }
    });
  });
}

function refreshParallelBadge() {
  const badge = document.getElementById('parallel-count-badge');
  const btn = document.getElementById('chat-parallel-btn');
  if (!badge || !btn) return;
  if (window.parallelConfig.enabled) {
    badge.textContent = String(window.parallelConfig.n);
    badge.style.display = 'inline-flex';
    btn.classList.add('active');
  } else {
    badge.style.display = 'none';
    btn.classList.remove('active');
  }
}

function refreshParallelJudge() {
  const sel = document.getElementById('parallel-judge');
  if (!sel) return;
  sel.innerHTML = modelOptionsHtml(true);
  if (window.parallelConfig.judge) sel.value = window.parallelConfig.judge;
}

function positionParallelPopover() {
  const btn = document.getElementById('chat-parallel-btn');
  const popover = document.getElementById('parallel-popover');
  if (!btn || !popover) return;

  const rect = btn.getBoundingClientRect();
  const margin = 16;
  const vw = window.innerWidth;
  const vh = window.innerHeight;

  // Decide width — already styled, but read it back to clamp left coord
  const width = Math.min(popover.offsetWidth || 480, vw - margin * 2);
  popover.style.width = width + 'px';

  // Available space above and below the button. Prefer above (composer
  // sits at bottom), fall back to below if there's no room.
  const spaceAbove = rect.top - margin;
  const spaceBelow = vh - rect.bottom - margin;
  const naturalHeight = popover.scrollHeight || 480;

  let placeAbove = spaceAbove >= 280 || spaceAbove >= spaceBelow;
  let maxH;
  if (placeAbove) {
    maxH = Math.min(naturalHeight, spaceAbove);
    popover.style.top = Math.max(margin, rect.top - 8 - maxH) + 'px';
    popover.style.bottom = '';
    popover.style.maxHeight = maxH + 'px';
  } else {
    maxH = Math.min(naturalHeight, spaceBelow);
    popover.style.top = (rect.bottom + 8) + 'px';
    popover.style.bottom = '';
    popover.style.maxHeight = maxH + 'px';
  }

  // Horizontal: align left edge with the button, clamp to viewport
  let left = rect.left;
  if (left + width > vw - margin) left = vw - margin - width;
  if (left < margin) left = margin;
  popover.style.left = left + 'px';
  popover.style.right = '';
}

function showParallelPopover() {
  const popover = document.getElementById('parallel-popover');
  if (!popover) return;
  // Backdrop — created on demand so closing one click outside is easy
  let bd = document.getElementById('parallel-popover-backdrop');
  if (!bd) {
    bd = document.createElement('div');
    bd.id = 'parallel-popover-backdrop';
    bd.className = 'parallel-popover-backdrop';
    bd.addEventListener('click', hideParallelPopover);
    document.body.appendChild(bd);
  }
  bd.style.display = 'block';
  popover.style.display = 'flex';
  refreshParallelJudge();
  refreshParallelRows();
  // Two rAFs: first lets the layout settle so scrollHeight is right.
  requestAnimationFrame(() => requestAnimationFrame(positionParallelPopover));
}

function hideParallelPopover() {
  const popover = document.getElementById('parallel-popover');
  const bd = document.getElementById('parallel-popover-backdrop');
  if (popover) popover.style.display = 'none';
  if (bd) bd.remove();
}

function isPopoverVisible() {
  const popover = document.getElementById('parallel-popover');
  return !!popover && popover.style.display !== 'none';
}

function initParallelPopover() {
  const btn = document.getElementById('chat-parallel-btn');
  const popover = document.getElementById('parallel-popover');
  const close = document.getElementById('parallel-popover-close');
  const enable = document.getElementById('parallel-enable');
  const nInp = document.getElementById('parallel-n');
  const strat = document.getElementById('parallel-strategy');
  const judge = document.getElementById('parallel-judge');
  if (!btn || !popover) return;

  btn.addEventListener('click', (e) => {
    e.stopPropagation();
    if (isPopoverVisible()) hideParallelPopover();
    else showParallelPopover();
  });
  close?.addEventListener('click', hideParallelPopover);

  enable?.addEventListener('change', () => {
    window.parallelConfig.enabled = enable.checked;
    refreshParallelBadge();
  });

  nInp?.addEventListener('change', () => {
    let v = parseInt(nInp.value);
    if (isNaN(v) || v < 2) v = 2;
    if (v > 8) v = 8;
    nInp.value = v;
    window.parallelConfig.n = v;
    window.parallelConfig.models = window.parallelConfig.models.slice(0, v);
    window.parallelConfig.profiles = window.parallelConfig.profiles.slice(0, v);
    refreshParallelRows();
    refreshParallelBadge();
    if (isPopoverVisible()) requestAnimationFrame(positionParallelPopover);
  });

  strat?.addEventListener('change', () => {
    window.parallelConfig.strategy = strat.value;
  });

  judge?.addEventListener('change', () => {
    window.parallelConfig.judge = judge.value;
  });

  // Reposition on viewport changes
  window.addEventListener('resize', () => {
    if (isPopoverVisible()) positionParallelPopover();
  });
  window.addEventListener('scroll', () => {
    if (isPopoverVisible()) positionParallelPopover();
  }, true);

  // Close on Escape
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && isPopoverVisible()) hideParallelPopover();
  });

  // Refresh content when the model dropdown changes (keeps per-candidate
  // dropdowns in sync with whatever the user can actually pick).
  document.getElementById('chat-model-select')?.addEventListener('change', () => {
    if (isPopoverVisible()) {
      refreshParallelRows();
      refreshParallelJudge();
    }
  });
}

// ──────────────────────────────────────────────────────────────
// @-file picker
// ──────────────────────────────────────────────────────────────

let pickerEl = null;
let pickerItems = [];
let pickerSelected = 0;
let pickerToken = null;       // {start, end} of the @… token in textarea

function ensurePickerEl() {
  if (pickerEl) return pickerEl;
  pickerEl = document.createElement('div');
  pickerEl.className = 'file-picker';
  pickerEl.style.display = 'none';
  document.body.appendChild(pickerEl);
  return pickerEl;
}

function filePickerVisible() {
  return pickerEl && pickerEl.style.display !== 'none';
}

function hideFilePicker() {
  if (pickerEl) pickerEl.style.display = 'none';
  pickerToken = null;
  pickerItems = [];
}

function handlePickerKey(e) {
  if (!filePickerVisible()) return false;
  if (e.key === 'ArrowDown') {
    e.preventDefault();
    pickerSelected = Math.min(pickerItems.length - 1, pickerSelected + 1);
    renderPicker();
    return true;
  }
  if (e.key === 'ArrowUp') {
    e.preventDefault();
    pickerSelected = Math.max(0, pickerSelected - 1);
    renderPicker();
    return true;
  }
  if (e.key === 'Enter' || e.key === 'Tab') {
    if (pickerItems.length) {
      e.preventDefault();
      acceptPicker(pickerItems[pickerSelected]);
      return true;
    }
  }
  if (e.key === 'Escape') {
    e.preventDefault();
    hideFilePicker();
    return true;
  }
  return false;
}

function renderPicker() {
  if (!pickerEl) return;
  if (!pickerItems.length) {
    pickerEl.innerHTML = '<div class="file-picker-empty">no matches</div>';
    return;
  }
  pickerEl.innerHTML = pickerItems.map((p, i) => {
    const cls = i === pickerSelected ? 'file-picker-item active' : 'file-picker-item';
    return `<div class="${cls}" data-idx="${i}">${esc(p)}</div>`;
  }).join('');
  Array.from(pickerEl.querySelectorAll('.file-picker-item')).forEach(node => {
    node.addEventListener('mousedown', (ev) => {
      ev.preventDefault();
      acceptPicker(pickerItems[parseInt(node.dataset.idx)]);
    });
  });
}

function acceptPicker(path) {
  const ta = document.getElementById('chat-textarea');
  if (!ta || !pickerToken) {
    hideFilePicker();
    return;
  }
  const value = ta.value;
  const before = value.slice(0, pickerToken.start);
  const after = value.slice(pickerToken.end);
  const insert = '@' + path + ' ';
  ta.value = before + insert + after;
  const caret = before.length + insert.length;
  ta.setSelectionRange(caret, caret);
  hideFilePicker();
  ta.focus();
}

function maybeShowFilePicker(ta) {
  const value = ta.value;
  const caret = ta.selectionStart || 0;
  // Walk back from caret to last @ or whitespace
  let i = caret - 1;
  while (i >= 0 && !/\s/.test(value[i]) && value[i] !== '@') i--;
  if (i < 0 || value[i] !== '@') {
    hideFilePicker();
    return;
  }
  // Make sure @ is preceded by whitespace or start-of-input (so emails don't trigger)
  if (i > 0 && !/\s/.test(value[i - 1])) {
    hideFilePicker();
    return;
  }
  pickerToken = { start: i, end: caret };
  const query = value.slice(i + 1, caret);
  const wd = document.getElementById('chat-dir-input')?.value || '.';
  fetch(`/api/picker?working_dir=${encodeURIComponent(wd)}&q=${encodeURIComponent(query)}`)
    .then(r => r.json())
    .then(data => {
      pickerItems = data.items || [];
      pickerSelected = 0;
      const pe = ensurePickerEl();
      const rect = ta.getBoundingClientRect();
      pe.style.left = rect.left + 'px';
      pe.style.bottom = (window.innerHeight - rect.top + 4) + 'px';
      pe.style.minWidth = Math.min(420, rect.width) + 'px';
      pe.style.display = 'block';
      renderPicker();
    })
    .catch(() => hideFilePicker());
}

// Wait for DOM to be ready before initializing
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initApp);
} else {
  initApp();
}
