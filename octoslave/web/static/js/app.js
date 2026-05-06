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
    default: break;
  }
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
  fetch('/api/profiles')
    .then(r => r.ok ? r.json() : Promise.reject())
    .then(data => populatePromptProfiles(data.profiles || []))
    .catch(() => populatePromptProfiles([]));
}

function populatePromptProfiles(profiles) {
  const sel = document.getElementById('chat-profile-select');
  if (!sel) return;
  const prev = sel.value;
  sel.innerHTML = '';
  if (!profiles.length) {
    sel.innerHTML = '<option value="base">base</option>';
    return;
  }
  profiles.forEach(p => {
    const o = document.createElement('option');
    o.value = p;
    o.textContent = p.charAt(0).toUpperCase() + p.slice(1);
    sel.appendChild(o);
  });
  // Restore previous selection if still valid, otherwise fall back to config or first item
  const pref = prev || window.appState?.config?.prompt_profile || '';
  if (pref && profiles.includes(pref)) {
    sel.value = pref;
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
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendChat();
      }
    });
    
    textarea.addEventListener('input', () => autoResizeTextarea(textarea));
  }

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

// Wait for DOM to be ready before initializing
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initApp);
} else {
  initApp();
}
