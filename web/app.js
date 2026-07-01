/**
 * Agent-Loop Console — Frontend App
 * Pure HTML + JS + CSS, no frameworks.
 */

(function () {
  'use strict';

  // ── Navigation ───────────────────────────────────
  const navLinks = document.querySelectorAll('.nav-link');
  const views = document.querySelectorAll('.view');

  navLinks.forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      navLinks.forEach(l => l.classList.remove('active'));
      link.classList.add('active');
      const target = link.dataset.view;
      views.forEach(v => v.classList.remove('active'));
      const view = document.getElementById('view-' + target);
      if (view) {
        view.classList.add('active');
        // Load data for the view
        loadView(target);
      }
      // Update hash without scrolling
      history.replaceState(null, '', '#' + target);
    });
  });

  // Handle initial hash
  const hash = location.hash.replace('#', '') || 'dashboard';
  const initLink = document.querySelector(`[data-view="${hash}"]`);
  if (initLink) initLink.click();

  // ── API helpers ───────────────────────────────────
  async function apiGet(path) {
    const resp = await fetch(path);
    if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`);
    return resp;
  }

  // ── WebSocket ─────────────────────────────────────
  let ws = null;
  const wsStatus = document.getElementById('ws-status');
  const wsStatusText = document.getElementById('ws-status-text');

  function connectWS() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${protocol}//${location.host}/ws/stream`;
    ws = new WebSocket(url);
    ws.onopen = () => {
      wsStatus.className = 'status-dot connected';
      wsStatusText.textContent = 'Connected';
    };
    ws.onclose = () => {
      wsStatus.className = 'status-dot';
      wsStatusText.textContent = 'Disconnected';
      setTimeout(connectWS, 5000);
    };
    ws.onerror = () => {
      wsStatus.className = 'status-dot error';
      wsStatusText.textContent = 'Error';
    };
    ws.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data);
        handleWSMessage(data);
      } catch (e) {
        // ignore parse errors
      }
    };
  }

  function handleWSMessage(data) {
    const messages = document.getElementById('chat-messages');
    if (!messages) return;
    const div = document.createElement('div');
    div.className = 'chat-msg';
    if (data.phase === 'DONE') {
      div.classList.add('done');
      div.textContent = '✅ ' + (data.data?.output || 'Done');
    } else if (data.phase === 'ERROR') {
      div.classList.add('error');
      div.textContent = '❌ ' + (data.data?.error || 'Error');
    } else {
      div.classList.add('phase');
      div.textContent = `[${data.phase}] ${JSON.stringify(data.data || {})}`;
    }
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
  }

  connectWS();

  // ── View loaders ──────────────────────────────────
  async function loadView(name) {
    switch (name) {
      case 'dashboard': return loadDashboard();
      case 'agents': return loadAgents();
      case 'tasks': return loadTasks();
      case 'chat': break;
      case 'traces': return loadTraces();
      case 'metrics': return loadMetrics();
    }
  }

  async function loadDashboard() {
    try {
      const resp = await apiGet('/api/status');
      const data = await resp.json();
      setText('d-phase', data.loop_phase || '-');
      setText('d-active-loops', String(data.active_loops ?? 0));
      setText('d-model', data.model || '-');
      const p = data.pipeline || {};
      setText('d-total', String(p.total ?? 0));
      setText('d-pending', String(p.pending ?? 0));
      setText('d-running', String(p.running ?? 0));
      setText('d-done', String(p.done ?? 0));
      setText('d-failed', String(p.failed ?? 0));
      setText('d-agents-active', String(p.agents_active ?? 0));
      setText('d-agents-idle', String(p.agents_idle ?? 0));
      setText('d-inflight', String(p.inflight ?? 0));
    } catch (e) {
      console.error('Dashboard load failed:', e);
    }
  }

  async function loadAgents() {
    const container = document.getElementById('agents-list');
    try {
      const resp = await apiGet('/api/agents');
      const data = await resp.json();
      const agents = data.agents || [];
      if (agents.length === 0) {
        container.innerHTML = '<p class="empty">No agents found</p>';
        return;
      }
      container.innerHTML = agents.map(a => `
        <div class="list-item">
          <div class="item-header">
            <span class="item-id">${esc(a.agent_id)}</span>
            <span class="item-status status-${esc(a.status)}">${esc(a.status)}</span>
          </div>
          <div class="item-details">
            <span>Role: ${esc(a.role)}</span>
            <span>Tasks: ${a.task_count ?? 0}</span>
            <span>Success: ${a.success_count ?? 0}</span>
            <span>Expertise: ${esc(a.expertise || '-')}</span>
          </div>
        </div>
      `).join('');
    } catch (e) {
      container.innerHTML = `<p class="error">Failed to load: ${esc(e.message)}</p>`;
    }
  }

  async function loadTasks() {
    const container = document.getElementById('tasks-list');
    try {
      const resp = await apiGet('/api/tasks');
      const data = await resp.json();
      const tasks = data.tasks || [];
      if (tasks.length === 0) {
        container.innerHTML = '<p class="empty">No tasks found</p>';
        return;
      }
      container.innerHTML = tasks.map(t => `
        <div class="list-item">
          <div class="item-header">
            <span class="item-id">${esc(t.task_id || '-')}</span>
            <span class="item-status status-${esc(t.status)}">${esc(t.status)}</span>
          </div>
          <div class="item-details">
            ${t.description ? `<span>${esc(t.description).substring(0, 80)}</span>` : ''}
          </div>
        </div>
      `).join('');
    } catch (e) {
      container.innerHTML = `<p class="error">Failed to load: ${esc(e.message)}</p>`;
    }
  }

  async function loadTraces() {
    const container = document.getElementById('traces-list');
    try {
      const resp = await apiGet('/api/traces');
      const data = await resp.json();
      const traces = data.traces || [];
      if (traces.length === 0) {
        container.innerHTML = '<p class="empty">No traces recorded</p>';
        return;
      }
      container.innerHTML = traces.map(t => `
        <div class="list-item trace-item">
          <div class="item-header">
            <span class="item-id">${esc(t.name)}</span>
            <span class="item-status ${t.status === 'ok' ? 'ok' : 'err'}">${esc(t.status)}</span>
          </div>
          <div class="item-details">
            <span>ID: ${esc(t.span_id)}</span>
            <span>Duration: ${(t.duration_ms ?? 0).toFixed(1)}ms</span>
            ${t.parent_id ? `<span>Parent: ${esc(t.parent_id)}</span>` : ''}
          </div>
          ${t.attributes && Object.keys(t.attributes).length ? `
          <div class="trace-attrs">
            ${Object.entries(t.attributes).map(([k, v]) => `<span class="attr">${esc(k)}: ${esc(String(v)).substring(0, 60)}</span>`).join('')}
          </div>` : ''}
        </div>
      `).join('');
    } catch (e) {
      container.innerHTML = `<p class="error">Failed to load: ${esc(e.message)}</p>`;
    }
  }

  async function loadMetrics() {
    const pre = document.getElementById('metrics-content');
    try {
      const resp = await apiGet('/metrics');
      pre.textContent = await resp.text();
    } catch (e) {
      pre.textContent = `Failed to load: ${e.message}`;
    }
  }

  // ── Chat ──────────────────────────────────────────
  document.getElementById('chat-send').addEventListener('click', sendChat);
  document.getElementById('chat-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendChat();
    }
  });

  async function sendChat() {
    const input = document.getElementById('chat-input');
    const message = input.value.trim();
    if (!message) return;

    const messages = document.getElementById('chat-messages');
    const userDiv = document.createElement('div');
    userDiv.className = 'chat-msg user';
    userDiv.textContent = 'You: ' + message;
    messages.appendChild(userDiv);

    input.value = '';
    document.getElementById('chat-send').disabled = true;

    // Try WebSocket first
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({
        message: message,
        thinking: document.getElementById('chat-thinking').checked,
      }));
    }

    // Also try HTTP fallback
    try {
      const resp = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: message,
          thinking: document.getElementById('chat-thinking').checked,
        }),
      });
      const data = await resp.json();
      const botDiv = document.createElement('div');
      botDiv.className = 'chat-msg bot';
      botDiv.textContent = '🤖 ' + (data.output || JSON.stringify(data));
      messages.appendChild(botDiv);
    } catch (e) {
      const errDiv = document.createElement('div');
      errDiv.className = 'chat-msg error';
      errDiv.textContent = '❌ ' + e.message;
      messages.appendChild(errDiv);
    }

    document.getElementById('chat-send').disabled = false;
    messages.scrollTop = messages.scrollHeight;
  }

  // ── Helpers ───────────────────────────────────────
  function setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
  }

  function esc(s) {
    if (s == null) return '';
    const div = document.createElement('div');
    div.textContent = String(s);
    return div.innerHTML;
  }

  // ── Auto-refresh dashboard ────────────────────────
  setInterval(() => {
    const active = document.querySelector('.view.active');
    if (active && active.id === 'view-dashboard') {
      loadDashboard();
    }
  }, 5000);
})();
