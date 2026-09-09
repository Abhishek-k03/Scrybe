// Backend origin. Override at runtime by setting window.SCRYBE_API_BASE before this
// file loads; otherwise the deployed Render service below is used.
const API_BASE = (window.SCRYBE_API_BASE || 'https://scrybe-api.onrender.com').replace(/\/$/, '');

async function _json(res) {
  const ct = res.headers.get('content-type') || '';
  const body = ct.includes('application/json') ? await res.json() : { detail: await res.text() };
  if (!res.ok) {
    const msg = body?.detail || `HTTP ${res.status}`;
    throw new Error(typeof msg === 'string' ? msg : JSON.stringify(msg));
  }
  return body;
}

const api = {
  base: API_BASE,

  // Under /api on purpose — a bare /health trips ad-blocker filter lists, which
  // surfaces as ERR_BLOCKED_BY_CLIENT and a permanently "offline" badge.
  async health() { return _json(await fetch(`${API_BASE}/api/health`)); },

  async listSources() { return _json(await fetch(`${API_BASE}/api/sources`)); },

  async deleteSource(sourceId) {
    return _json(await fetch(`${API_BASE}/api/sources/${encodeURIComponent(sourceId)}`, { method: 'DELETE' }));
  },

  async ingestUrl(url) {
    return _json(await fetch(`${API_BASE}/api/ingest/url`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url }),
    }));
  },

  async ingestFile(file) {
    const fd = new FormData();
    fd.append('file', file);
    return _json(await fetch(`${API_BASE}/api/ingest/file`, { method: 'POST', body: fd }));
  },

  async query(question, topK = 5, chatId = null) {
    return _json(await fetch(`${API_BASE}/api/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, top_k: topK, chat_id: chatId }),
    }));
  },

  async chatsStatus() { return _json(await fetch(`${API_BASE}/api/chats/status`)); },
  async listChats() { return _json(await fetch(`${API_BASE}/api/chats`)); },
  async createChat(title = null) {
    return _json(await fetch(`${API_BASE}/api/chats`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title }),
    }));
  },
  async getChat(chatId) {
    return _json(await fetch(`${API_BASE}/api/chats/${encodeURIComponent(chatId)}`));
  },
  async renameChat(chatId, title) {
    return _json(await fetch(`${API_BASE}/api/chats/${encodeURIComponent(chatId)}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title }),
    }));
  },
  async deleteChat(chatId) {
    return _json(await fetch(`${API_BASE}/api/chats/${encodeURIComponent(chatId)}`, { method: 'DELETE' }));
  },

  async vectorMap() { return _json(await fetch(`${API_BASE}/api/vector_map`)); },
  async vectorMapQuery(question, topK = 6) {
    return _json(await fetch(`${API_BASE}/api/vector_map/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, top_k: topK }),
    }));
  },
};

window.api = api;
