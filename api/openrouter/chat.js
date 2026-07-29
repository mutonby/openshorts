// api/openrouter/chat.js
// Vercel Serverless Function (Node 18+)

let roster = [];
let lastRefresh = 0;
let index = 0;
const TTL = 300 * 1000; // 300s default
const RETRY_ATTEMPTS = 3;
const MODELS_ENDPOINT = 'https://api.openrouter.ai/v1/models';
const CHAT_ENDPOINT = 'https://api.openrouter.ai/v1/chat/completions';

function now() { return Date.now(); }

async function fetchModels(apiKey) {
  const res = await fetch(MODELS_ENDPOINT, { headers: { Authorization: `Bearer ${apiKey}` } });
  if (!res.ok) throw new Error(`Failed to fetch models: ${res.status}`);
  const payload = await res.json();
  let models = [];
  if (Array.isArray(payload)) models = payload;
  else if (payload.models) models = payload.models;
  else if (payload.data) models = payload.data;

  // filter heuristically for free/public models
  const filtered = [];
  for (const m of models) {
    const name = m.id || m.name || m.model_id || '';
    const desc = (m.description || '').toLowerCase();
    const tags = m.tags || [];
    const visibility = m.visibility || m.public || m.isPublic || '';
    let isPublic = false;
    if (typeof visibility === 'boolean') isPublic = visibility;
    else if (typeof visibility === 'string') isPublic = ['public','open'].includes(visibility.toLowerCase());

    if (isPublic || desc.includes('free') || desc.includes('open') || tags.some(t => String(t).toLowerCase().includes('free'))) {
      filtered.push({ id: name, raw: m });
    } else if ((m.owner || '').toString().toLowerCase().includes('openrouter')) {
      filtered.push({ id: name, raw: m });
    } else if (name && name.length < 40) {
      filtered.push({ id: name, raw: m });
    }
  }
  return filtered;
}

async function ensureFresh(apiKey) {
  if (now() - lastRefresh > TTL) {
    try {
      const models = await fetchModels(apiKey);
      if (models && models.length) {
        roster = models;
        lastRefresh = now();
        index = 0;
      }
    } catch (e) {
      // ignore errors and keep existing roster
      console.error('fetchModels error', e.message || e);
    }
  }
}

function pickModel() {
  if (!roster || roster.length === 0) return null;
  const m = roster[index % roster.length];
  index = (index + 1) % roster.length;
  return m;
}

export default async function handler(req, res) {
  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) {
    res.status(500).json({ error: 'OPENROUTER_API_KEY not set in environment' });
    return;
  }

  if (req.method !== 'POST') {
    res.setHeader('Allow', 'POST');
    res.status(405).json({ error: 'Method not allowed' });
    return;
  }

  let body;
  try {
    body = req.body;
    if (!body) body = await jsonBody(req);
  } catch (e) {
    res.status(400).json({ error: 'Invalid JSON body' });
    return;
  }

  const messages = body.messages;
  if (!Array.isArray(messages)) {
    res.status(400).json({ error: 'messages must be an array' });
    return;
  }

  await ensureFresh(apiKey);

  let attempt = 0;
  let lastErr = null;
  while (attempt < RETRY_ATTEMPTS) {
    attempt += 1;
    const model = pickModel();
    if (!model) {
      // fallback: call OpenRouter without specifying model (let API choose) or return error
      try {
        const resp = await fetch(CHAT_ENDPOINT, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${apiKey}` },
          body: JSON.stringify({ messages }),
        });
        const data = await resp.json();
        if (!resp.ok) {
          lastErr = data;
          continue;
        }
        res.status(200).json(data);
        return;
      } catch (e) {
        lastErr = e;
        continue;
      }
    }

    try {
      const resp = await fetch(CHAT_ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${apiKey}` },
        body: JSON.stringify({ model: model.id, messages }),
      });
      const data = await resp.json();
      if (!resp.ok) {
        lastErr = data;
        // on 429/5xx, try next model
        if (resp.status === 429 || (resp.status >= 500 && resp.status < 600)) {
          // remove model from roster temporarily
          roster = roster.filter(r => r.id !== model.id);
          continue;
        }
        res.status(resp.status).json(data);
        return;
      }
      res.status(200).json(data);
      return;
    } catch (e) {
      lastErr = e;
      // try next model
      roster = roster.filter(r => r.id !== model.id);
      continue;
    }
  }

  res.status(502).json({ error: 'OpenRouter request failed', detail: String(lastErr) });
}

// Simple helper to parse body in older Vercel runtime environments
function jsonBody(req) {
  return new Promise((resolve, reject) => {
    let data = '';
    req.on('data', chunk => { data += chunk; });
    req.on('end', () => {
      try { resolve(JSON.parse(data)); } catch (e) { reject(e); }
    });
    req.on('error', reject);
  });
}
