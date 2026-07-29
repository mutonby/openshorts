// api/openrouter/health.js
let roster = [];
let lastRefresh = 0;
const TTL = 300 * 1000;
const MODELS_ENDPOINT = 'https://api.openrouter.ai/v1/models';

function now() { return Date.now(); }

async function fetchModels(apiKey) {
  const res = await fetch(MODELS_ENDPOINT, { headers: { Authorization: `Bearer ${apiKey}` } });
  if (!res.ok) throw new Error(`Failed to fetch models: ${res.status}`);
  const payload = await res.json();
  let models = [];
  if (Array.isArray(payload)) models = payload;
  else if (payload.models) models = payload.models;
  else if (payload.data) models = payload.data;

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
      }
    } catch (e) {
      console.error('fetchModels error', e.message || e);
    }
  }
}

export default async function handler(req, res) {
  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) {
    res.status(500).json({ error: 'OPENROUTER_API_KEY not set in environment' });
    return;
  }
  await ensureFresh(apiKey);
  res.status(200).json({ models: roster, lastRefresh });
}
