import React, { useState, useEffect, useCallback } from 'react';
import { KeyRound, Save, Check, Loader2, Eye, EyeOff, Palette, Plus, X, ShieldAlert } from 'lucide-react';
import { apiFetch } from '../lib/api';

// Free AI providers you can configure straight from the app — no server env
// edits needed. Multiple keys per provider are supported ("add new" rows);
// the gateway tries them all and auto-switches when one is limited.
const PROVIDERS = [
    { id: 'openrouter', label: 'OpenRouter', placeholder: 'sk-or-v1-...', hint: 'auto-discovers FREE models only' },
    { id: 'gemini', label: 'Google AI Studio (Gemini)', placeholder: 'AIzaSy...', hint: 'Gemini free tier' },
    { id: 'groq', label: 'Groq', placeholder: 'gsk_...', hint: 'llama-3.3-70b, fast' },
    { id: 'deepseek', label: 'DeepSeek', placeholder: 'sk-...', hint: 'deepseek-chat / reasoner' },
    { id: 'zhipu', label: 'Zhipu (GLM)', placeholder: '...', hint: 'glm-4.5-air / glm-4-flash' },
    { id: 'dashscope', label: 'Alibaba Qwen', placeholder: 'sk-...', hint: 'qwen-plus / qwen-turbo' },
    { id: 'moonshot', label: 'Moonshot (Kimi)', placeholder: 'sk-...', hint: 'kimi models' },
];

// Must match subtitles.CAPTION_THEMES on the backend.
const CAPTION_THEMES = [
    { id: 'auto', label: 'Default (signature look)', swatch: '#FFE500' },
    { id: 'tiktok', label: 'TikTok', swatch: '#FE2C55' },
    { id: 'reels', label: 'Reels', swatch: '#E1306C' },
    { id: 'shorts', label: 'Shorts Pop', swatch: '#FF0000' },
    { id: 'gold', label: 'Gold Glow', swatch: '#FFD700' },
    { id: 'neon', label: 'Neon', swatch: '#00FF88' },
    { id: 'cyber', label: 'Cyber', swatch: '#00FFFF' },
    { id: 'karaoke', label: 'Karaoke', swatch: '#FF6B6B' },
    { id: 'minimal', label: 'Minimal', swatch: '#FFFFFF' },
    { id: 'beast', label: 'Beast', swatch: '#FFD700' },
    { id: 'boxed', label: 'Boxed', swatch: '#7C3AED' },
    { id: 'classic', label: 'Classic', swatch: '#CCCCCC' },
];

let rowCounter = 0;

export default function ServerSettingsCard({ aiConfigured, aiProviders }) {
    const [rows, setRows] = useState([{ id: 0, provider: 'openrouter', key: '' }]);
    const [captionTheme, setCaptionTheme] = useState('auto');
    const [visible, setVisible] = useState({});
    const [keyCounts, setKeyCounts] = useState({});
    const [providerStatus, setProviderStatus] = useState({});
    const [freeModelCount, setFreeModelCount] = useState(null);
    const [saving, setSaving] = useState(false);
    const [saved, setSaved] = useState(false);
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(true);

    const load = useCallback(async () => {
        try {
            const res = await apiFetch('/api/settings');
            if (!res.ok) throw new Error('settings unavailable');
            const data = await res.json();
            setKeyCounts(data.keyCounts || {});
            setProviderStatus(data.providerStatus || {});
            setCaptionTheme(data.captionTheme || 'auto');
            if (typeof data.openrouterFreeModels === 'number') {
                setFreeModelCount(data.openrouterFreeModels);
            }
            // Prefill one row per configured provider (first key), so the UI
            // reflects what's active and rows can be extended with "add new".
            const counts = data.keyCounts || {};
            const prefilled = Object.keys(counts).map((p, i) => ({
                id: rowCounter++, provider: p, key: '',
            }));
            setRows(prefilled.length ? prefilled : [{ id: rowCounter++, provider: 'openrouter', key: '' }]);
        } catch (e) {
            setError('Could not load server settings — is the backend reachable?');
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => { load(); }, [load]);

    const updateRow = (id, patch) => {
        setRows((rs) => rs.map((r) => (r.id === id ? { ...r, ...patch } : r)));
        setSaved(false);
    };
    const addRow = () => {
        setRows((rs) => [...rs, { id: rowCounter++, provider: 'openrouter', key: '' }]);
        setSaved(false);
    };
    const removeRow = (id) => {
        setRows((rs) => (rs.length > 1 ? rs.filter((r) => r.id !== id) : rs));
        setSaved(false);
    };

    const handleSave = async () => {
        setSaving(true);
        setSaved(false);
        setError('');
        try {
            // Group rows by provider → list of keys. An existing key is kept
            // unless its row is removed (removal sends '' for that provider).
            const keys = {};
            rows.forEach((r) => {
                const val = (r.key || '').trim();
                if (val) {
                    keys[r.provider] = keys[r.provider] || [];
                    keys[r.provider].push(val);
                }
            });
            Object.keys(keyCounts).forEach((p) => {
                if (!keys[p] && !rows.some((r) => r.provider === p && (r.key || '').trim())) {
                    keys[p] = []; // provider row removed → clear it
                }
            });
            const res = await apiFetch('/api/settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ keys, caption_theme: captionTheme }),
            });
            if (!res.ok) throw new Error('save failed');
            const data = await res.json();
            setKeyCounts(data.keyCounts || {});
            setProviderStatus(data.providerStatus || {});
            setSaved(true);
            setRows([{ id: rowCounter++, provider: 'openrouter', key: '' }]);
            setTimeout(() => setSaved(false), 3000);
            // Refresh the app-wide config so the "no AI provider" banner clears.
            window.dispatchEvent(new Event('os-settings-saved'));
        } catch (e) {
            setError(String(e.message || e));
        } finally {
            setSaving(false);
        }
    };

    const toggleVisible = (id) => setVisible((v) => ({ ...v, [id]: !v[id] }));

    const totalKeys = Object.values(keyCounts).reduce((a, b) => a + (b || 0), 0);
    const coolingDown = Object.entries(providerStatus)
        .filter(([, s]) => (s.cooldown || 0) > 0);

    return (
        <div className="card p-4 sm:p-6 mb-6 animate-fade">
            <div className="flex flex-wrap items-center justify-between gap-2 mb-1">
                <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-input bg-paper3 flex items-center justify-center shrink-0">
                        <KeyRound size={16} className="text-brass" />
                    </div>
                    <h2 className="text-base font-medium text-ink lowercase">Free AI keys (server)</h2>
                </div>
                {!loading && (totalKeys > 0 ? (
                    <span className="badge-ok">
                        <Check size={12} /> {totalKeys} key{totalKeys > 1 ? 's' : ''} active
                    </span>
                ) : (
                    <span className="badge-warn">no key set</span>
                ))}
            </div>
            <p className="text-xs text-muted mb-5 leading-relaxed">
                Paste one or more free provider keys — saved <strong>on your server</strong>, works from
                any device (phone included), no Vercel env needed. Add as many as you like with{' '}
                <strong>+ add new</strong>. The gateway <strong>only uses free models</strong>, tries every
                key, and <strong>auto-switches automatically</strong> when one gets rate-limited.
            </p>

            {loading ? (
                <div className="flex items-center gap-2 text-muted text-sm py-4">
                    <Loader2 size={16} className="animate-spin" /> Loading…
                </div>
            ) : (
                <div className="space-y-2.5">
                    {rows.map((r) => {
                        const isSet = (keyCounts[r.provider] || 0) > 0;
                        return (
                            <div key={r.id} className="flex flex-col sm:flex-row sm:items-center gap-2">
                                <div className="sm:w-44 shrink-0">
                                    <select
                                        value={r.provider}
                                        onChange={(e) => updateRow(r.id, { provider: e.target.value })}
                                        className="input-field text-sm px-2.5 py-2"
                                    >
                                        {PROVIDERS.map((p) => (
                                            <option key={p.id} value={p.id}>{p.label}</option>
                                        ))}
                                    </select>
                                </div>
                                <div className="relative flex-1">
                                    <input
                                        type={visible[r.id] ? 'text' : 'password'}
                                        value={r.key}
                                        onChange={(e) => updateRow(r.id, { key: e.target.value })}
                                        placeholder={isSet ? `•••••••• (${keyCounts[r.provider]} set — type to add another)` : PROVIDERS.find((p) => p.id === r.provider)?.placeholder}
                                        className={`input-field pr-10 font-mono text-sm ${isSet ? 'border-ok/40' : ''}`}
                                    />
                                    <button
                                        type="button"
                                        onClick={() => toggleVisible(r.id)}
                                        className="absolute right-8 top-1/2 -translate-y-1/2 text-muted hover:text-ink transition-colors"
                                        tabIndex={-1}
                                    >
                                        {visible[r.id] ? <EyeOff size={15} /> : <Eye size={15} />}
                                    </button>
                                    <button
                                        type="button"
                                        onClick={() => removeRow(r.id)}
                                        className="absolute right-2 top-1/2 -translate-y-1/2 text-muted hover:text-danger transition-colors"
                                        title="Remove this key"
                                        tabIndex={-1}
                                    >
                                        <X size={15} />
                                    </button>
                                </div>
                                <div className="sm:w-32 shrink-0 text-right">
                                    {isSet && !(r.key || '').trim() && (
                                        <span className="text-[11px] text-ok">● active</span>
                                    )}
                                    {providerStatus[r.provider]?.cooldown > 0 && (
                                        <span className="text-[11px] text-warn flex items-center gap-1 justify-end">
                                            <ShieldAlert size={11} /> cooling down {Math.ceil(providerStatus[r.provider].cooldown)}s
                                        </span>
                                    )}
                                </div>
                            </div>
                        );
                    })}
                    <button
                        type="button"
                        onClick={addRow}
                        className="w-full py-2.5 rounded-input border border-dashed border-rule2 text-sm text-muted hover:text-ink hover:border-[color:var(--color-accent)] transition-colors flex items-center justify-center gap-2"
                    >
                        <Plus size={15} /> add new
                    </button>
                </div>
            )}

            {/* Auto-switching + free-only status */}
            {(totalKeys > 0 || freeModelCount !== null) && (
                <div className="mt-4 pt-4 border-t border-rule grid grid-cols-1 sm:grid-cols-3 gap-2 text-xs">
                    <div className="bg-paper3 rounded-input px-3 py-2.5">
                        <div className="text-muted mb-0.5">Auto-switching</div>
                        <div className="text-ink2">
                            {coolingDown.length
                                ? `${coolingDown.length} provider(s) cooling down — routing around them`
                                : 'all providers healthy — failover armed'}
                        </div>
                    </div>
                    <div className="bg-paper3 rounded-input px-3 py-2.5">
                        <div className="text-muted mb-0.5">OpenRouter models</div>
                        <div className="text-ink2">
                            {freeModelCount !== null
                                ? `${freeModelCount} free models available — paid models never used`
                                : '—'}
                        </div>
                    </div>
                    <div className="bg-paper3 rounded-input px-3 py-2.5">
                        <div className="text-muted mb-0.5">Fallback order</div>
                        <div className="text-ink2">
                            {(aiProviders && aiProviders.length) ? aiProviders.join(' → ') : '—'}
                        </div>
                    </div>
                </div>
            )}

            {/* Default caption theme for auto-captions on every new clip */}
            <div className="mt-5 pt-5 border-t border-rule">
                <div className="flex items-center gap-2 mb-2">
                    <Palette size={14} className="text-brass" />
                    <p className="text-sm text-ink2">Default caption theme</p>
                    <span className="text-[11px] text-muted">— every new clip burns this look (per-clip themes in the subtitles editor)</span>
                </div>
                <div className="flex flex-wrap gap-1.5">
                    {CAPTION_THEMES.map((t) => (
                        <button
                            key={t.id}
                            type="button"
                            onClick={() => { setCaptionTheme(t.id); setSaved(false); }}
                            className={`px-2.5 py-1.5 rounded-input border text-xs transition-colors flex items-center gap-1.5
                                ${captionTheme === t.id ? 'border-[color:var(--color-accent)] text-ink' : 'border-rule2 text-muted hover:border-[color:var(--color-accent)]'}`}
                            title={t.label}
                        >
                            <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: t.swatch }} />
                            {t.label}
                        </button>
                    ))}
                </div>
            </div>

            {error && <p className="mt-3 text-xs text-warn">{error}</p>}

            <div className="mt-5 flex items-center gap-3">
                <button
                    type="button"
                    onClick={handleSave}
                    disabled={saving}
                    className="btn-primary"
                >
                    {saving ? <Loader2 size={16} className="animate-spin" /> : <Save size={16} />}
                    Save to server
                </button>
                {saved && (
                    <span className="badge-ok animate-fade">
                        <Check size={12} /> Saved & active
                    </span>
                )}
                {aiConfigured && totalKeys === 0 && (
                    <span className="text-xs text-muted">
                        gateway active via env: {aiProviders.join(', ')}
                    </span>
                )}
            </div>
        </div>
    );
}
