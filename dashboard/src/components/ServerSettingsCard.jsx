import React, { useState, useEffect, useCallback } from 'react';
import { KeyRound, Save, Check, Loader2, Eye, EyeOff, Palette } from 'lucide-react';
import { apiFetch } from '../lib/api';

// Free AI providers you can configure straight from the app — no server env
// edits needed. Keys are stored on the server (server_settings.json) and the
// AI gateway picks them up immediately.
const PROVIDERS = [
    { id: 'openrouter', label: 'OpenRouter', placeholder: 'sk-or-v1-...', hint: 'many :free models — easiest single key' },
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

export default function ServerSettingsCard({ aiConfigured, aiProviders }) {
    const [keys, setKeys] = useState({});
    const [captionTheme, setCaptionTheme] = useState('auto');
    const [configured, setConfigured] = useState([]);
    const [visible, setVisible] = useState({});
    const [saving, setSaving] = useState(false);
    const [saved, setSaved] = useState(false);
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(true);

    const load = useCallback(async () => {
        try {
            const res = await apiFetch('/api/settings');
            if (!res.ok) throw new Error('settings unavailable');
            const data = await res.json();
            setConfigured(data.configuredProviders || []);
            setCaptionTheme(data.captionTheme || 'auto');
        } catch (e) {
            setError('Could not load server settings — is the backend reachable?');
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => { load(); }, [load]);

    const handleSave = async () => {
        setSaving(true);
        setSaved(false);
        setError('');
        try {
            const payload = { keys: {}, caption_theme: captionTheme };
            let touched = false;
            for (const p of PROVIDERS) {
                const val = (keys[p.id] || '').trim();
                if (val) { payload.keys[p.id] = val; touched = true; }
                else if (configured.includes(p.id)) { payload.keys[p.id] = ''; touched = true; }
            }
            const res = await apiFetch('/api/settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            if (!res.ok) throw new Error('save failed');
            const data = await res.json();
            setConfigured(data.configuredProviders || []);
            setSaved(true);
            setKeys({});
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

    return (
        <div className="card p-4 sm:p-6 mb-6 animate-fade">
            <div className="flex flex-wrap items-center justify-between gap-2 mb-1">
                <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-input bg-paper3 flex items-center justify-center shrink-0">
                        <KeyRound size={16} className="text-brass" />
                    </div>
                    <h2 className="text-base font-medium text-ink lowercase">Free AI keys (server)</h2>
                </div>
                {!loading && (configured.length > 0 ? (
                    <span className="badge-ok">
                        <Check size={12} /> {configured.length} provider{configured.length > 1 ? 's' : ''} active
                    </span>
                ) : (
                    <span className="badge-warn">no key set</span>
                ))}
            </div>
            <p className="text-xs text-muted mb-5 leading-relaxed">
                Paste one or more free provider keys here — saved <strong>on your server</strong>,
                works from any device (phone included), no Vercel env needed.
                The gateway falls back across all of them automatically.
            </p>

            {loading ? (
                <div className="flex items-center gap-2 text-muted text-sm py-4">
                    <Loader2 size={16} className="animate-spin" /> Loading…
                </div>
            ) : (
                <div className="space-y-2.5">
                    {PROVIDERS.map((p) => {
                        const isSet = configured.includes(p.id);
                        return (
                            <div key={p.id} className="flex flex-col sm:flex-row sm:items-center gap-2">
                                <div className="sm:w-56 shrink-0">
                                    <div className="flex items-center gap-1.5">
                                        <span className="text-sm text-ink2">{p.label}</span>
                                        {isSet && <Check size={12} className="text-ok shrink-0" />}
                                    </div>
                                    <div className="text-[11px] text-muted">{p.hint}</div>
                                </div>
                                <div className="relative flex-1">
                                    <input
                                        type={visible[p.id] ? 'text' : 'password'}
                                        value={keys[p.id] || ''}
                                        onChange={(e) => setKeys((k) => ({ ...k, [p.id]: e.target.value }))}
                                        placeholder={isSet ? '•••••••• (set — type to replace)' : p.placeholder}
                                        className={`input-field pr-10 font-mono text-sm ${isSet ? 'border-ok/40' : ''}`}
                                    />
                                    <button
                                        type="button"
                                        onClick={() => toggleVisible(p.id)}
                                        className="absolute right-2 top-1/2 -translate-y-1/2 text-muted hover:text-ink transition-colors"
                                        tabIndex={-1}
                                    >
                                        {visible[p.id] ? <EyeOff size={15} /> : <Eye size={15} />}
                                    </button>
                                </div>
                            </div>
                        );
                    })}
                </div>
            )}

            {/* Default caption theme for auto-captions on every new clip */}
            <div className="mt-5 pt-5 border-t border-rule">
                <div className="flex items-center gap-2 mb-2">
                    <Palette size={14} className="text-brass" />
                    <p className="text-sm text-ink2">Default caption theme</p>
                    <span className="text-[11px] text-muted">— every new clip burns this look (per-clip themes still available in the subtitles editor)</span>
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
                {aiConfigured && (
                    <span className="text-xs text-muted">
                        gateway active: {aiProviders.join(', ')}
                    </span>
                )}
            </div>
        </div>
    );
}
