import React, { useState, useEffect, useCallback } from 'react';
import { Youtube, Loader2, Check, ExternalLink, LogOut } from 'lucide-react';
import { apiFetch } from '../lib/api';

const STEPS = [
    'Go to console.cloud.google.com → create a project (free)',
    'APIs & Services → Library → enable "YouTube Data API v3"',
    'APIs & Services → Credentials → Create Credentials → OAuth client ID → Web application',
    'Add this redirect URI: your backend URL + /api/youtube/callback (e.g. https://yourapp.onrender.com/api/youtube/callback)',
    'Copy the Client ID and Client Secret and put them in your backend env vars: GOOGLE_YT_CLIENT_ID and GOOGLE_YT_CLIENT_SECRET (see deploy guide)',
];

export default function YoutubeDirectCard() {
    const [status, setStatus] = useState(null);
    const [loading, setLoading] = useState(true);
    const [connecting, setConnecting] = useState(false);
    const [error, setError] = useState('');
    const [showSteps, setShowSteps] = useState(false);

    const load = useCallback(async () => {
        try {
            const res = await apiFetch('/api/youtube/status');
            if (res.ok) setStatus(await res.json());
        } catch (_) { /* backend down */ }
        finally { setLoading(false); }
    }, []);

    useEffect(() => { load(); }, [load]);

    const connect = async () => {
        setConnecting(true);
        setError('');
        try {
            const res = await apiFetch('/api/youtube/auth-url');
            if (!res.ok) {
                const data = await res.json().catch(() => ({}));
                setError(data.detail?.message || 'YouTube OAuth is not configured on the server.');
                setShowSteps(true);
                return;
            }
            const { url } = await res.json();
            window.location.href = url;
        } catch (e) {
            setError(String(e.message || e));
        } finally {
            setConnecting(false);
        }
    };

    const disconnect = async () => {
        try { await apiFetch('/api/youtube/disconnect', { method: 'POST' }); } catch (_) { /* ignore */ }
        setStatus({ ...(status || {}), connected: false, channelTitle: '', channelId: '' });
    };

    return (
        <div className="card p-4 sm:p-6 mb-6 animate-fade">
            <div className="flex flex-wrap items-center justify-between gap-2 mb-1">
                <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-input bg-paper3 flex items-center justify-center shrink-0">
                        <Youtube size={16} className="text-danger" />
                    </div>
                    <h2 className="text-base font-medium text-ink lowercase">YouTube direct upload</h2>
                </div>
                {!loading && status?.connected && (
                    <span className="badge-ok">
                        <Check size={12} /> {status.channelTitle || 'channel connected'}
                    </span>
                )}
            </div>
            <p className="text-xs text-muted mb-4 leading-relaxed">
                Post clips <strong>straight to your own YouTube channel</strong> with your Google account —
                no third party, no Upload-Post needed. Free API quota (~6 uploads/day). Optional:
                the normal "post" button via Upload-Post still works for TikTok/Instagram/YouTube.
            </p>

            {loading ? (
                <div className="flex items-center gap-2 text-muted text-sm py-3">
                    <Loader2 size={16} className="animate-spin" /> Loading…
                </div>
            ) : status?.configured ? (
                status?.connected ? (
                    <div className="flex flex-wrap items-center gap-3">
                        <span className="text-sm text-ink2">Connected: <strong>{status.channelTitle}</strong></span>
                        <button onClick={disconnect} className="btn-ghost text-xs flex items-center gap-1.5">
                            <LogOut size={13} /> disconnect
                        </button>
                    </div>
                ) : (
                    <div className="flex flex-wrap items-center gap-3">
                        <button onClick={connect} disabled={connecting} className="btn-primary">
                            {connecting ? <Loader2 size={16} className="animate-spin" /> : <Youtube size={16} />}
                            Connect YouTube
                        </button>
                        <span className="text-xs text-muted">You'll be asked to allow OpenShorts+ to upload to your channel.</span>
                    </div>
                )
            ) : (
                <>
                    <div className="flex items-center gap-3">
                        <span className="text-xs text-muted">Not configured on this server yet — it's free and takes ~5 minutes:</span>
                        <button
                            onClick={() => setShowSteps(!showSteps)}
                            className="text-xs text-brass underline underline-offset-2"
                        >
                            {showSteps ? 'hide steps' : 'show steps'}
                        </button>
                    </div>
                    {showSteps && (
                        <ol className="mt-3 space-y-1.5 text-xs text-ink2 bg-paper3 rounded-input p-3 list-decimal list-inside">
                            {STEPS.map((s) => <li key={s}>{s}</li>)}
                        </ol>
                    )}
                </>
            )}
            {error && <p className="mt-3 text-xs text-warn">{error}</p>}
            {status?.connected && (
                <p className="mt-3 text-[11px] text-muted flex items-center gap-1">
                    <ExternalLink size={11} /> YouTube API quota: ~6 uploads/day free — more than enough for shorts.
                </p>
            )}
        </div>
    );
}
