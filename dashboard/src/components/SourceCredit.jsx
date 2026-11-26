import { useEffect, useState } from 'react';
import { apiJson } from '../lib/api';

export default function SourceCredit({ jobId, open, value, onChange }) {
    const [busy, setBusy] = useState(false);
    const [message, setMessage] = useState('');
    const [error, setError] = useState('');
    useEffect(() => {
        if (!open) return undefined;
        let cancelled = false;
        apiJson(`/api/projects/${jobId}/publishing`).then((settings) => {
            if (!cancelled) onChange(settings.source_credit_url || '');
        }).catch(() => {
            if (!cancelled) setError('Could not load the saved source credit.');
        });
        return () => { cancelled = true; };
    }, [jobId, open, onChange]);

    const save = async () => {
        setBusy(true);
        setError('');
        setMessage('');
        try {
            const settings = await apiJson(`/api/projects/${jobId}/publishing`, {
                method: 'PUT', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ source_credit_url: value?.trim() || null }),
            });
            onChange(settings.source_credit_url || '');
            setMessage('Saved for future posts from this project.');
        } catch { setError('Could not save source credit. Enter a full HTTP or HTTPS URL.'); }
        finally { setBusy(false); }
    };

    return (
        <div className="space-y-2">
            <label className="block text-xs text-ink2">Source credit URL
                <input type="url" className="input-field mt-1" placeholder="https://…" value={value || ''}
                    onChange={(event) => { onChange(event.target.value); setMessage(''); }} />
            </label>
            <p className="text-xs text-muted">Appends this link to the post description.</p>
            <button type="button" className="btn-ghost text-xs" disabled={busy || value === undefined} onClick={save}>
                {busy ? 'saving…' : 'save for this project'}
            </button>
            {message && <p role="status" className="text-xs text-muted">{message}</p>}
            {error && <p role="alert" className="text-xs text-danger">{error}</p>}
        </div>
    );
}
