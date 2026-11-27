import { useEffect, useRef, useState } from 'react';
import { apiJson } from '../lib/api';

export default function SourceCredit({ jobId, value, onChange, includesX }) {
    const [savedValue, setSavedValue] = useState(null);
    const [busy, setBusy] = useState(false);
    const [message, setMessage] = useState('');
    const [error, setError] = useState('');
    const request = useRef(null);

    useEffect(() => {
        const controller = new AbortController();
        request.current = controller;
        apiJson(`/api/projects/${jobId}/publishing`, { signal: controller.signal }).then((settings) => {
            if (!controller.signal.aborted) setSavedValue(settings.source_credit_url || '');
        }).catch(() => {
            if (!controller.signal.aborted) {
                setSavedValue('');
                setError('Could not load the saved source credit.');
            }
        });
        return () => controller.abort();
    }, [jobId]);

    const displayedValue = value ?? savedValue ?? '';

    const save = async () => {
        if (value === undefined || savedValue === null || busy) return;
        const signal = request.current.signal;
        setBusy(true);
        setError('');
        setMessage('');
        try {
            const settings = await apiJson(`/api/projects/${jobId}/publishing`, {
                method: 'PUT', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ source_credit_url: value.trim() || null }), signal,
            });
            if (signal.aborted) return;
            setSavedValue(settings.source_credit_url || '');
            onChange(undefined);
            setMessage('Saved for future posts from this project.');
        } catch (failure) {
            if (!signal.aborted) setError(failure.detail || 'Could not save source credit. Enter a full HTTP or HTTPS URL.');
        } finally {
            if (!signal.aborted) setBusy(false);
        }
    };

    return (
        <div className="space-y-2">
            <label className="block text-xs text-ink2">Source credit URL
                <input type="url" className="input-field mt-1" placeholder="https://…" value={displayedValue} disabled={busy}
                    onChange={(event) => { onChange(event.target.value); setMessage(''); }} />
            </label>
            <p className="text-xs text-muted">Appends this link to the post description.</p>
            {includesX && displayedValue.trim() && (
                <p className="text-xs text-muted">Source links on X require Upload-Post's X Links add-on.</p>
            )}
            <button type="button" className="btn-ghost text-xs"
                disabled={busy || savedValue === null || value === undefined} onClick={save}>
                {busy ? 'saving…' : 'save for this project'}
            </button>
            {message && <p role="status" className="text-xs text-muted">{message}</p>}
            {error && <p role="alert" className="text-xs text-danger">{error}</p>}
        </div>
    );
}
