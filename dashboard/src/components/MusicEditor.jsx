import { useRef, useState } from 'react';
import { apiJson } from '../lib/api';
import { getApiUrl } from '../config';

export default function MusicEditor({ jobId, clipIndex, assets, setAssets, value, onChange, disabled }) {
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState('');
    const input = useRef(null);
    const selected = assets.find((asset) => asset.id === value?.asset_id);

    const upload = async (event) => {
        const file = event.target.files?.[0];
        event.target.value = '';
        if (!file) return;
        if (file.size > 40 * 1024 * 1024) { setError('Choose music up to 40 MB.'); return; }
        setBusy(true);
        setError('');
        try {
            const form = new FormData();
            form.set('file', file);
            const asset = await apiJson(`/api/clip/${jobId}/${clipIndex}/music/assets`, { method: 'POST', body: form });
            setAssets((previous) => [...previous, asset]);
            onChange({ asset_id: asset.id, volume: 0.15, offset: 0, ducking: true });
        } catch (failure) {
            setError(failure.detail || 'Could not upload music.');
        } finally { setBusy(false); }
    };

    return (
        <section aria-label="Background music" className="border-t border-rule pt-3 space-y-2 text-xs text-ink2">
            <div className="flex items-center justify-between gap-2">
                <p className="eyebrow">Background music</p>
                <button type="button" className="btn-ghost py-1 text-xs" disabled={busy || disabled}
                    onClick={() => input.current?.click()}>{busy ? 'uploading…' : 'upload music'}</button>
                <input ref={input} type="file" accept=".mp3,.wav,.m4a,.ogg,.flac" className="hidden"
                    aria-label="Upload music" onChange={upload} />
            </div>
            {error && <p role="alert" className="text-danger">{error}</p>}
            <select aria-label="Music track" className="input-field" disabled={disabled || busy}
                value={value?.asset_id || ''} onChange={(event) => onChange(event.target.value
                    ? { asset_id: event.target.value, volume: 0.15, offset: 0, ducking: true } : null)}>
                <option value="">No background music</option>
                {assets.map((asset) => <option key={asset.id} value={asset.id}>{asset.name}</option>)}
            </select>
            {value && <>
                <label className="block">Music volume · {Math.round(value.volume * 100)}%
                    <input type="range" min="0" max="1" step="0.01" className="w-full accent-brass"
                        value={value.volume} disabled={disabled} onChange={(event) => onChange({ ...value, volume: Number(event.target.value) })} />
                </label>
                <label className="block">In-point (s)
                    <input type="number" min="0" max={Math.max(0, (selected?.duration || 0) - 0.01)} step="0.01"
                        className="input-field mt-1" value={value.offset} disabled={disabled}
                        onChange={(event) => onChange({ ...value, offset: Number(event.target.value) })} />
                </label>
                <label className="flex gap-2 items-center">
                    <input type="checkbox" className="accent-brass" checked={value.ducking} disabled={disabled}
                        onChange={(event) => onChange({ ...value, ducking: event.target.checked })} />
                    Lower music during speech
                </label>
                {selected && <audio controls preload="metadata" src={getApiUrl(selected.url)} className="w-full" aria-label="Music preview" />}
            </>}
            {assets.filter((asset) => asset.id !== value?.asset_id).map((asset) => (
                <button type="button" key={asset.id} className="block text-xs text-muted hover:text-danger" disabled={disabled || busy}
                    onClick={async () => {
                        setError('');
                        try {
                            await apiJson(`/api/clip/${jobId}/${clipIndex}/assets/music/${asset.id}`, { method: 'DELETE' });
                            setAssets(assets.filter((item) => item.id !== asset.id));
                        } catch (failure) { setError(failure.detail || 'Could not delete this track.'); }
                    }}>Delete unused {asset.name}</button>
            ))}
            <p className="text-muted">Use music you have permission to include. The track loops to fit the clip.</p>
        </section>
    );
}
