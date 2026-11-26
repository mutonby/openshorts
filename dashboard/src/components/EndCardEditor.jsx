import { useRef, useState } from 'react';
import { apiJson } from '../lib/api';
import { getApiUrl } from '../config';

const DEFAULT = { text: 'Follow for more', handle: '', duration: 2, sound: 'chime', asset_id: null };

export default function EndCardEditor({ jobId, clipIndex, value, onChange, assets, setAssets, disabled, preview }) {
    const [uploading, setUploading] = useState(false);
    const [error, setError] = useState('');
    const input = useRef(null);
    const selected = assets.find((item) => item.id === value?.asset_id);
    const upload = async (event) => {
        const file = event.target.files?.[0];
        event.target.value = '';
        if (!file) return;
        if (file.size > 10 * 1024 * 1024) { setError('Choose an image up to 10 MB.'); return; }
        setUploading(true);
        setError('');
        try {
            const form = new FormData();
            form.set('file', file);
            const asset = await apiJson(`/api/clip/${jobId}/${clipIndex}/endcard/assets`, { method: 'POST', body: form });
            setAssets((previous) => [...previous, asset]);
            onChange({ ...value, asset_id: asset.id });
        } catch (failure) { setError(failure.detail || 'Could not upload this image.'); }
        finally { setUploading(false); }
    };
    return (
        <section className="border-t border-rule pt-3 space-y-2 text-xs text-ink2" aria-label="End card">
            <label className="flex items-center justify-between">
                <span className="eyebrow">End card</span>
                <input type="checkbox" className="accent-brass" checked={!!value} disabled={disabled}
                    onChange={(event) => onChange(event.target.checked ? { ...DEFAULT } : null)} />
            </label>
            {value && <>
                <label className="block">Message
                    <input className="input-field mt-1" maxLength={96} value={value.text} disabled={disabled}
                        onChange={(event) => onChange({ ...value, text: event.target.value })} />
                </label>
                <label className="block">Channel handle
                    <input className="input-field mt-1" maxLength={40} value={value.handle} disabled={disabled}
                        onChange={(event) => onChange({ ...value, handle: event.target.value })} />
                </label>
                <label className="block">Length
                    <select className="input-field mt-1" value={value.duration} disabled={disabled}
                        onChange={(event) => onChange({ ...value, duration: Number(event.target.value) })}>
                        {[1, 2, 3].map((duration) => <option key={duration} value={duration}>{duration} seconds</option>)}
                    </select>
                </label>
                <label className="flex items-center gap-2">
                    <input type="checkbox" className="accent-brass" checked={value.sound === 'chime'} disabled={disabled}
                        onChange={(event) => onChange({ ...value, sound: event.target.checked ? 'chime' : 'silent' })} />Chime
                </label>
                <select className="input-field" aria-label="End card image" value={value.asset_id || ''} disabled={disabled}
                    onChange={(event) => onChange({ ...value, asset_id: event.target.value || null })}>
                    <option value="">Text card</option>
                    {assets.map((asset) => <option key={asset.id} value={asset.id}>{asset.name}</option>)}
                </select>
                {selected && <img src={getApiUrl(selected.url)} alt="End card artwork" className="max-h-24 mx-auto" />}
                <input ref={input} type="file" accept=".png,.jpg,.jpeg,.gif,.webp" className="hidden"
                    aria-label="Upload end card image" onChange={upload} />
                <button type="button" className="btn-ghost text-xs w-full" disabled={disabled || uploading}
                    onClick={() => input.current?.click()}>{uploading ? 'uploading…' : 'upload image or GIF'}</button>
                <button type="button" className="btn-ghost text-xs w-full" disabled={disabled} onClick={preview}>Preview saved end card</button>
                <p className="text-muted">Appends {value.duration}s to the clip. Re-render to save changes.</p>
            </>}
            {error && <p role="alert" className="text-danger">{error}</p>}
        </section>
    );
}
