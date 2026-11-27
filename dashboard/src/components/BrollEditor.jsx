import { useRef, useState } from 'react';
import { Film, Loader2, Plus, Trash2, Upload } from 'lucide-react';
import { getApiUrl } from '../config';
import { apiJson } from '../lib/api';
import { MIN_BROLL_DURATION as MIN_DURATION, roundBrollTime as millis, brollCaptionStart } from '../lib/brollTimeline';
import StockVideoSearch from './StockVideoSearch';

const MAX_UPLOAD_BYTES = 40 * 1024 * 1024;

export default function BrollEditor({ jobId, clipIndex, assets, setAssets, layers, setLayers, bottomVideoAssetId, spans,
    maxLayers, segments, total, playhead, selectedWord, seek, disabled }) {
    const [assetId, setAssetId] = useState('');
    const [previewAsset, setPreviewAsset] = useState(null);
    const [uploading, setUploading] = useState(false);
    const [error, setError] = useState(null);
    const input = useRef(null);
    const preview = useRef(null);
    const selected = assets.find((asset) => asset.id === assetId) || assets[0];
    const captionPosition = selectedWord ? brollCaptionStart(segments, selectedWord.s) : null;
    const assetPath = `/api/clip/${jobId}/${clipIndex}/broll/assets`;

    const upload = async (event) => {
        const file = event.target.files?.[0];
        event.target.value = '';
        if (!file) return;
        setError(null);
        if (file.size > MAX_UPLOAD_BYTES) {
            setError('Choose a video up to 40 MB.');
            return;
        }
        const form = new FormData();
        form.set('file', file);
        setUploading(true);
        try {
            const asset = await apiJson(assetPath, { method: 'POST', body: form });
            setAssets((previous) => [...previous, asset]);
            setAssetId(asset.id);
            setPreviewAsset(null);
        } catch (failure) {
            setError(failure.detail || 'Could not upload B-roll.');
        } finally {
            setUploading(false);
        }
    };

    const add = (sourceTime) => {
        const start = sourceTime == null ? Math.min(playhead, Math.max(0, total - MIN_DURATION)) : captionPosition;
        if (!selected || layers.length >= maxLayers || start == null) return;
        const duration = Math.max(MIN_DURATION, Math.min(3, selected.duration, total - start));
        setLayers([...layers, { asset_id: selected.id, start: millis(start), duration: millis(duration), offset: 0,
            source_time: sourceTime }]);
    };

    const setLayer = (index, fields) => setLayers(layers.map((layer, i) => i === index ? { ...layer, ...fields } : layer));
    const deleteAsset = async () => {
        setError(null);
        try {
            await apiJson(`/api/clip/${jobId}/${clipIndex}/assets/broll/${selected.id}`, { method: 'DELETE' });
            setAssets(assets.filter((asset) => asset.id !== selected.id));
            setPreviewAsset(null);
        } catch (failure) { setError(failure.detail || 'Could not delete this asset.'); }
    };

    return (
        <section aria-label="B-roll" className="space-y-2 border-t border-rule pt-3">
            <div className="flex items-center justify-between gap-2">
                <p className="eyebrow">B-roll</p>
                <button type="button" className="btn-ghost flex items-center gap-1.5 text-xs py-1" onClick={() => input.current?.click()}
                    disabled={disabled || uploading}>
                    {uploading ? <Loader2 size={13} className="animate-spin" /> : <Upload size={13} />}
                    {uploading ? 'uploading…' : 'upload video'}
                </button>
                <input ref={input} type="file" accept=".mp4,.mov,.webm" className="hidden" onChange={upload} aria-label="Upload B-roll video" />
            </div>
            {error && <p className="text-danger text-xs" role="alert">{error}</p>}
            <StockVideoSearch jobId={jobId} clipIndex={clipIndex} selectedWord={selectedWord} disabled={disabled || uploading}
                onImport={(asset) => {
                    setAssets((previous) => [...previous, asset]);
                    setAssetId(asset.id);
                    setPreviewAsset(null);
                }} />
            {!assets.length && !uploading && <p className="text-xs text-muted">MP4, MOV or WebM. Up to 40 MB and 60 seconds.</p>}
            {selected && (
                <>
                    <label className="block text-xs text-ink2">
                        <span className="sr-only">B-roll asset</span>
                        <select className="input-field py-1.5" value={selected.id} disabled={disabled}
                            onChange={(event) => { setAssetId(event.target.value); setPreviewAsset(null); }}>
                            {assets.map((asset) => <option value={asset.id} key={asset.id}>{asset.name} · {asset.duration}s</option>)}
                        </select>
                    </label>
                    {selected.attribution && <a href={selected.attribution.source_url} target="_blank" rel="noopener noreferrer"
                        className="block text-xs text-brass underline">{selected.attribution.creator} on Pexels</a>}
                    <div className="flex flex-wrap gap-1.5">
                        <button type="button" className="btn-ghost text-xs py-1" onClick={deleteAsset}
                            disabled={disabled || selected.id === bottomVideoAssetId || layers.some((layer) => layer.asset_id === selected.id)}>delete unused asset</button>
                        <button type="button" className="btn-ghost flex items-center gap-1 py-1 text-xs" onClick={() => add(null)}
                            disabled={disabled || layers.length >= maxLayers}><Plus size={13} /> at playhead</button>
                        <button type="button" className="btn-ghost flex items-center gap-1 py-1 text-xs" onClick={() => add(selectedWord.s)}
                            disabled={disabled || layers.length >= maxLayers || captionPosition == null}><Plus size={13} /> at selected word</button>
                        <button type="button" className="btn-ghost flex items-center gap-1 py-1 text-xs" aria-expanded={!!previewAsset}
                            onClick={() => setPreviewAsset(previewAsset ? null : { asset: selected, offset: 0 })}>
                            <Film size={13} /> {previewAsset ? 'hide preview' : 'preview asset'}
                        </button>
                    </div>
                    {previewAsset && (
                        <video key={`${previewAsset.asset.id}-${previewAsset.offset}`} ref={preview}
                            src={getApiUrl(previewAsset.asset.url)} controls muted playsInline preload="metadata"
                            onLoadedMetadata={() => { if (preview.current) preview.current.currentTime = previewAsset.offset; }}
                            aria-label="B-roll asset preview" className="max-h-40 w-full rounded-input bg-black" />
                    )}
                </>
            )}
            {layers.map((layer, index) => {
                const asset = assets.find((item) => item.id === layer.asset_id);
                const active = spans.find((item) => item.index === index);
                return (
                    <div key={index} className="rounded-input border border-rule p-2 space-y-1.5 text-xs">
                        <div className="flex items-center justify-between gap-2">
                            <button type="button" className="truncate text-left text-ink2 disabled:text-muted" disabled={!active}
                                onClick={() => seek(active.start)}>{asset?.name || 'Unavailable B-roll asset'}</button>
                            <button type="button" className="p-1 rounded-input text-muted hover:text-danger" disabled={disabled}
                                aria-label={`Remove B-roll placement ${index + 1}`} onClick={() => setLayers(layers.filter((_, i) => i !== index))}>
                                <Trash2 size={13} />
                            </button>
                        </div>
                        <div className="grid grid-cols-3 gap-1.5">
                            <label className="text-muted">Start (s)
                                <input type="number" min="0" max={Math.max(0, total - MIN_DURATION)} step="0.1"
                                    className="input-field px-1.5 py-1" value={active?.start ?? layer.start} disabled={disabled}
                                    onChange={(event) => setLayer(index, { start: Math.max(0, Math.min(total, Number(event.target.value))), source_time: null })} />
                            </label>
                            <label className="text-muted">In-point (s)
                                <input type="number" min="0" max={Math.max(0, (asset?.duration || 0) - layer.duration)} step="0.1"
                                    className="input-field px-1.5 py-1" value={layer.offset} disabled={disabled || !asset}
                                    onChange={(event) => setLayer(index, { offset: millis(Math.max(0, Math.min(asset.duration - layer.duration, Number(event.target.value)))) })} />
                            </label>
                            <label className="text-muted">Length (s)
                                <input type="number" min={MIN_DURATION} max={(asset?.duration || 0) - layer.offset} step="0.1"
                                    className="input-field px-1.5 py-1" value={layer.duration} disabled={disabled || !asset}
                                    onChange={(event) => setLayer(index, { duration: millis(Math.max(MIN_DURATION, Math.min(asset.duration - layer.offset, Number(event.target.value)))) })} />
                            </label>
                        </div>
                        <div className="flex flex-wrap items-center justify-between gap-1 text-[11px] text-muted">
                            <span>{!active ? 'Outside the current cut' : layer.source_time == null ? 'Timeline placement' : 'Follows a caption word'}</span>
                            <button type="button" className="text-brass disabled:opacity-40" disabled={disabled || captionPosition == null}
                                onClick={() => setLayer(index, { start: captionPosition, source_time: selectedWord.s })}>link to selected word</button>
                            {asset && <button type="button" className="text-brass"
                                onClick={() => setPreviewAsset({ asset, offset: layer.offset })}>preview from in-point</button>}
                        </div>
                    </div>
                );
            })}
            {!!layers.length && <p className="text-[11px] text-muted">Re-render to preview the new visuals with the clip audio and captions.</p>}
        </section>
    );
}
