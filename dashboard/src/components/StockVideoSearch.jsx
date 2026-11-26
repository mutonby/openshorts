import { useState } from 'react';
import { apiJson } from '../lib/api';

export default function StockVideoSearch({ jobId, clipIndex, selectedWord, onImport, disabled }) {
    const [query, setQuery] = useState('');
    const [result, setResult] = useState(null);
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState('');
    const endpoint = `/api/clip/${jobId}/${clipIndex}/stock/videos`;
    const search = async (page = 1, terms = query) => {
        if (!terms.trim() || busy) return;
        setQuery(terms);
        setBusy(true);
        setError('');
        try {
            const data = await apiJson(`${endpoint}?query=${encodeURIComponent(terms.trim())}&page=${page}`);
            setResult(data);
        } catch (failure) { setError(failure.detail || 'Could not search stock footage.'); }
        finally { setBusy(false); }
    };
    const add = async (video) => {
        setBusy(true);
        setError('');
        try {
            const asset = await apiJson(endpoint, {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ video_id: video.id }),
            });
            onImport(asset);
        } catch (failure) { setError(failure.detail || 'Could not import this stock video.'); }
        finally { setBusy(false); }
    };
    return (
        <div className="space-y-2 text-xs">
            <label className="block text-muted">Search stock footage
                <input className="input-field mt-1" maxLength={120} value={query} disabled={busy || disabled}
                    placeholder="Search videos" onChange={(event) => { setQuery(event.target.value); setResult(null); }}
                    onKeyDown={(event) => { if (event.key === 'Enter') { event.preventDefault(); search(); } }} />
            </label>
            <div className="flex flex-wrap gap-2">
                <button type="button" className="btn-ghost text-xs py-1" disabled={busy || disabled || !query.trim()}
                    onClick={() => search()}>{busy ? 'working…' : 'search'}</button>
                {selectedWord && <button type="button" className="btn-ghost text-xs py-1" disabled={busy || disabled}
                    onClick={() => search(1, selectedWord.w)}>search selected word</button>}
            </div>
            {error && <p role="alert" className="text-danger">{error}</p>}
            {result?.configured === false && <p className="text-muted">Enable stock search by setting PEXELS_API_KEY on the server.</p>}
            {result?.configured && <>
                <a href="https://www.pexels.com/" target="_blank" rel="noopener noreferrer" className="text-brass underline">Videos provided by Pexels</a>
                {!result.videos.length && <p className="text-muted">No supported videos found. Try another search.</p>}
                <div className="max-h-80 overflow-y-auto space-y-3">
                    {result.videos.map((video) => (
                        <div key={video.id} className="border border-rule rounded-input p-2 space-y-1">
                            <video controls muted playsInline preload="none" poster={video.image} src={video.preview_url}
                                className="w-full max-h-32" aria-label={`Stock footage by ${video.creator}`} />
                            <a href={video.source_url} target="_blank" rel="noopener noreferrer" className="text-brass underline">
                                {video.creator} on Pexels
                            </a>
                            <button type="button" className="btn-ghost w-full text-xs py-1" disabled={busy || disabled}
                                onClick={() => add(video)}>Add {video.duration}s video to assets</button>
                        </div>
                    ))}
                </div>
                <div className="flex justify-between">
                    <button type="button" className="text-brass disabled:opacity-40" disabled={busy || result.page <= 1}
                        onClick={() => search(result.page - 1)}>Previous</button>
                    <button type="button" className="text-brass disabled:opacity-40" disabled={busy || !result.has_more}
                        onClick={() => search(result.page + 1)}>Next</button>
                </div>
            </>}
        </div>
    );
}
