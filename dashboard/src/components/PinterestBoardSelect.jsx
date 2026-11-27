import { useEffect, useState } from 'react';
import { apiJson } from '../lib/api';

export default function PinterestBoardSelect({
    value, onChange, isManaged, uploadPostKey, uploadUserId, disabled,
}) {
    const [response, setResult] = useState(null);
    const [revision, setRevision] = useState(0);
    const canLoad = isManaged || Boolean(uploadPostKey && uploadUserId);
    const query = isManaged ? '' : `?${new URLSearchParams({ user: uploadUserId })}`;
    const requestKey = JSON.stringify([isManaged, query, uploadPostKey, revision]);
    const result = response?.requestKey === requestKey ? response : null;

    useEffect(() => {
        if (!canLoad) return undefined;
        const controller = new AbortController();
        const headers = isManaged ? {} : { 'X-Upload-Post-Key': uploadPostKey };
        apiJson(`/api/social/pinterest-boards${query}`, { headers, signal: controller.signal }).then((data) => {
            if (!controller.signal.aborted) setResult({ requestKey, boards: data.boards || [] });
        }).catch((error) => {
            if (!controller.signal.aborted) setResult({ requestKey, error: error.detail || 'Could not load Pinterest boards.' });
        });
        return () => controller.abort();
    }, [canLoad, isManaged, query, requestKey, uploadPostKey]);

    const boards = result?.boards || [];
    const selectedUnavailable = Boolean(value && result?.boards && !boards.some((board) => board.id === value));

    return (
        <div className="space-y-2">
            <label className="block text-xs text-ink2">
                Pinterest board
                <select className="input-field mt-1" value={value || ''} disabled={disabled || !canLoad || !result?.boards}
                    onChange={(event) => onChange(event.target.value)}>
                    <option value="">{canLoad && !result ? 'Loading boards…' : 'Choose a board'}</option>
                    {selectedUnavailable && <option value={value} disabled>Unavailable board</option>}
                    {boards.map((board) => <option key={board.id} value={board.id}>{board.name}</option>)}
                </select>
            </label>
            {canLoad && result?.boards?.length === 0 && (
                <p className="text-xs text-muted">No publishing boards found for this Pinterest account.</p>
            )}
            {result?.error && (
                <div className="flex flex-wrap items-center gap-2 text-xs">
                    <p className="text-danger" role="alert">{result.error}</p>
                    <button type="button" className="btn-quiet py-1" disabled={disabled}
                        onClick={() => { setResult(null); setRevision((current) => current + 1); }}>
                        Retry
                    </button>
                </div>
            )}
        </div>
    );
}
