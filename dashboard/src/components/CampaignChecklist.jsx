import { useState } from 'react';

const VISUAL_CHECKS = [
    'The footage is from the campaign’s official source.',
    'The official logo is visible throughout this export.',
    'The edit is original, and the title and caption match the clip.',
];

export default function CampaignChecklist({ description, media, onDownload, downloading }) {
    const [caption, setCaption] = useState(description || '');
    const [requiredText, setRequiredText] = useState('');
    const [hashtags, setHashtags] = useState('');
    const [minimum, setMinimum] = useState('');
    const [maximum, setMaximum] = useState('');
    const [verified, setVerified] = useState([]);
    const presentTags = new Set((caption.match(/#[\p{L}\p{N}_]+/gu) || []).map(tag => tag.toLowerCase()));
    const missingTags = (hashtags.match(/#?[\p{L}\p{N}_]+/gu) || [])
        .map(tag => `#${tag.replace(/^#/, '').toLowerCase()}`)
        .filter(tag => !presentTags.has(tag));
    const exactText = !requiredText.trim() || caption.includes(requiredText.trim());
    const enough = !minimum || media?.duration >= Number(minimum);
    const shortEnough = !maximum || media?.duration <= Number(maximum);
    const vertical = media && Math.abs(media.width / media.height - 9 / 16) <= 0.01;
    const ready = vertical && Number.isFinite(media.duration) && enough && shortEnough
        && exactText && !missingTags.length && verified.length === VISUAL_CHECKS.length;

    return (
        <details className="mb-3 rounded-input border border-rule px-3 py-2 text-xs">
            <summary className="cursor-pointer text-ink2">Campaign export checklist</summary>
            <div className="space-y-3 mt-3">
                <label className="block">
                    <span className="block mb-1 text-muted">Caption you plan to publish</span>
                    <textarea className="input-field" rows={2} value={caption} onChange={e => setCaption(e.target.value)} />
                </label>
                <label className="block">
                    <span className="block mb-1 text-muted">Required wording</span>
                    <input className="input-field" value={requiredText} onChange={e => setRequiredText(e.target.value)} />
                </label>
                <label className="block">
                    <span className="block mb-1 text-muted">Required hashtags</span>
                    <input className="input-field" value={hashtags} onChange={e => setHashtags(e.target.value)} />
                </label>
                <div className="grid grid-cols-2 gap-2">
                    <label>
                        <span className="block mb-1 text-muted">Min length (s)</span>
                        <input className="input-field" type="number" min="0" value={minimum} onChange={e => setMinimum(e.target.value)} />
                    </label>
                    <label>
                        <span className="block mb-1 text-muted">Max length (s)</span>
                        <input className="input-field" type="number" min="0" value={maximum} onChange={e => setMaximum(e.target.value)} />
                    </label>
                </div>
                <p className={vertical && enough && shortEnough ? 'text-ink2' : 'text-warn'}>
                    {media ? `${media.width}×${media.height} · ${media.duration.toFixed(2)}s` : 'Load the final video to check its format.'}
                    {media && (!vertical || !enough || !shortEnough) && ' · Outside the 9:16 or length requirements'}
                </p>
                {!exactText && <p className="text-warn">The caption is missing the required wording.</p>}
                {missingTags.length > 0 && <p className="text-warn">Missing {missingTags.join(', ')}</p>}
                {VISUAL_CHECKS.map((check, i) => (
                    <label key={check} className="flex items-start gap-2 text-ink2">
                        <input type="checkbox" className="accent-brass mt-0.5" checked={verified.includes(i)}
                            onChange={(e) => {
                                const checked = e.target.checked;
                                setVerified((previous) => checked ? [...previous, i] : previous.filter(index => index !== i));
                            }} />
                        {check}
                    </label>
                ))}
                <button type="button" className="btn-ghost px-3 py-2" disabled={!ready || downloading} onClick={onDownload}>
                    Download checked clip
                </button>
            </div>
        </details>
    );
}
