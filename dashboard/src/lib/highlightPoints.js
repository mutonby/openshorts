// Shared by the highlight timeline and the candidate review cards. Kept out
// of the component files so React Fast Refresh can hot-swap those.

export function timecode(seconds) {
    const whole = Math.max(0, Math.floor(seconds));
    const time = [Math.floor(whole / 60) % 60, whole % 60]
        .map(value => String(value).padStart(2, '0')).join(':');
    return whole >= 3600 ? `${Math.floor(whole / 3600)}:${time}` : time;
}

export function describePoint(point) {
    if (point.kind === 'chat') {
        return `${point.activity} chat messages · typical ${point.baseline}${point.participants == null
            ? '' : ` · ${point.participants} participants`}`;
    }
    if (point.kind === 'audio') return `Mixed audio rose ${point.change_db} dB`;
    if (point.kind === 'voice') return `Streamer excitement ${point.excitement} (3 is extreme)`;
    if (point.kind === 'viewer') return 'Clipped by a Twitch viewer';
    return point.title || 'Candidate';
}

// Why a hype-engine candidate ranked where it did, for the review card.
export function hypeSummary(candidate) {
    const analysis = candidate?.assessment || {};
    const parts = [];
    if (Number.isFinite(analysis.virality)) parts.push(`Viral score ${analysis.virality}`);
    else if (Number.isFinite(candidate?.hype?.score)) parts.push(`Hype ${candidate.hype.score.toFixed(2)}`);
    for (const reason of analysis.reasons || []) parts.push(reason);
    return parts;
}
