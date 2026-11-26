export const MIN_BROLL_DURATION = 0.1;
export const roundBrollTime = (value) => Math.round(value * 1000) / 1000;

export function brollCaptionStart(segments, sourceTime) {
    let position = 0;
    for (const segment of segments) {
        if (sourceTime >= segment.start && sourceTime < segment.end) return roundBrollTime(position + sourceTime - segment.start);
        position += segment.end - segment.start;
    }
    return null;
}

export function brollSpans(layers, segments, total) {
    return layers.flatMap((layer, index) => {
        const start = layer.source_time == null ? layer.start : brollCaptionStart(segments, layer.source_time);
        if (start == null) return [];
        const duration = roundBrollTime(Math.min(layer.duration, total - start));
        return duration >= MIN_BROLL_DURATION ? [{ ...layer, index, start, duration }] : [];
    });
}
