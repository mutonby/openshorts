const FIELDS = {
    style: 'style', position: 'position', fontSize: 'font_size', fontName: 'font_name',
    fontColor: 'font_color', borderColor: 'border_color', borderWidth: 'border_width',
    bgColor: 'bg_color', bgOpacity: 'bg_opacity', highlightColor: 'highlight_color',
    effect: 'effect', baseOpacity: 'base_opacity', uppercase: 'uppercase', reveal: 'reveal',
    shadow: 'shadow', maxChars: 'max_chars', maxDuration: 'max_duration', maxWords: 'max_words',
};

export const captionStylePayload = (options) => Object.fromEntries(
    Object.entries(FIELDS).filter(([key]) => options[key] != null)
        .map(([key, field]) => [field, options[key]]),
);

export const captionStyleOptions = (payload) => Object.fromEntries(
    Object.entries(FIELDS).filter(([, field]) => payload?.[field] != null)
        .map(([key, field]) => [key, payload[field]]),
);
