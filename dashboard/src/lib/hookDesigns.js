export const DEFAULT_HOOK_DESIGN = {
    shape: 'pill',
    text_color: '#000000',
    background_color: '#FFFFFF',
    outline_color: '#000000',
    outline_width: 0,
    corner_radius: 24,
    text_case: 'as_written',
};

const storageKey = (userId) => `os_hook_designs:${userId || 'local'}`;

export function readHookDesigns(userId) {
    try {
        const items = JSON.parse(localStorage.getItem(storageKey(userId)) || '[]');
        return Array.isArray(items)
            ? items.filter((item) => typeof item?.name === 'string' && item.design?.shape)
            : [];
    } catch { return []; }
}

export function writeHookDesigns(userId, items) {
    try {
        localStorage.setItem(storageKey(userId), JSON.stringify(items));
        return true;
    } catch { return false; }
}
