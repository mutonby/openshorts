// Notification feed for the header bell. Frontend-only — backend has no
// push channel yet, so publish/render events are pushed here at the
// moment of each call. Persists to localStorage so the bell survives
// reloads. See plan TODO #9 for the missing backend status endpoint.

import { useEffect, useState } from 'react';

const STORAGE_KEY = 'openshorts.notifications.v1';
const MAX_ITEMS = 50;
const EVENT = 'openshorts:notifications-changed';

function load() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

let _items = load();

function persist() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(_items));
  } catch {
    // localStorage full — drop oldest and retry once
    _items = _items.slice(0, Math.max(0, _items.length - 5));
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(_items)); } catch {}
  }
}

function emit() {
  persist();
  window.dispatchEvent(new CustomEvent(EVENT, { detail: _items }));
}

export function listNotifications() {
  return _items;
}

export function unreadCount() {
  return _items.filter((n) => !n.read).length;
}

export function pushNotification(input) {
  const item = {
    id: input.id || `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    type: input.type || 'event',          // publish | render | job | event
    platform: input.platform || null,     // youtube | tiktok | instagram | snapchat | facebook
    status: input.status || 'submitted',  // submitted | scheduled | published | failed
    jobId: input.jobId || null,
    publishId: input.publishId || null,
    ts: input.ts || Date.now(),
    message: input.message || '',
    read: false,
  };
  _items = [item, ..._items].slice(0, MAX_ITEMS);
  emit();
  return item;
}

export function markRead(id) {
  let changed = false;
  _items = _items.map((n) => {
    if (n.id === id && !n.read) { changed = true; return { ...n, read: true }; }
    return n;
  });
  if (changed) emit();
}

export function markAllRead() {
  if (!_items.some((n) => !n.read)) return;
  _items = _items.map((n) => ({ ...n, read: true }));
  emit();
}

export function clearNotifications() {
  _items = [];
  emit();
}

export function useNotifications() {
  const [items, setItems] = useState(_items);
  useEffect(() => {
    const onChange = (e) => setItems(e.detail || listNotifications());
    const onStorage = (e) => {
      if (e.key === STORAGE_KEY) {
        _items = load();
        setItems(_items);
      }
    };
    window.addEventListener(EVENT, onChange);
    window.addEventListener('storage', onStorage);
    return () => {
      window.removeEventListener(EVENT, onChange);
      window.removeEventListener('storage', onStorage);
    };
  }, []);
  return {
    items,
    unread: items.filter((n) => !n.read).length,
    markRead,
    markAllRead,
    clearNotifications,
  };
}
