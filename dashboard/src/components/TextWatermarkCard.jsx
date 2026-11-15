import React, { useState, useEffect } from 'react';
import { Type, Loader2, Check } from 'lucide-react';
import { apiJson } from '../lib/api';
import WatermarkControls from './WatermarkControls';
import WatermarkOverlay from './WatermarkOverlay';

const PREVIEW_HEIGHT = 288;
const PREVIEW_WIDTH = Math.round(PREVIEW_HEIGHT * 9 / 16);

export default function TextWatermarkCard() {
  const [settings, setSettings] = useState(null);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    apiJson('/api/watermark')
      .then(setSettings)
      .catch(() => setError('Could not load the watermark settings.'));
  }, []);

  if (!settings) {
    return (
      <div className="card p-6">
        {error ? <p className="text-danger text-sm">{error}</p> : <Loader2 size={16} className="animate-spin text-muted" />}
      </div>
    );
  }

  const save = async () => {
    setSaving(true);
    setError('');
    try {
      const data = await apiJson('/api/watermark', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...settings, enabled: settings.enabled && !!settings.text.trim() }),
      });
      setSettings(data);
      setSaved(true);
    } catch (e) {
      setError(e?.detail || 'Could not save.');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="card p-6">
      <div className="flex items-center gap-3 mb-3">
        <div className="w-9 h-9 rounded-input bg-paper3 flex items-center justify-center shrink-0">
          <Type size={16} className="text-brass" />
        </div>
        <h2 className="text-base font-medium text-ink lowercase">Text watermark</h2>
      </div>
      <p className="text-xs text-muted mb-5 leading-relaxed">
        Burned into every new clip, on top of the captions. Also editable from a clip&apos;s subtitle editor.
      </p>

      <div className="flex flex-col sm:flex-row gap-6">
        <div className="flex-1 min-w-0">
          <WatermarkControls value={settings} onChange={(v) => { setSettings(v); setSaved(false); }} />
        </div>
        <div
          className="relative shrink-0 rounded-input overflow-hidden border border-rule"
          style={{
            width: PREVIEW_WIDTH,
            height: PREVIEW_HEIGHT,
            background: 'linear-gradient(160deg, #3a4a5c 0%, #1d2530 55%, #4b3b2a 100%)',
          }}
        >
          <WatermarkOverlay
            value={{ ...settings, enabled: true, text: settings.text || 'your text' }}
            height={PREVIEW_HEIGHT}
            width={PREVIEW_WIDTH}
          />
        </div>
      </div>

      {error && <p className="text-danger text-sm mt-4">{error}</p>}
      <div className="flex items-center gap-3 mt-5">
        <button onClick={save} disabled={saving} className="btn-primary py-2 px-4 text-sm">
          {saving ? <Loader2 size={16} className="animate-spin" /> : <Check size={16} />} save watermark
        </button>
        {saved && <span className="text-ok text-xs">saved: applies to new clips</span>}
      </div>
    </div>
  );
}
