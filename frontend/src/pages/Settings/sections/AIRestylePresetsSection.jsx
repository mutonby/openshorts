// AI Restyle preset CRUD. Two dimensions: backgrounds + lightings.
// Star toggles the default; the default cannot be deleted.

import { useState } from 'react';
import { Pencil, Plus, Star, Trash2 } from 'lucide-react';
import SectionHeader from './SectionHeader.jsx';
import {
  deletePreset,
  setDefault,
  upsertPreset,
  useAIRestylePresets,
} from '../../../state/aiRestylePresets.js';

export default function AIRestylePresetsSection() {
  const presets = useAIRestylePresets();
  const [editing, setEditing] = useState(null); // { dimension, preset } | null

  return (
    <section>
      <SectionHeader
        title="AI Restyle presets"
        description="Edit the prompts used to relight the first frame. Star marks the default; the default can't be deleted."
      />

      <div className="space-y-6">
        <Dimension
          title="Backgrounds"
          dimension="background"
          items={presets.backgrounds}
          defaultId={presets.defaultBackgroundId}
          onEdit={(p) => setEditing({ dimension: 'background', preset: p })}
          onAdd={() => setEditing({ dimension: 'background', preset: { id: '', label: '', prompt: '' } })}
        />

        <Dimension
          title="Lightings"
          dimension="lighting"
          items={presets.lightings}
          defaultId={presets.defaultLightingId}
          onEdit={(p) => setEditing({ dimension: 'lighting', preset: p })}
          onAdd={() => setEditing({ dimension: 'lighting', preset: { id: '', label: '', prompt: '' } })}
        />
      </div>

      {editing && (
        <EditModal
          dimension={editing.dimension}
          preset={editing.preset}
          onClose={() => setEditing(null)}
          onSave={(p) => { upsertPreset(editing.dimension, p); setEditing(null); }}
        />
      )}
    </section>
  );
}

function Dimension({ title, dimension, items, defaultId, onEdit, onAdd }) {
  return (
    <div>
      <h3 className="text-[13px] font-medium text-white mb-2">{title}</h3>
      <div className="rounded-lg border border-border overflow-hidden">
        {items.map((p) => {
          const isDefault = p.id === defaultId;
          return (
            <div key={p.id} className="p-3 hover:bg-white/[0.03] flex items-start gap-3 border-b border-border last:border-b-0">
              <button
                onClick={() => setDefault(dimension, p.id)}
                title={isDefault ? 'Default' : 'Set as default'}
                className={`mt-0.5 transition-colors ${isDefault ? 'text-yellow-400' : 'text-zinc-600 hover:text-zinc-300'}`}
              >
                <Star size={14} fill={isDefault ? 'currentColor' : 'none'} />
              </button>
              <div className="flex-1 min-w-0">
                <div className="text-[13px] text-white font-medium">{p.label}</div>
                <div className="text-[11px] text-zinc-500 mt-0.5 leading-snug">{p.prompt}</div>
              </div>
              <button
                onClick={() => onEdit(p)}
                className="p-1.5 text-zinc-500 hover:text-white transition-colors"
                title="Edit"
              >
                <Pencil size={12} />
              </button>
              <button
                onClick={() => deletePreset(dimension, p.id)}
                disabled={isDefault}
                className="p-1.5 text-zinc-500 hover:text-red-400 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                title={isDefault ? 'Cannot delete the default preset' : 'Delete'}
              >
                <Trash2 size={12} />
              </button>
            </div>
          );
        })}
        <button
          onClick={onAdd}
          className="w-full p-3 text-[12px] text-zinc-400 hover:text-white hover:bg-white/[0.03] border-t border-border flex items-center justify-center gap-2 transition-colors"
        >
          <Plus size={12} /> Add {dimension} preset
        </button>
      </div>
    </div>
  );
}

function EditModal({ dimension, preset, onClose, onSave }) {
  const [label, setLabel] = useState(preset.label || '');
  const [prompt, setPrompt] = useState(preset.prompt || '');

  const isNew = !preset.id;
  const canSave = label.trim().length > 0 && prompt.trim().length > 0;

  function save() {
    if (!canSave) return;
    const id = preset.id || label.trim().toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 40) || `preset-${Date.now()}`;
    onSave({ id, label: label.trim().slice(0, 40), prompt: prompt.trim().slice(0, 500) });
  }

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50" onClick={onClose}>
      <div className="bg-surface border border-border rounded-lg p-5 w-full max-w-md" onClick={(e) => e.stopPropagation()}>
        <h3 className="text-[14px] font-medium text-white mb-3">
          {isNew ? 'Add' : 'Edit'} {dimension} preset
        </h3>
        <label className="block text-[11px] text-zinc-500 uppercase tracking-wider mb-1">Name</label>
        <input
          value={label}
          onChange={(e) => setLabel(e.target.value)}
          maxLength={40}
          className="w-full bg-background border border-border rounded px-3 py-1.5 text-[13px] text-white mb-3 focus:outline-none focus:border-primary"
        />
        <label className="block text-[11px] text-zinc-500 uppercase tracking-wider mb-1">Prompt</label>
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          maxLength={500}
          rows={4}
          className="w-full bg-background border border-border rounded px-3 py-2 text-[12px] text-zinc-200 font-mono focus:outline-none focus:border-primary"
        />
        <div className="text-[10px] text-zinc-500 text-right mt-1">{prompt.length}/500</div>
        <div className="mt-4 flex justify-end gap-2">
          <button onClick={onClose} className="px-3 py-1.5 text-[12px] text-zinc-400 hover:text-white transition-colors">
            Cancel
          </button>
          <button
            onClick={save}
            disabled={!canSave}
            className="px-3 py-1.5 text-[12px] bg-primary text-white rounded hover:bg-primary/90 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            Save
          </button>
        </div>
      </div>
    </div>
  );
}
