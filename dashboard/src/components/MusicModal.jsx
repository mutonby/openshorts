import React, { useState } from 'react';
import { Loader2, Music, AlertCircle, Volume2, FileAudio, Waves } from 'lucide-react';
import Modal from './ui/Modal';
import SegmentedControl from './ui/SegmentedControl';

/**
 * Optional, opt-in music / sound-effects step for a finished clip.
 *
 * Provider-agnostic: the user picks a provider, and Sonilo is only one option.
 * "Local file" needs no key and no paid service — it proves the step isn't
 * locked to any provider. The bed is ducked under the narration, with a volume
 * control (see the backend /api/music endpoint).
 *
 * Wording: Sonilo music is licensed and safe for commercial use (terms apply);
 * Sonilo sound effects are royalty-free.
 */
export default function MusicModal({
    isOpen, onClose, onApply, isProcessing, videoUrl, hasSoniloKey,
}) {
    const [provider, setProvider] = useState('local');
    const [capability, setCapability] = useState('music');
    const [prompt, setPrompt] = useState('');
    const [localPath, setLocalPath] = useState('');
    const [musicVolume, setMusicVolume] = useState(0.35);
    const [duck, setDuck] = useState(true);

    if (!isOpen) return null;

    const providerOptions = [
        { value: 'local', label: 'local file', icon: <FileAudio size={16} />, hint: 'free' },
        { value: 'sonilo', label: 'sonilo', icon: <Waves size={16} />, hint: 'byok' },
    ];

    const capabilityOptions = [
        { value: 'music', label: 'music', icon: <Music size={16} /> },
        { value: 'sfx', label: 'sound fx', icon: <Waves size={16} /> },
    ];

    const soniloKeyMissing = provider === 'sonilo' && !hasSoniloKey;
    const localPathMissing = provider === 'local' && !localPath.trim();
    const disabled = isProcessing || soniloKeyMissing || localPathMissing;

    const handleSubmit = () => {
        onApply({
            provider,
            capability: provider === 'local' ? 'music' : capability,
            prompt: prompt.trim() || undefined,
            localAudioPath: provider === 'local' ? localPath.trim() : undefined,
            musicVolume,
            duck,
        });
    };

    return (
        <Modal
            isOpen={isOpen}
            onClose={isProcessing ? undefined : onClose}
            eyebrow="AUDIO"
            title="add music"
            size="md"
            footer={
                <div className="flex gap-3">
                    <button onClick={onClose} disabled={isProcessing} className="btn-ghost flex-1">
                        Cancel
                    </button>
                    <button onClick={handleSubmit} disabled={disabled} className="btn-primary flex-1">
                        {isProcessing ? (
                            <><Loader2 size={16} className="animate-spin" /> Mixing...</>
                        ) : (
                            <><Music size={16} /> Add Audio</>
                        )}
                    </button>
                </div>
            }
        >
            <p className="text-xs text-muted mb-5">
                Lay an audio bed under the narration, ducked while the voice speaks.
                Off by default — this only runs when you add it.
            </p>

            {/* Provider */}
            <div className="mb-5">
                <label className="eyebrow block mb-2">Provider</label>
                <SegmentedControl options={providerOptions} value={provider} onChange={setProvider} columns={2} />
                <p className="mt-2 text-xs text-muted leading-relaxed">
                    {provider === 'local'
                        ? 'Use an audio file you already have on this machine. No key, no service.'
                        : 'Sonilo watches the cut and returns a bed timed to the picture. Music is licensed and safe for commercial use (terms apply); sound effects are royalty-free.'}
                </p>
            </div>

            {/* Capability (Sonilo only) */}
            {provider === 'sonilo' && (
                <div className="mb-5">
                    <label className="eyebrow block mb-2">Type</label>
                    <SegmentedControl options={capabilityOptions} value={capability} onChange={setCapability} columns={2} />
                </div>
            )}

            {soniloKeyMissing && (
                <div className="mb-4 flex items-start gap-2">
                    <span className="badge-warn shrink-0"><AlertCircle size={12} /> key missing</span>
                    <p className="text-sm text-muted">Configure your Sonilo API Key in Settings first.</p>
                </div>
            )}

            {/* Local file path */}
            {provider === 'local' && (
                <div className="mb-5">
                    <label className="eyebrow block mb-2">Audio File Path</label>
                    <input
                        type="text"
                        value={localPath}
                        onChange={(e) => setLocalPath(e.target.value)}
                        className="input-field font-mono"
                        placeholder="/absolute/path/to/track.mp3"
                        disabled={isProcessing}
                    />
                    <p className="mt-2 text-xs text-muted">
                        Absolute path to a .mp3/.wav/.m4a/.aac/.ogg/.flac file on the server.
                    </p>
                </div>
            )}

            {/* Style prompt (Sonilo only) */}
            {provider === 'sonilo' && (
                <div className="mb-5">
                    <label className="eyebrow block mb-2">Style Hint (optional)</label>
                    <input
                        type="text"
                        value={prompt}
                        onChange={(e) => setPrompt(e.target.value)}
                        className="input-field"
                        placeholder={capability === 'sfx' ? 'e.g. footsteps, whoosh, ambience' : 'e.g. upbeat lo-fi, cinematic build'}
                        disabled={isProcessing}
                    />
                </div>
            )}

            {/* Preview */}
            <div className="mb-5 rounded-card overflow-hidden bg-black aspect-video">
                <video src={videoUrl} className="w-full h-full object-contain" muted playsInline />
            </div>

            {/* Volume control (condition 4) */}
            <div className="mb-5">
                <label className="eyebrow flex items-center justify-between mb-2">
                    <span className="flex items-center gap-1.5"><Volume2 size={14} /> Music Volume</span>
                    <span className="readout">{Math.round(musicVolume * 100)}%</span>
                </label>
                <input
                    type="range"
                    min="0" max="1" step="0.05"
                    value={musicVolume}
                    onChange={(e) => setMusicVolume(parseFloat(e.target.value))}
                    className="w-full accent-brass cursor-pointer"
                    disabled={isProcessing}
                />
            </div>

            {/* Duck toggle */}
            <div className="p-3 bg-paper rounded-input border border-rule">
                <label className="flex items-center justify-between cursor-pointer">
                    <span className="flex flex-col text-sm text-ink2 lowercase">
                        duck under narration
                        <span className="text-xs text-muted normal-case">Lower the bed while the voice is speaking.</span>
                    </span>
                    <input
                        type="checkbox"
                        checked={duck}
                        onChange={(e) => setDuck(e.target.checked)}
                        className="w-4 h-4 accent-brass cursor-pointer shrink-0"
                        disabled={isProcessing}
                    />
                </label>
            </div>

            {isProcessing && (
                <div className="mt-4 p-3 bg-paper3 rounded-input">
                    <div className="flex items-center gap-3">
                        <Loader2 size={18} className="text-brass animate-spin" />
                        <div>
                            <p className="text-sm text-ink font-medium lowercase">Mixing audio...</p>
                            <p className="text-xs text-muted lowercase">This may take a moment</p>
                        </div>
                    </div>
                </div>
            )}
        </Modal>
    );
}
