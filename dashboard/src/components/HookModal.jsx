import React, { useState } from 'react';
import { Loader2 } from 'lucide-react';
import RemotionPreview from './RemotionPreview';
import Modal from './ui/Modal';
import SegmentedControl from './ui/SegmentedControl';

const ENTRANCE_OPTIONS = [
    { value: 'spring', label: 'Bounce' },
    { value: 'fade', label: 'Fade' },
    { value: 'slide-up', label: 'Slide Up' },
    { value: 'none', label: 'None' },
];

const POSITION_OPTIONS = [
    { value: 'top', label: 'top' },
    { value: 'center', label: 'center' },
    { value: 'bottom', label: 'bottom' },
];

const SIZE_OPTIONS = [
    { value: 'S', label: 'Small' },
    { value: 'M', label: 'Medium' },
    { value: 'L', label: 'Large' },
];

export default function HookModal({ isOpen, onClose, onGenerate, isProcessing, videoUrl, initialText, durationInSeconds, existingSubtitles }) {
    const [text, setText] = useState(initialText || 'POV: You are using the viral hook feature');
    const [position, setPosition] = useState('top');
    const [size, setSize] = useState('M');
    const [entranceAnimation, setEntranceAnimation] = useState('spring');
    const [displayDuration, setDisplayDuration] = useState(5);

    if (!isOpen) return null;

    // Build hook config for Remotion preview
    const hookConfig = {
        text: text || 'Enter your text...',
        position,
        size,
        entranceAnimation,
        displayDurationSec: displayDuration,
    };

    const useRemotionPreview = !!videoUrl;

    // Fallback preview logic (same as original)
    const getPositionClass = () => {
        switch (position) {
            case 'center': return 'items-center justify-center';
            case 'bottom': return 'items-center justify-end pb-[20%]';
            case 'top': default: return 'items-center justify-start pt-[20%]';
        }
    };

    const getSizeStyle = () => {
        switch (size) {
            case 'S': return { fontSize: '14px', maxWidth: '80%' };
            case 'L': return { fontSize: '24px', maxWidth: '95%' };
            case 'M': default: return { fontSize: '18px', maxWidth: '90%' };
        }
    };

    return (
        <Modal isOpen={isOpen} onClose={onClose} size="lg" eyebrow="EDITOR · HOOK" title="viral hook">
            <div className="flex flex-col md:flex-row gap-6">
                {/* Left: Preview */}
                <div className="flex-1 flex flex-col items-center justify-center bg-black rounded-card border border-rule overflow-hidden relative aspect-[9/16] max-h-[600px]">
                    {useRemotionPreview ? (
                        <RemotionPreview
                            videoUrl={videoUrl}
                            durationInSeconds={durationInSeconds || 30}
                            hook={hookConfig}
                            subtitles={existingSubtitles || null}
                        />
                    ) : (
                        <>
                            <video src={videoUrl} className="w-full h-full object-contain opacity-50" muted playsInline />
                            <div className={`absolute w-full px-8 text-center transition-all duration-300 pointer-events-none flex flex-col h-full ${getPositionClass()}`}>
                                <div
                                    className="text-black font-bold px-3 py-2 rounded-xl shadow-2xl text-center whitespace-pre-wrap transition-all duration-200"
                                    style={{
                                        ...getSizeStyle(),
                                        backgroundColor: 'rgba(255, 255, 255, 0.82)',
                                        fontFamily: 'Noto Serif, serif',
                                        boxShadow: '0 4px 15px rgba(0,0,0,0.5)',
                                        paddingTop: '10px',
                                        paddingBottom: '10px',
                                        paddingLeft: '12px',
                                        paddingRight: '12px'
                                    }}
                                >
                                    {text || "Enter your text..."}
                                </div>
                            </div>
                        </>
                    )}
                </div>

                {/* Right: Controls */}
                <div className="w-full md:w-80 flex flex-col">
                    <div className="space-y-5 flex-1 overflow-y-auto custom-scrollbar pr-1">
                        {/* Text Input */}
                        <div>
                            <p className="eyebrow mb-2">Text</p>
                            <textarea
                                value={text}
                                onChange={(e) => setText(e.target.value)}
                                rows={4}
                                className="input-field resize-none font-serif"
                                style={{ fontFamily: 'Noto Serif, serif' }}
                                placeholder="Enter text that will stop the scroll..."
                            />
                        </div>

                        {/* Position Control */}
                        <div>
                            <p className="eyebrow mb-2">Position</p>
                            <SegmentedControl
                                options={POSITION_OPTIONS}
                                value={position}
                                onChange={setPosition}
                                size="sm"
                            />
                        </div>

                        {/* Size Control */}
                        <div>
                            <p className="eyebrow mb-2">Size</p>
                            <SegmentedControl
                                options={SIZE_OPTIONS}
                                value={size}
                                onChange={setSize}
                                size="sm"
                            />
                        </div>

                        {/* Entrance Animation (new) */}
                        <div>
                            <p className="eyebrow mb-2">Entrance</p>
                            <SegmentedControl
                                options={ENTRANCE_OPTIONS}
                                value={entranceAnimation}
                                onChange={setEntranceAnimation}
                                columns={2}
                                size="sm"
                            />
                        </div>

                        {/* Display Duration (new) */}
                        <div>
                            <div className="flex items-center justify-between mb-2">
                                <p className="eyebrow">Duration</p>
                                <span className="readout">{displayDuration}S</span>
                            </div>
                            <input
                                type="range"
                                min="2"
                                max="15"
                                value={displayDuration}
                                onChange={(e) => setDisplayDuration(parseInt(e.target.value))}
                                className="w-full accent-[var(--color-accent)]"
                            />
                            <div className="flex justify-between">
                                <span className="readout">2S</span>
                                <span className="readout">15S</span>
                            </div>
                        </div>

                        <div className="p-3 border border-rule rounded-input text-xs text-muted">
                            Tip: keep it short and punchy. Using "POV:" or specific questions works best for retention.
                        </div>
                    </div>

                    <div className="flex gap-2 mt-5 shrink-0">
                        <button onClick={onClose} className="btn-ghost">
                            cancel
                        </button>
                        <button
                            onClick={() => onGenerate({
                                text, position, size,
                                // Remotion data
                                remotion: hookConfig,
                            })}
                            disabled={isProcessing || !text.trim()}
                            className="btn-primary flex-1"
                        >
                            {isProcessing && <Loader2 size={16} className="animate-spin text-brassink" />}
                            {isProcessing ? 'generating...' : 'add hook'}
                        </button>
                    </div>
                </div>
            </div>
        </Modal>
    );
}
