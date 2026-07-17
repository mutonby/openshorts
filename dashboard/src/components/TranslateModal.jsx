import React, { useState } from 'react';
import { Loader2, Languages, AlertCircle } from 'lucide-react';
import Modal from './ui/Modal';

const LANGUAGES = {
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "pl": "Polish",
    "hi": "Hindi",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "ar": "Arabic",
    "ru": "Russian",
    "tr": "Turkish",
    "nl": "Dutch",
    "sv": "Swedish",
    "id": "Indonesian",
    "fil": "Filipino",
    "ms": "Malay",
    "vi": "Vietnamese",
    "th": "Thai",
    "uk": "Ukrainian",
    "el": "Greek",
    "cs": "Czech",
    "fi": "Finnish",
    "ro": "Romanian",
    "da": "Danish",
    "bg": "Bulgarian",
    "hr": "Croatian",
    "sk": "Slovak",
    "ta": "Tamil",
    "en": "English",
};

export default function TranslateModal({ isOpen, onClose, onTranslate, isProcessing, videoUrl, hasApiKey }) {
    const [targetLanguage, setTargetLanguage] = useState('es');

    if (!isOpen) return null;

    const handleSubmit = () => {
        console.log('[TranslateModal] handleSubmit called, targetLanguage:', targetLanguage);
        onTranslate({ targetLanguage });
    };

    return (
        <Modal
            isOpen={isOpen}
            onClose={isProcessing ? undefined : onClose}
            eyebrow="DUB"
            title="dub voice"
            size="md"
            footer={
                <div className="flex gap-3">
                    <button
                        onClick={onClose}
                        disabled={isProcessing}
                        className="btn-ghost flex-1"
                    >
                        Cancel
                    </button>
                    <button
                        onClick={handleSubmit}
                        disabled={isProcessing || !hasApiKey}
                        className="btn-primary flex-1"
                    >
                        {isProcessing ? (
                            <>
                                <Loader2 size={16} className="animate-spin" />
                                Dubbing...
                            </>
                        ) : (
                            <>
                                <Languages size={16} />
                                Dub Voice
                            </>
                        )}
                    </button>
                </div>
            }
        >
            <div className="flex items-center gap-3 mb-5">
                <div className="w-10 h-10 rounded-input bg-paper3 flex items-center justify-center shrink-0">
                    <Languages size={18} className="text-brass" />
                </div>
                <p className="text-xs text-muted">AI voice translation by ElevenLabs</p>
            </div>

            {!hasApiKey && (
                <div className="mb-4 flex items-start gap-2">
                    <span className="badge-warn shrink-0"><AlertCircle size={12} /> key missing</span>
                    <p className="text-sm text-muted">Configure ElevenLabs API Key in Settings first.</p>
                </div>
            )}

            {/* Preview */}
            <div className="mb-5 rounded-card overflow-hidden bg-black aspect-video">
                <video
                    src={videoUrl}
                    className="w-full h-full object-contain"
                    muted
                    playsInline
                />
            </div>

            {/* Language Selection */}
            <div className="mb-5">
                <label className="eyebrow block mb-2">
                    Target Language
                </label>
                <select
                    value={targetLanguage}
                    onChange={(e) => setTargetLanguage(e.target.value)}
                    className="input-field appearance-none cursor-pointer"
                    disabled={isProcessing}
                >
                    {Object.entries(LANGUAGES).sort((a, b) => a[1].localeCompare(b[1])).map(([code, name]) => (
                        <option key={code} value={code}>
                            {name}
                        </option>
                    ))}
                </select>
            </div>

            {/* Info */}
            <p className="text-xs text-muted leading-relaxed mb-2">
                The audio will be dubbed with AI-generated voice in the selected language, matching the original speaker's characteristics.
            </p>

            {/* Processing State */}
            {isProcessing && (
                <div className="mt-4 p-3 bg-paper3 rounded-input">
                    <div className="flex items-center gap-3">
                        <Loader2 size={18} className="text-brass animate-spin" />
                        <div>
                            <p className="text-sm text-ink font-medium lowercase">Dubbing audio...</p>
                            <p className="text-xs text-muted lowercase">This may take a few minutes</p>
                        </div>
                    </div>
                </div>
            )}
        </Modal>
    );
}
