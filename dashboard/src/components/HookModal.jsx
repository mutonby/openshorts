import React, { useState } from 'react';
import { X, Sparkles, Loader2, Maximize, MoveVertical } from 'lucide-react';

export default function HookModal({ isOpen, onClose, onGenerate, isProcessing, videoUrl, initialText }) {
    const [text, setText] = useState(initialText || 'POV: You are using the viral hook feature');
    const [position, setPosition] = useState('top'); // top, center, bottom
    const [size, setSize] = useState('M'); // S, M, L

    if (!isOpen) return null;

    // Preview Logic
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
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-[fadeIn_0.2s_ease-out]">
            <div className="bg-[#121214] border border-white/10 p-6 rounded-2xl w-full max-w-4xl shadow-2xl relative flex flex-col md:flex-row gap-6 max-h-[90vh]">
                <button
                    onClick={onClose}
                    className="absolute top-4 right-4 text-zinc-500 hover:text-white z-10"
                >
                    <X size={20} />
                </button>

                {/* Left: Preview */}
                <div className="flex-1 flex flex-col items-center justify-center bg-black rounded-lg border border-white/5 overflow-hidden relative aspect-[9/16] max-h-[600px]">
                    <video src={videoUrl} className="w-full h-full object-contain opacity-50" muted playsInline />

                    {/* Hook Overlay Preview */}
                    <div className={`absolute w-full px-8 text-center transition-all duration-300 pointer-events-none flex flex-col h-full ${getPositionClass()}`}>
                        <div
                            className="text-black font-bold px-3 py-2 rounded-xl shadow-2xl text-center whitespace-pre-wrap transition-all duration-200"
                            style={{
                                ...getSizeStyle(),
                                backgroundColor: 'rgba(255, 255, 255, 0.82)', // Match backend 210/255 alpha
                                fontFamily: 'Noto Serif, serif',
                                boxShadow: '0 4px 15px rgba(0,0,0,0.5)',
                                paddingTop: '10px', // Reduced to scale (backend is 20px on 1080p ~ 2%)
                                paddingBottom: '10px',
                                paddingLeft: '12px',
                                paddingRight: '12px'
                            }}
                        >
                            {text || "Enter your text..."}
                        </div>
                    </div>
                </div>

                {/* Right: Controls */}
                <div className="w-full md:w-80 flex flex-col">
                    <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
                        <Sparkles className="text-yellow-400" /> Viral Hook
                    </h3>

                    <div className="space-y-6 flex-1 overflow-y-auto custom-scrollbar pr-2">
                        {/* Text Input */}
                        <div>
                            <label className="text-xs font-bold text-zinc-400 uppercase tracking-wider mb-3 block">Text</label>
                            <textarea
                                value={text}
                                onChange={(e) => setText(e.target.value)}
                                rows={4}
                                className="w-full bg-black/40 border border-white/10 rounded-xl p-3 text-white placeholder-zinc-600 focus:outline-none focus:border-yellow-500/50 resize-none font-serif"
                                placeholder="Enter text that will stop the scroll..."
                            />
                        </div>

                        {/* Position Control */}
                        <div>
                            <label className="text-xs font-bold text-zinc-400 uppercase tracking-wider mb-3 flex items-center gap-2">
                                <MoveVertical size={12} /> Position
                            </label>
                            <div className="grid grid-cols-3 gap-2">
                                {['top', 'center', 'bottom'].map((pos) => (
                                    <button
                                        key={pos}
                                        onClick={() => setPosition(pos)}
                                        className={`py-2 px-1 rounded-lg text-xs font-bold capitalize transition-all border ${position === pos
                                            ? 'bg-white text-black border-white'
                                            : 'bg-white/5 text-zinc-400 border-white/5 hover:bg-white/10'
                                            }`}
                                    >
                                        {pos}
                                    </button>
                                ))}
                            </div>
                        </div>

                        {/* Size Control */}
                        <div>
                            <label className="text-xs font-bold text-zinc-400 uppercase tracking-wider mb-3 flex items-center gap-2">
                                <Maximize size={12} /> Size
                            </label>
                            <div className="grid grid-cols-3 gap-2">
                                {['S', 'M', 'L'].map((sz) => (
                                    <button
                                        key={sz}
                                        onClick={() => setSize(sz)}
                                        className={`py-2 px-1 rounded-lg text-xs font-bold transition-all border ${size === sz
                                            ? 'bg-white text-black border-white'
                                            : 'bg-white/5 text-zinc-400 border-white/5 hover:bg-white/10'
                                            }`}
                                    >
                                        {sz === 'S' ? 'Small' : sz === 'M' ? 'Medium' : 'Large'}
                                    </button>
                                ))}
                            </div>
                        </div>

                        <div className="p-3 bg-white/5 rounded-lg border border-white/5 text-[11px] text-zinc-400">
                            <strong>Tip:</strong> Keep it short and punchy. Using "POV:" or specific questions works best for retention.
                        </div>
                    </div>

                    <button
                        onClick={() => onGenerate({ text, position, size })}
                        disabled={isProcessing || !text.trim()}
                        className="w-full py-4 mt-4 bg-gradient-to-r from-yellow-500 to-amber-600 hover:from-yellow-400 hover:to-amber-500 text-black font-bold rounded-xl shadow-lg shadow-amber-500/20 transition-all active:scale-[0.98] flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed shrink-0"
                    >
                        {isProcessing ? <Loader2 size={20} className="animate-spin" /> : <Sparkles size={20} />}
                        {isProcessing ? 'Generating...' : 'Add Hook'}
                    </button>
                </div>
            </div>
        </div>
    );
}
