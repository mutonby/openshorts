import React, { useState } from 'react';
import { Youtube, Upload, FileVideo, X, Sparkles, Smartphone, Monitor, Square } from 'lucide-react';

const FORMAT_OPTIONS = [
    { id: 'auto', label: 'Auto', hint: 'Smart detect', Icon: Sparkles },
    { id: 'vertical', label: '9:16', hint: 'Shorts / Reels', Icon: Smartphone },
    { id: 'horizontal', label: '16:9', hint: 'Original wide', Icon: Monitor },
    { id: 'square', label: '1:1', hint: 'Square feed', Icon: Square },
];

export default function MediaInput({ onProcess, isProcessing }) {
    const [mode, setMode] = useState('url'); // 'url' | 'file'
    const [url, setUrl] = useState('');
    const [file, setFile] = useState(null);
    const [outputFormat, setOutputFormat] = useState('auto');

    const handleSubmit = (e) => {
        e.preventDefault();
        if (mode === 'url' && url) {
            onProcess({ type: 'url', payload: url, outputFormat });
        } else if (mode === 'file' && file) {
            onProcess({ type: 'file', payload: file, outputFormat });
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            setFile(e.dataTransfer.files[0]);
            setMode('file');
        }
    };

    return (
        <div className="bg-surface border border-white/5 rounded-2xl p-6 animate-[fadeIn_0.6s_ease-out]">
            <div className="flex gap-4 mb-6 border-b border-white/5 pb-4">
                <button
                    onClick={() => setMode('url')}
                    className={`flex items-center gap-2 pb-2 px-2 transition-all ${mode === 'url'
                        ? 'text-primary border-b-2 border-primary -mb-[17px]'
                        : 'text-zinc-400 hover:text-white'
                        }`}
                >
                    <Youtube size={18} />
                    YouTube URL
                </button>
                <button
                    onClick={() => setMode('file')}
                    className={`flex items-center gap-2 pb-2 px-2 transition-all ${mode === 'file'
                        ? 'text-primary border-b-2 border-primary -mb-[17px]'
                        : 'text-zinc-400 hover:text-white'
                        }`}
                >
                    <Upload size={18} />
                    Upload File
                </button>
            </div>

            <form onSubmit={handleSubmit}>
                {mode === 'url' ? (
                    <div className="space-y-4">
                        <input
                            type="url"
                            value={url}
                            onChange={(e) => setUrl(e.target.value)}
                            placeholder="https://www.youtube.com/watch?v=..."
                            className="input-field"
                            required
                        />
                    </div>
                ) : (
                    <div
                        className={`border-2 border-dashed rounded-xl p-8 text-center transition-all ${file ? 'border-primary/50 bg-primary/5' : 'border-zinc-700 hover:border-zinc-500 bg-white/5'
                            }`}
                        onDragOver={(e) => e.preventDefault()}
                        onDrop={handleDrop}
                    >
                        {file ? (
                            <div className="flex items-center justify-center gap-3 text-white">
                                <FileVideo className="text-primary" />
                                <span className="font-medium">{file.name}</span>
                                <button
                                    type="button"
                                    onClick={() => setFile(null)}
                                    className="p-1 hover:bg-white/10 rounded-full"
                                >
                                    <X size={16} />
                                </button>
                            </div>
                        ) : (
                            <label className="cursor-pointer block">
                                <input
                                    type="file"
                                    accept="video/*"
                                    onChange={(e) => setFile(e.target.files?.[0] || null)}
                                    className="hidden"
                                />
                                <Upload className="mx-auto mb-3 text-zinc-500" size={24} />
                                <p className="text-zinc-400">Click to upload or drag and drop</p>
                                <p className="text-xs text-zinc-600 mt-1">MP4, MOV up to 500MB</p>
                            </label>
                        )}
                    </div>
                )}

                {/* Output format */}
                <div className="mt-5">
                    <div className="text-xs font-bold text-zinc-400 uppercase tracking-wider mb-2">Output Format</div>
                    <div className="grid grid-cols-4 gap-2">
                        {FORMAT_OPTIONS.map(({ id, label, hint, Icon }) => (
                            <button
                                key={id}
                                type="button"
                                onClick={() => setOutputFormat(id)}
                                className={`flex flex-col items-center gap-1 py-2.5 px-1 rounded-xl border text-center transition-all ${outputFormat === id
                                    ? 'border-primary/60 bg-primary/10 text-white'
                                    : 'border-white/5 bg-white/5 text-zinc-400 hover:border-white/15 hover:text-white'
                                    }`}
                            >
                                <Icon size={16} className={outputFormat === id ? 'text-primary' : ''} />
                                <span className="text-xs font-bold">{label}</span>
                                <span className="text-[9px] text-zinc-500 leading-tight">{hint}</span>
                            </button>
                        ))}
                    </div>
                </div>

                <button
                    type="submit"
                    disabled={isProcessing || (mode === 'url' && !url) || (mode === 'file' && !file)}
                    className="w-full btn-primary mt-6 flex items-center justify-center gap-2"
                >
                    {isProcessing ? (
                        <>
                            <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                            Processing Video...
                        </>
                    ) : (
                        <>
                            Generate Clips
                        </>
                    )}
                </button>
            </form>
        </div>
    );
}
