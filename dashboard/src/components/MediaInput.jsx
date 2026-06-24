import React, { useState, useEffect } from 'react';
import { Youtube, Upload, FileVideo, X, Film, Crop, Tag } from 'lucide-react';
import { getApiUrl } from '../config';

export default function MediaInput({ onProcess, isProcessing }) {
    const [youtubeUrlEnabled, setYoutubeUrlEnabled] = useState(true);
    const [mode, setMode] = useState('url'); // 'url' | 'file'
    const [url, setUrl] = useState('');
    const [file, setFile] = useState(null);
    const [acknowledged, setAcknowledged] = useState(false);
    const [cropStyle, setCropStyle] = useState('blur_bars'); // 'blur_bars' | 'auto'
    const [category, setCategory] = useState('general'); // 'general' | 'podcast' | ...

    useEffect(() => {
        fetch(getApiUrl('/api/config'))
            .then((r) => r.ok ? r.json() : null)
            .then((cfg) => {
                if (cfg && cfg.youtubeUrlEnabled === false) {
                    setYoutubeUrlEnabled(false);
                    setMode('file');
                }
            })
            .catch(() => {});
    }, []);

    const handleSubmit = (e) => {
        e.preventDefault();
        if (!acknowledged) return;
        if (mode === 'url' && url) {
            onProcess({ type: 'url', payload: url, acknowledged: true, cropStyle, category });
        } else if (mode === 'file' && file) {
            onProcess({ type: 'file', payload: file, acknowledged: true, cropStyle, category });
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
                {youtubeUrlEnabled && (
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
                )}
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

                <div className="mt-5 p-4 bg-white/5 rounded-xl border border-white/5">
                    <label htmlFor="category" className="flex items-center gap-2 text-xs text-zinc-500 mb-3 uppercase tracking-wider">
                        <Tag size={12} />
                        Kategori Konten
                    </label>
                    <select
                        id="category"
                        value={category}
                        onChange={(e) => setCategory(e.target.value)}
                        className="w-full bg-black/30 border border-white/10 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-primary/50 transition-colors"
                    >
                        <option value="general">Umum (Auto-Detect)</option>
                        <option value="podcast">Podcast / Diskusi</option>
                        <option value="podcast_comedy">Podcast Komedi</option>
                        <option value="tutorial">Tutorial / Edukasi</option>
                        <option value="gaming">Gaming</option>
                        <option value="reaction">Reaksi</option>
                        <option value="interview">Wawancara</option>
                        <option value="news">Berita</option>
                    </select>
                    <p className="text-[10px] text-zinc-600 mt-2">
                        Semua kategori menggunakan multi-pass AI (scout → judge) untuk hasil terbaik.
                        {category === 'general' && ' AI akan mendeteksi jenis konten otomatis.'}
                    </p>
                </div>

                <div className="mt-5 p-4 bg-white/5 rounded-xl border border-white/5">
                    <p className="text-xs text-zinc-500 mb-3 uppercase tracking-wider">Crop Style</p>
                    <div className="flex gap-2">
                        <button
                            type="button"
                            onClick={() => setCropStyle('blur_bars')}
                            className={`flex-1 flex items-center gap-2 p-3 rounded-lg text-sm transition-all ${cropStyle === 'blur_bars'
                                ? 'bg-primary/20 text-primary border border-primary/40'
                                : 'bg-white/5 text-zinc-400 hover:text-white border border-transparent'
                                }`}
                        >
                            <Film size={16} />
                            <div className="text-left">
                                <div className="font-medium">Blur Bars</div>
                                <div className="text-[10px] opacity-60">Full content, sharp quality</div>
                            </div>
                        </button>
                        <button
                            type="button"
                            onClick={() => setCropStyle('auto')}
                            className={`flex-1 flex items-center gap-2 p-3 rounded-lg text-sm transition-all ${cropStyle === 'auto'
                                ? 'bg-primary/20 text-primary border border-primary/40'
                                : 'bg-white/5 text-zinc-400 hover:text-white border border-transparent'
                                }`}
                        >
                            <Crop size={16} />
                            <div className="text-left">
                                <div className="font-medium">AI Crop &amp; Track</div>
                                <div className="text-[10px] opacity-60">Smart cropping, follows speaker</div>
                            </div>
                        </button>
                    </div>
                </div>

                <label className="flex items-start gap-2 mt-5 text-xs text-zinc-400 cursor-pointer select-none">
                    <input
                        type="checkbox"
                        checked={acknowledged}
                        onChange={(e) => setAcknowledged(e.target.checked)}
                        className="mt-0.5 accent-primary cursor-pointer"
                    />
                    <span>
                        I confirm I own this content or have the rights to process it. I am responsible for any content I submit. See our <a href="/#legal" target="_blank" rel="noopener noreferrer" className="text-primary underline" onClick={(e) => e.stopPropagation()}>Terms & Privacy</a>.
                    </span>
                </label>

                <button
                    type="submit"
                    disabled={isProcessing || !acknowledged || (mode === 'url' && !url) || (mode === 'file' && !file)}
                    className="w-full btn-primary mt-4 flex items-center justify-center gap-2"
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
