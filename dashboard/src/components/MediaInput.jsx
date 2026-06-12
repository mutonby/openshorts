import React, { useState, useEffect, useRef } from 'react';
import { Youtube, Upload, FileVideo, X, Plus, Layers, Trash2 } from 'lucide-react';
import { getApiUrl } from '../config';

export default function MediaInput({ onProcess, onBatchProcess, isProcessing }) {
    const [youtubeUrlEnabled, setYoutubeUrlEnabled] = useState(true);
    const [mode, setMode] = useState('url'); // 'url' | 'file' | 'batch'
    const [url, setUrl] = useState('');
    const [file, setFile] = useState(null);
    const [acknowledged, setAcknowledged] = useState(false);

    // Batch state
    const [batchItems, setBatchItems] = useState([]);
    const [batchUrl, setBatchUrl] = useState('');
    const batchFileRef = useRef(null);

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
            onProcess({ type: 'url', payload: url, acknowledged: true });
        } else if (mode === 'file' && file) {
            onProcess({ type: 'file', payload: file, acknowledged: true });
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            setFile(e.dataTransfer.files[0]);
            setMode('file');
        }
    };

    // Batch helpers
    const addBatchUrl = () => {
        const trimmed = batchUrl.trim();
        if (!trimmed) return;
        setBatchItems(prev => [...prev, { id: crypto.randomUUID(), type: 'url', payload: trimmed, name: trimmed }]);
        setBatchUrl('');
    };

    const addBatchFiles = (files) => {
        const newItems = Array.from(files).map(f => ({
            id: crypto.randomUUID(), type: 'file', payload: f, name: f.name
        }));
        setBatchItems(prev => [...prev, ...newItems]);
    };

    const removeBatchItem = (id) => setBatchItems(prev => prev.filter(item => item.id !== id));

    const handleBatchDrop = (e) => {
        e.preventDefault();
        if (e.dataTransfer.files?.length) addBatchFiles(e.dataTransfer.files);
    };

    const handleBatchSubmit = (e) => {
        e.preventDefault();
        if (!acknowledged || batchItems.length === 0) return;
        onBatchProcess && onBatchProcess(batchItems);
    };

    return (
        <div className="bg-surface border border-white/5 rounded-2xl p-6 animate-[fadeIn_0.6s_ease-out]">
            {/* Tab bar */}
            <div className="flex gap-4 mb-6 border-b border-white/5 pb-4">
                {youtubeUrlEnabled && (
                    <button
                        onClick={() => setMode('url')}
                        className={`flex items-center gap-2 pb-2 px-2 transition-all ${mode === 'url'
                            ? 'text-primary border-b-2 border-primary -mb-[17px]'
                            : 'text-zinc-400 hover:text-white'}`}
                    >
                        <Youtube size={18} />
                        YouTube URL
                    </button>
                )}
                <button
                    onClick={() => setMode('file')}
                    className={`flex items-center gap-2 pb-2 px-2 transition-all ${mode === 'file'
                        ? 'text-primary border-b-2 border-primary -mb-[17px]'
                        : 'text-zinc-400 hover:text-white'}`}
                >
                    <Upload size={18} />
                    Upload File
                </button>
                <button
                    onClick={() => setMode('batch')}
                    className={`flex items-center gap-2 pb-2 px-2 transition-all ${mode === 'batch'
                        ? 'text-primary border-b-2 border-primary -mb-[17px]'
                        : 'text-zinc-400 hover:text-white'}`}
                >
                    <Layers size={18} />
                    Batch
                    {batchItems.length > 0 && (
                        <span className="bg-primary text-white text-[10px] font-bold px-1.5 py-0.5 rounded-full leading-none">
                            {batchItems.length}
                        </span>
                    )}
                </button>
            </div>

            {/* Single URL form */}
            {mode !== 'batch' && (
                <form onSubmit={handleSubmit}>
                    {mode === 'url' ? (
                        <input
                            type="url"
                            value={url}
                            onChange={(e) => setUrl(e.target.value)}
                            placeholder="https://www.youtube.com/watch?v=..."
                            className="input-field"
                            required
                        />
                    ) : (
                        <div
                            className={`border-2 border-dashed rounded-xl p-8 text-center transition-all ${file ? 'border-primary/50 bg-primary/5' : 'border-zinc-700 hover:border-zinc-500 bg-white/5'}`}
                            onDragOver={(e) => e.preventDefault()}
                            onDrop={handleDrop}
                        >
                            {file ? (
                                <div className="flex items-center justify-center gap-3 text-white">
                                    <FileVideo className="text-primary" />
                                    <span className="font-medium">{file.name}</span>
                                    <button type="button" onClick={() => setFile(null)} className="p-1 hover:bg-white/10 rounded-full">
                                        <X size={16} />
                                    </button>
                                </div>
                            ) : (
                                <label className="cursor-pointer block">
                                    <input type="file" accept="video/*" onChange={(e) => setFile(e.target.files?.[0] || null)} className="hidden" />
                                    <Upload className="mx-auto mb-3 text-zinc-500" size={24} />
                                    <p className="text-zinc-400">Click to upload or drag and drop</p>
                                    <p className="text-xs text-zinc-600 mt-1">MP4, MOV up to 500MB</p>
                                </label>
                            )}
                        </div>
                    )}

                    <label className="flex items-start gap-2 mt-5 text-xs text-zinc-400 cursor-pointer select-none">
                        <input type="checkbox" checked={acknowledged} onChange={(e) => setAcknowledged(e.target.checked)} className="mt-0.5 accent-primary cursor-pointer" />
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
                            <><div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />Processing Video...</>
                        ) : 'Generate Clips'}
                    </button>
                </form>
            )}

            {/* Batch form */}
            {mode === 'batch' && (
                <form onSubmit={handleBatchSubmit} className="space-y-4">
                    {/* Add URL row */}
                    {youtubeUrlEnabled && (
                        <div className="flex gap-2">
                            <input
                                type="url"
                                value={batchUrl}
                                onChange={(e) => setBatchUrl(e.target.value)}
                                onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); addBatchUrl(); } }}
                                placeholder="https://www.youtube.com/watch?v=..."
                                className="input-field flex-1 text-sm"
                            />
                            <button
                                type="button"
                                onClick={addBatchUrl}
                                disabled={!batchUrl.trim()}
                                className="px-3 py-2 bg-primary/20 hover:bg-primary/30 disabled:opacity-40 text-primary border border-primary/30 rounded-lg transition-colors flex items-center gap-1 text-sm shrink-0"
                            >
                                <Plus size={15} /> Add URL
                            </button>
                        </div>
                    )}

                    {/* File drop zone */}
                    <div
                        className="border-2 border-dashed border-zinc-700 hover:border-zinc-500 rounded-xl p-4 text-center cursor-pointer transition-colors bg-white/5"
                        onDragOver={(e) => e.preventDefault()}
                        onDrop={handleBatchDrop}
                        onClick={() => batchFileRef.current?.click()}
                    >
                        <input ref={batchFileRef} type="file" accept="video/*" multiple className="hidden"
                            onChange={(e) => { if (e.target.files?.length) addBatchFiles(e.target.files); e.target.value = ''; }} />
                        <Upload className="mx-auto mb-1 text-zinc-600" size={18} />
                        <p className="text-xs text-zinc-500">Drop files or click to add videos</p>
                    </div>

                    {/* Queue list */}
                    {batchItems.length > 0 && (
                        <div className="space-y-1.5 max-h-48 overflow-y-auto custom-scrollbar pr-1">
                            {batchItems.map((item, idx) => (
                                <div key={item.id} className="flex items-center gap-2 bg-black/30 border border-white/5 rounded-lg px-3 py-2">
                                    <span className="text-[10px] text-zinc-600 font-mono w-4 shrink-0">{idx + 1}</span>
                                    {item.type === 'url'
                                        ? <Youtube size={13} className="text-red-400 shrink-0" />
                                        : <FileVideo size={13} className="text-blue-400 shrink-0" />}
                                    <span className="flex-1 text-xs text-zinc-300 truncate" title={item.name}>{item.name}</span>
                                    <button type="button" onClick={() => removeBatchItem(item.id)} className="text-zinc-600 hover:text-red-400 transition-colors shrink-0">
                                        <X size={13} />
                                    </button>
                                </div>
                            ))}
                        </div>
                    )}

                    {batchItems.length === 0 && (
                        <p className="text-xs text-zinc-600 text-center py-2">Add URLs or files above to build your batch queue.</p>
                    )}

                    <label className="flex items-start gap-2 text-xs text-zinc-400 cursor-pointer select-none">
                        <input type="checkbox" checked={acknowledged} onChange={(e) => setAcknowledged(e.target.checked)} className="mt-0.5 accent-primary cursor-pointer" />
                        <span>
                            I confirm I own this content or have the rights to process it. See our <a href="/#legal" target="_blank" rel="noopener noreferrer" className="text-primary underline" onClick={(e) => e.stopPropagation()}>Terms & Privacy</a>.
                        </span>
                    </label>

                    <div className="flex gap-2">
                        <button
                            type="button"
                            onClick={() => setBatchItems([])}
                            disabled={batchItems.length === 0}
                            className="px-3 py-2.5 bg-white/5 hover:bg-white/10 disabled:opacity-40 text-zinc-400 rounded-lg text-xs transition-colors flex items-center gap-1.5 border border-white/5"
                        >
                            <Trash2 size={13} /> Clear
                        </button>
                        <button
                            type="submit"
                            disabled={isProcessing || !acknowledged || batchItems.length === 0}
                            className="flex-1 btn-primary flex items-center justify-center gap-2"
                        >
                            {isProcessing
                                ? <><div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />Processing...</>
                                : <><Layers size={16} />Process {batchItems.length > 0 ? `${batchItems.length} Videos` : 'Batch'}</>}
                        </button>
                    </div>
                </form>
            )}
        </div>
    );
}
