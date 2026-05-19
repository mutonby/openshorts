import React, { useRef, useState, useEffect } from 'react';
import { Download, Youtube, Instagram, Video, Copy, Check, Play } from 'lucide-react';

export default function GalleryCard({ clip }) {
    const [copied, setCopied] = useState(null);
    const [isVisible, setIsVisible] = useState(false);
    const [hasLoaded, setHasLoaded] = useState(false);
    const cardRef = useRef(null);
    const videoRef = useRef(null);

    // Lazy loading with IntersectionObserver
    useEffect(() => {
        const observer = new IntersectionObserver(
            (entries) => {
                entries.forEach((entry) => {
                    if (entry.isIntersecting) {
                        setIsVisible(true);
                        // Once loaded, we don't need to observe anymore
                        observer.unobserve(entry.target);
                    }
                });
            },
            {
                rootMargin: '200px', // Start loading 200px before entering viewport
                threshold: 0.1
            }
        );

        if (cardRef.current) {
            observer.observe(cardRef.current);
        }

        return () => {
            if (cardRef.current) {
                observer.unobserve(cardRef.current);
            }
        };
    }, []);

    const handleCopy = (text, field) => {
        navigator.clipboard.writeText(text);
        setCopied(field);
        setTimeout(() => setCopied(null), 2000);
    };

    const handleDownload = async (e) => {
        e.preventDefault();
        try {
            const response = await fetch(clip.url);
            if (!response.ok) throw new Error('Download failed');
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = `clip_${clip.job_id}_${clip.index + 1}.mp4`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        } catch (err) {
            console.error('Download error:', err);
            window.open(clip.url, '_blank');
        }
    };

    return (
        <div
            ref={cardRef}
            className="bg-surface border border-white/5 rounded-xl overflow-hidden flex flex-col hover:border-white/10 transition-all group animate-[fadeIn_0.5s_ease-out]"
        >
            {/* Video Player - Lazy loaded */}
            <div className="aspect-[9/16] bg-black relative group/video">
                {isVisible ? (
                    <video
                        ref={videoRef}
                        src={clip.url}
                        controls
                        className="w-full h-full object-cover"
                        playsInline
                        preload="metadata"
                        onLoadedData={() => setHasLoaded(true)}
                    />
                ) : (
                    <div className="w-full h-full flex items-center justify-center bg-zinc-900">
                        <div className="w-12 h-12 rounded-full bg-white/10 flex items-center justify-center">
                            <Play size={24} className="text-white/50 ml-1" />
                        </div>
                    </div>
                )}
                <div className="absolute top-2 left-2">
                    <span className="bg-black/60 backdrop-blur-md text-white text-[10px] font-bold px-2 py-1 rounded-md border border-white/10 tracking-wide">
                        {new Date(clip.created_at).toLocaleDateString()}
                    </span>
                </div>
            </div>

            {/* Content & Details */}
            <div className="flex-1 p-4 flex flex-col bg-[#121214] min-w-0">
                <div className="mb-3">
                    <h3 className="text-sm font-bold text-white leading-tight line-clamp-2 mb-2 break-words" title={clip.title}>
                        {clip.title}
                    </h3>
                    <div className="flex flex-wrap gap-2 text-[10px] text-zinc-500 font-mono">
                        <span className="bg-white/5 px-1.5 py-0.5 rounded border border-white/5">{clip.duration.toFixed(1)}s</span>
                        <span className="bg-white/5 px-1.5 py-0.5 rounded border border-white/5 truncate max-w-[150px]" title={clip.job_id}>ID: {clip.job_id.substring(0, 8)}</span>
                    </div>
                </div>

                <div className="space-y-2 flex-1 overflow-y-auto custom-scrollbar max-h-[150px] pr-1 mb-3">
                    {/* YouTube Title */}
                    <div className="bg-black/20 rounded-lg p-2 border border-white/5 relative group/item">
                        <div className="flex items-center gap-1.5 text-[10px] font-bold text-red-400 mb-1 uppercase tracking-wider">
                            <Youtube size={10} className="shrink-0" /> YouTube Title
                        </div>
                        <p className="text-xs text-zinc-300 select-all line-clamp-2 hover:line-clamp-none transition-all">{clip.title}</p>
                        <button
                            onClick={() => handleCopy(clip.title, 'yt')}
                            className="absolute top-2 right-2 p-1 text-zinc-500 hover:text-white transition-colors opacity-0 group-hover/item:opacity-100"
                            title="Copy Title"
                        >
                            {copied === 'yt' ? <Check size={12} className="text-green-400" /> : <Copy size={12} />}
                        </button>
                    </div>

                    {/* TikTok / IG Caption */}
                    <div className="bg-black/20 rounded-lg p-2 border border-white/5 relative group/item">
                        <div className="flex items-center gap-1.5 text-[10px] font-bold text-zinc-400 mb-1 uppercase tracking-wider">
                            <Video size={10} className="text-cyan-400 shrink-0" />
                            <span className="text-zinc-600">/</span>
                            <Instagram size={10} className="text-pink-400 shrink-0" /> Caption
                        </div>
                        <p className="text-xs text-zinc-300 select-all line-clamp-3 hover:line-clamp-none transition-all cursor-pointer">
                            {clip.tiktok_desc || clip.insta_desc}
                        </p>
                        <button
                            onClick={() => handleCopy(clip.tiktok_desc || clip.insta_desc, 'caption')}
                            className="absolute top-2 right-2 p-1 text-zinc-500 hover:text-white transition-colors opacity-0 group-hover/item:opacity-100"
                            title="Copy Caption"
                        >
                            {copied === 'caption' ? <Check size={12} className="text-green-400" /> : <Copy size={12} />}
                        </button>
                    </div>
                </div>

                {/* Footer Action */}
                <button
                    onClick={handleDownload}
                    className="w-full py-2 bg-white/5 hover:bg-white/10 text-zinc-300 hover:text-white rounded-lg text-xs font-medium transition-colors flex items-center justify-center gap-2 border border-white/5"
                >
                    <Download size={14} className="shrink-0" /> Download Clip
                </button>
            </div>
        </div>
    );
}
