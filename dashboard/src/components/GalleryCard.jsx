import React, { useRef, useState, useEffect } from 'react';
import { Download, Copy, Check, Play } from 'lucide-react';

export default function GalleryCard({ clip }) {
    const [copied, setCopied] = useState(null);
    const [isVisible, setIsVisible] = useState(false);
    const [_hasLoaded, setHasLoaded] = useState(false);
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

        const node = cardRef.current;
        if (node) {
            observer.observe(node);
        }

        return () => {
            if (node) {
                observer.unobserve(node);
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
            className="card card-hover overflow-hidden flex flex-col group animate-fade"
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
                    <div className="w-full h-full flex items-center justify-center bg-paper">
                        <div className="w-12 h-12 rounded-full bg-paper3 flex items-center justify-center">
                            <Play size={24} className="text-muted ml-1" />
                        </div>
                    </div>
                )}
                <div className="absolute top-2 left-2">
                    <span className="readout bg-black/70 px-2 py-1 rounded-full">
                        {new Date(clip.created_at).toLocaleDateString()}
                    </span>
                </div>
            </div>

            {/* Content & Details */}
            <div className="flex-1 p-4 flex flex-col min-w-0">
                <div className="mb-3">
                    <h3 className="text-sm font-semibold text-ink leading-tight line-clamp-2 mb-2 break-words" title={clip.title}>
                        {clip.title}
                    </h3>
                    <div className="flex flex-wrap gap-2">
                        <span className="readout bg-paper3 px-1.5 py-0.5 rounded-full">{clip.duration.toFixed(1)}s</span>
                        <span className="readout bg-paper3 px-1.5 py-0.5 rounded-full truncate max-w-[150px]" title={clip.job_id}>ID: {clip.job_id.substring(0, 8)}</span>
                    </div>
                </div>

                <div className="space-y-2 flex-1 overflow-y-auto custom-scrollbar max-h-[150px] pr-1 mb-3">
                    {/* YouTube Title */}
                    <div className="bg-paper rounded-input p-2 relative group/item">
                        <p className="eyebrow mb-1">YouTube Title</p>
                        <p className="text-xs text-ink2 select-all line-clamp-2 hover:line-clamp-none transition-all">{clip.title}</p>
                        <button
                            onClick={() => handleCopy(clip.title, 'yt')}
                            className="absolute top-2 right-2 p-1 text-muted hover:text-brass transition-colors opacity-0 group-hover/item:opacity-100"
                            title="Copy Title"
                        >
                            {copied === 'yt' ? <Check size={12} className="text-ok" /> : <Copy size={12} />}
                        </button>
                    </div>

                    {/* TikTok / IG Caption */}
                    <div className="bg-paper rounded-input p-2 relative group/item">
                        <p className="eyebrow mb-1">TikTok · IG Caption</p>
                        <p className="text-xs text-ink2 select-all line-clamp-3 hover:line-clamp-none transition-all cursor-pointer">
                            {clip.tiktok_desc || clip.insta_desc}
                        </p>
                        <button
                            onClick={() => handleCopy(clip.tiktok_desc || clip.insta_desc, 'caption')}
                            className="absolute top-2 right-2 p-1 text-muted hover:text-brass transition-colors opacity-0 group-hover/item:opacity-100"
                            title="Copy Caption"
                        >
                            {copied === 'caption' ? <Check size={12} className="text-ok" /> : <Copy size={12} />}
                        </button>
                    </div>
                </div>

                {/* Footer Action */}
                <button
                    onClick={handleDownload}
                    className="btn-quiet w-full"
                >
                    <Download size={14} className="shrink-0" /> Download Clip
                </button>
            </div>
        </div>
    );
}
