import React, { useState, useEffect, useRef } from 'react';
import { Columns2, Lock, Unlock } from 'lucide-react';

const TRIPLE_LAYOUTS = [
    [
        { crop: { x: 0.33, y: 0, w: 0.33, h: 0.5 }, dest: { x: 0, y: 0, w: 1, h: 0.5 }, label: 'top (wide)' },
        { crop: { x: 0, y: 0, w: 0.33, h: 0.5 }, dest: { x: 0, y: 0.5, w: 0.5, h: 0.5 }, label: 'bottom left' },
        { crop: { x: 0.66, y: 0, w: 0.33, h: 0.5 }, dest: { x: 0.5, y: 0.5, w: 0.5, h: 0.5 }, label: 'bottom right' }
    ],
    [
        { crop: { x: 0, y: 0, w: 0.33, h: 0.5 }, dest: { x: 0, y: 0, w: 0.5, h: 0.5 }, label: 'top left' },
        { crop: { x: 0.66, y: 0, w: 0.33, h: 0.5 }, dest: { x: 0.5, y: 0, w: 0.5, h: 0.5 }, label: 'top right' },
        { crop: { x: 0.33, y: 0, w: 0.33, h: 0.5 }, dest: { x: 0, y: 0.5, w: 1, h: 0.5 }, label: 'bottom (wide)' }
    ],
    [
        { crop: { x: 0, y: 0, w: 0.33, h: 0.5 }, dest: { x: 0, y: 0, w: 1, h: 0.3333 }, label: 'top' },
        { crop: { x: 0.33, y: 0, w: 0.33, h: 0.5 }, dest: { x: 0, y: 0.3333, w: 1, h: 0.3333 }, label: 'middle' },
        { crop: { x: 0.66, y: 0, w: 0.33, h: 0.5 }, dest: { x: 0, y: 0.6666, w: 1, h: 0.3333 }, label: 'bottom' }
    ]
];

const SPLIT_LAYOUTS = [
    [
        { crop: { x: 0, y: 0, w: 0.5, h: 1 }, dest: { x: 0, y: 0, w: 1, h: 0.5 }, label: 'top' },
        { crop: { x: 0.5, y: 0, w: 0.5, h: 1 }, dest: { x: 0, y: 0.5, w: 1, h: 0.5 }, label: 'bottom' }
    ],
    [
        { crop: { x: 0, y: 0, w: 1, h: 0.5 }, dest: { x: 0, y: 0, w: 1, h: 0.5 }, label: 'top' },
        { crop: { x: 0, y: 0.5, w: 1, h: 0.5 }, dest: { x: 0, y: 0.5, w: 1, h: 0.5 }, label: 'bottom' }
    ]
];

function CustomBox({ crop, label, onChange, parentRef, interactive }) {
    const startRef = useRef(null);

    const handlePointerDown = (e, mode) => {
        if (!interactive) return;
        e.stopPropagation();
        e.preventDefault();
        const parentRect = parentRef.current.getBoundingClientRect();
        const x = e.touches ? e.touches[0].clientX : e.clientX;
        const y = e.touches ? e.touches[0].clientY : e.clientY;
        startRef.current = { x, y, cropX: crop.x, cropY: crop.y, cropW: crop.w, cropH: crop.h, mode, parentRect };
        
        const move = (eMove) => {
            const currentX = eMove.touches ? eMove.touches[0].clientX : eMove.clientX;
            const currentY = eMove.touches ? eMove.touches[0].clientY : eMove.clientY;
            const dx = (currentX - startRef.current.x) / startRef.current.parentRect.width;
            const dy = (currentY - startRef.current.y) / startRef.current.parentRect.height;
            
            let newCrop = { ...crop };
            if (startRef.current.mode === 'move') {
                newCrop.x = Math.max(0, Math.min(1 - newCrop.w, startRef.current.cropX + dx));
                newCrop.y = Math.max(0, Math.min(1 - newCrop.h, startRef.current.cropY + dy));
            } else if (startRef.current.mode === 'resize') {
                let w = startRef.current.cropW + dx;
                let h = startRef.current.cropH + dy;
                w = Math.max(0.1, Math.min(1 - startRef.current.cropX, w));
                h = Math.max(0.1, Math.min(1 - startRef.current.cropY, h));
                newCrop.w = w;
                newCrop.h = h;
            }
            onChange(newCrop);
        };
        
        const up = () => {
            window.removeEventListener('mousemove', move);
            window.removeEventListener('mouseup', up);
            window.removeEventListener('touchmove', move);
            window.removeEventListener('touchend', up);
        };
        window.addEventListener('mousemove', move);
        window.addEventListener('mouseup', up);
        window.addEventListener('touchmove', move);
        window.addEventListener('touchend', up);
    };

    return (
        <div 
            className={`absolute border-2 transition-colors ${interactive ? 'border-brass bg-[color:var(--color-brass)]/20 cursor-move hover:bg-[color:var(--color-brass)]/30' : 'border-brass/50 bg-transparent pointer-events-none'}`}
            style={{ 
                left: `${crop.x * 100}%`, 
                top: `${crop.y * 100}%`, 
                width: `${crop.w * 100}%`, 
                height: `${crop.h * 100}%` 
            }}
            onMouseDown={(e) => handlePointerDown(e, 'move')}
            onTouchStart={(e) => handlePointerDown(e, 'move')}
        >
            <span className={`absolute top-1 left-1 text-[10px] px-1 rounded lowercase shadow-sm ${interactive ? 'bg-brass text-paper' : 'bg-brass/70 text-paper/80'}`}>
                {label}
            </span>
            {interactive && (
                <div 
                    className="absolute bottom-0 right-0 w-3 h-3 bg-brass cursor-se-resize"
                    onMouseDown={(e) => handlePointerDown(e, 'resize')}
                    onTouchStart={(e) => handlePointerDown(e, 'resize')}
                />
            )}
        </div>
    );
}

export default function GlobalLayoutSetup({ mode, url, file, layoutType, onChange }) {
    const [videoId, setVideoId] = useState('');
    const [localVideoUrl, setLocalVideoUrl] = useState('');
    const [layoutIdx, setLayoutIdx] = useState(0);
    const [customBoxes, setCustomBoxes] = useState([]);
    const [boxesLocked, setBoxesLocked] = useState(false); // If true, boxes are disabled so user can seek video
    const containerRef = useRef(null);

    const activeLayouts = layoutType === 'triple' ? TRIPLE_LAYOUTS : SPLIT_LAYOUTS;

    // Reset layouts when type changes
    useEffect(() => {
        setCustomBoxes(activeLayouts[0]);
        setLayoutIdx(0);
        // By default, if video is loaded, we might want to start with unlocked video to let them seek
        setBoxesLocked(true); // Default to letting them seek the video first
    }, [layoutType]);

    // Handle video URL / file
    useEffect(() => {
        if (mode === 'url' && url) {
            const match = url.match(/(?:youtu\.be\/|youtube\.com\/(?:embed\/|v\/|watch\?v=|watch\?.+&v=|shorts\/|live\/))([a-zA-Z0-9_-]{11})/);
            if (match && match[1]) {
                setVideoId(match[1]);
            } else {
                setVideoId('');
            }
            setLocalVideoUrl('');
        } else if (mode === 'file' && file) {
            const vidUrl = URL.createObjectURL(file);
            setLocalVideoUrl(vidUrl);
            setVideoId('');
            return () => URL.revokeObjectURL(vidUrl);
        } else {
            setVideoId('');
            setLocalVideoUrl('');
        }
    }, [mode, url, file]);

    // Update parent
    useEffect(() => {
        if (customBoxes && customBoxes.length > 0) {
            onChange(customBoxes);
        }
    }, [customBoxes, onChange]);

    const swapLayout = (e) => {
        e.preventDefault();
        e.stopPropagation();
        const nextIdx = (layoutIdx + 1) % activeLayouts.length;
        setLayoutIdx(nextIdx);
        setCustomBoxes(activeLayouts[nextIdx]);
    };

    const hasVideo = videoId || localVideoUrl;

    return (
        <div className="mt-4 border border-rule rounded-card overflow-hidden bg-paper animate-fade">
            <div className="p-3 bg-paper3 border-b border-rule flex items-center justify-between">
                <span className="text-xs font-medium text-ink flex items-center gap-2">
                    <Columns2 size={14} className="rotate-90 text-brass" /> 
                    Global {layoutType === 'triple' ? 'Triple' : 'Split'} Layout Setup
                </span>
                <div className="flex items-center gap-4">
                    {hasVideo && (
                        <button 
                            type="button"
                            onClick={() => setBoxesLocked(!boxesLocked)} 
                            className={`text-[11px] flex items-center gap-1 hover:underline transition-colors ${boxesLocked ? 'text-amber-500' : 'text-emerald-500'}`}
                        >
                            {boxesLocked ? <><Unlock size={12}/> Click here to arrange boxes</> : <><Lock size={12}/> Finish & Lock Boxes</>}
                        </button>
                    )}
                    <button 
                        type="button"
                        onClick={swapLayout} 
                        className="text-[11px] text-brass hover:underline transition-colors font-medium cursor-pointer"
                    >
                        Swap Layout ({layoutIdx + 1}/{activeLayouts.length})
                    </button>
                </div>
            </div>
            
            <div className="relative w-full aspect-video bg-black select-none overflow-hidden" ref={containerRef}>
                {videoId ? (
                    <iframe 
                        className="absolute inset-0 w-full h-full"
                        src={`https://www.youtube.com/embed/${videoId}?rel=0&modestbranding=1`}
                        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                        allowFullScreen
                    ></iframe>
                ) : localVideoUrl ? (
                    <video 
                        src={localVideoUrl}
                        controls
                        className="absolute inset-0 w-full h-full object-contain"
                    />
                ) : (
                    <div className="absolute inset-0 flex flex-col items-center justify-center text-muted/70 pointer-events-none p-4 text-center">
                        <p className="text-xs font-medium">Paste a YouTube link or upload a video above to see its video player here.</p>
                        <p className="text-[10px] mt-1 text-muted/50">You can still position and resize the {layoutType === 'triple' ? '3' : '2'} split boxes below in advance.</p>
                    </div>
                )}
                
                {/* The overlay layer for boxes */}
                <div className={`absolute inset-0 ${!boxesLocked ? 'pointer-events-auto z-10' : 'pointer-events-none z-10'}`}>
                    {customBoxes.map((box, i) => (
                        <CustomBox
                            key={i}
                            crop={box.crop}
                            label={box.label}
                            interactive={!boxesLocked || !hasVideo} // Always interactive if no video
                            onChange={(newCrop) => {
                                const updated = [...customBoxes];
                                updated[i] = { ...updated[i], crop: newCrop };
                                setCustomBoxes(updated);
                            }}
                            parentRef={containerRef}
                        />
                    ))}
                </div>
            </div>
            <div className="p-2.5 text-[11px] text-muted leading-relaxed flex items-center justify-between bg-paper3 border-t border-rule">
                <span>
                    {!boxesLocked || !hasVideo
                        ? "Boxes are unlocked. Drag boxes to move, drag bottom-right corner to resize." 
                        : "Video is unlocked. Play or scrub the video to find the perfect frame, then click 'Click here to arrange boxes'."}
                </span>
            </div>
        </div>
    );
}
