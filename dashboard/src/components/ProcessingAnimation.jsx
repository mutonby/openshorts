import React, { useEffect, useState, useRef } from 'react';
import { Scan, Scissors, Activity, Radio, CheckCircle } from 'lucide-react';
import { getApiUrl } from '../config';

const ProcessingAnimation = ({ media, isComplete, syncedTime, isSyncedPlaying, syncTrigger }) => {
  const [videoSrc, setVideoSrc] = useState(null);
  const [isYouTube, setIsYouTube] = useState(false);
  const videoRef = useRef(null);
  const iframeRef = useRef(null);

  useEffect(() => {
    if (!media) return;

    if (media.type === 'file') {
      const url = URL.createObjectURL(media.payload);
      setIsYouTube(false);
      setVideoSrc(url);
      return () => URL.revokeObjectURL(url);
    } else if (media.type === 'server') {
      // Uploaded source served from the backend (survives a page reload).
      setIsYouTube(false);
      setVideoSrc(getApiUrl(media.payload));
    } else if (media.type === 'url') {
      setIsYouTube(true);
      const videoId = getYouTubeId(media.payload);
      setVideoSrc(videoId);
    }
  }, [media]);

  // Handle Sync Playback for Local Video
  useEffect(() => {
    if (!isYouTube && videoRef.current) {
      if (isSyncedPlaying) {
        // Sync Mode: Seek to time and Play
        videoRef.current.currentTime = syncedTime;
        videoRef.current.play().catch(e => console.log("Auto-play prevented", e));
        videoRef.current.loop = false;
        videoRef.current.muted = true; // Keep muted to avoid double audio with clip
      } else {
        // Stop Sync: Pause. Once analysis is complete, resume the ambient loop.
        videoRef.current.pause();

        if (isComplete) {
             videoRef.current.loop = true;
             videoRef.current.play().catch(e => console.log("Ambient play prevented", e));
        }
      }
    }
  }, [syncedTime, isSyncedPlaying, isYouTube, isComplete, syncTrigger]);

  // Handle Sync Playback for YouTube (Basic Iframe Control via PostMessage)
  useEffect(() => {
    if (isYouTube && iframeRef.current && videoSrc) {
        const iframeWindow = iframeRef.current.contentWindow;
        if (isSyncedPlaying) {
             // Seek and Play
             iframeWindow.postMessage(JSON.stringify({ event: 'command', func: 'seekTo', args: [syncedTime, true] }), '*');
             iframeWindow.postMessage(JSON.stringify({ event: 'command', func: 'playVideo', args: [] }), '*');
        } else {
             // Pause
             iframeWindow.postMessage(JSON.stringify({ event: 'command', func: 'pauseVideo', args: [] }), '*');
        }
    }
  }, [syncedTime, isSyncedPlaying, isYouTube, videoSrc, syncTrigger]);


  const getYouTubeId = (url) => {
    const regExp = /^.*(youtu.be\/|v\/|u\/\w\/|embed\/|watch\?v=|&v=)([^#&?]*).*/;
    const match = url.match(regExp);
    return (match && match[2].length === 11) ? match[2] : null;
  };

  const containerClasses = `relative w-full aspect-video rounded-card overflow-hidden bg-black border border-rule2 mb-8 group animate-fade transition-all duration-500
    ${isComplete && !isSyncedPlaying ? 'grayscale brightness-50' : ''}
    ${isSyncedPlaying ? 'ring-2 ring-brass ring-offset-2 ring-offset-black' : ''}`;

  const getVideoOpacityClass = () => {
    if (isSyncedPlaying) return 'opacity-100'; // Playing: Full visibility
    if (isComplete) return 'opacity-30';       // Idle Result: Darker
    return 'opacity-40 grayscale group-hover:grayscale-0'; // Processing: Dark + Grayscale effect
  };

  return (
    <div className={containerClasses}>
      {/* Video Layer */}
      <div className={`absolute inset-0 transition-all duration-700 ${getVideoOpacityClass()}`}>
        {isYouTube && videoSrc ? (
            <iframe
            ref={iframeRef}
            className={`w-full h-full ${isSyncedPlaying ? '' : 'pointer-events-none scale-110'}`}
            // Add enablejsapi=1 for postMessage control
            src={`https://www.youtube.com/embed/${videoSrc}?autoplay=1&mute=1&controls=0&loop=1&playlist=${videoSrc}&modestbranding=1&showinfo=0&rel=0&enablejsapi=1`}
            title="Processing Video"
            frameBorder="0"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
          />
        ) : videoSrc ? (
          <video
            ref={videoRef}
            src={videoSrc}
            className="w-full h-full object-cover"
            autoPlay
            muted
            loop
            playsInline
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center bg-paper">
             <div className="w-16 h-16 border-4 border-paper3 border-t-muted rounded-full animate-spin"></div>
          </div>
        )}
      </div>

      {/* Overlays - Hide when synced playing so user sees clean video */}
      {!isSyncedPlaying && !isComplete && (
        <>
            <div className="absolute inset-0 bg-[linear-gradient(var(--rule-blueprint)_1px,transparent_1px),linear-gradient(90deg,var(--rule-blueprint)_1px,transparent_1px)] bg-[size:40px_40px] z-10 pointer-events-none"></div>
            <div className="absolute left-0 w-full h-[2px] bg-brass shadow-[0_0_15px_2px_var(--color-glow)] animate-[scan_2.5s_linear_infinite] z-20 pointer-events-none"></div>
            <div className="absolute left-0 w-full h-[15%] bg-[var(--color-paper-emit)] animate-[scan-overlay_2.5s_linear_infinite] z-10 pointer-events-none"></div>
        </>
      )}

      {/* HUD Elements - Hide when synced playing */}
      {!isSyncedPlaying && (
          <div className={`absolute top-4 left-4 z-30 flex items-center gap-2 px-3 py-1.5 rounded-full bg-black/70 readout transition-colors duration-500 ${isComplete ? 'text-ok' : 'text-brass animate-pulse'}`}>
            {isComplete ? (
                <>
                    <CheckCircle size={14} /> Analysis Complete
                </>
            ) : (
                <>
                    <Scan size={14} /> Scanning Content...
                </>
            )}
          </div>
      )}

      {!isSyncedPlaying && !isComplete && (
          <div className="absolute top-4 right-4 z-30 flex items-center gap-2 px-3 py-1.5 bg-black/70 rounded-full readout">
            VIRAL_DETECTION: ACTIVE
          </div>
      )}

      {/* Visual Flair */}
      {!isSyncedPlaying && !isComplete && (
          <div className="absolute inset-0 pointer-events-none z-20 overflow-hidden">
             <div className="absolute top-0 bottom-0 left-[35%] w-px border-r border-dashed border-brass opacity-40"></div>
             <div className="absolute top-0 bottom-0 right-[35%] w-px border-l border-dashed border-brass opacity-40"></div>
             <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-12 h-12 border border-rule2 rounded-full flex items-center justify-center">
                <div className="w-1 h-1 bg-brass rounded-full animate-ping"></div>
             </div>
             <div className="absolute bottom-1/3 left-1/2 -translate-x-1/2 flex flex-col items-center justify-center gap-2 opacity-60">
                 <Scissors size={24} className="text-white/20" />
             </div>
          </div>
      )}

       {/* Synced Playing Indicator */}
       {isSyncedPlaying && (
           <div className="absolute top-4 right-4 z-30 badge-brass bg-black/70 animate-pulse">
               <Activity size={12} /> Live Sync
           </div>
       )}

       {/* Bottom Info Bar */}
      {!isSyncedPlaying && !isComplete && (
          <div className="absolute bottom-0 left-0 right-0 p-4 bg-black/70 z-30 flex justify-between items-end border-t border-rule">
              <div className="readout text-brass space-y-1">
                 <div className="flex items-center gap-2"><Activity size={10} className="animate-pulse" /> {'>'} ANALYSIS_THREAD_01: ACTIVE</div>
                 <div className="flex items-center gap-2"><Radio size={10} /> {'>'} AUDIO_TRANSCRIPT: PROCESSING</div>
              </div>
              <div className="flex gap-1">
                 <div className="w-1 h-3 bg-brass opacity-40 animate-[pulse_0.5s_infinite]"></div>
                 <div className="w-1 h-5 bg-brass opacity-60 animate-[pulse_0.7s_infinite]"></div>
                 <div className="w-1 h-2 bg-brass opacity-30 animate-[pulse_0.4s_infinite]"></div>
                 <div className="w-1 h-4 bg-brass opacity-80 animate-[pulse_0.6s_infinite]"></div>
                 <div className="w-1 h-3 bg-brass opacity-50 animate-[pulse_0.5s_infinite]"></div>
              </div>
          </div>
      )}
    </div>
  );
};

export default ProcessingAnimation;
