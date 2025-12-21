import React, { useEffect, useState, useRef } from 'react';
import { Scan, Scissors, Activity, Radio, CheckCircle, Play } from 'lucide-react';

const ProcessingAnimation = ({ media, isComplete, syncedTime, isSyncedPlaying, syncTrigger }) => {
  const [videoSrc, setVideoSrc] = useState(null);
  const [isYouTube, setIsYouTube] = useState(false);
  const videoRef = useRef(null);
  const iframeRef = useRef(null);

  useEffect(() => {
    if (!media) return;

    if (media.type === 'file') {
      const url = URL.createObjectURL(media.payload);
      setVideoSrc(url);
      return () => URL.revokeObjectURL(url);
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
        // Stop Sync: Pause
        videoRef.current.pause();

        // If Analysis Complete and we just stopped syncing (paused clip), we might want to return to ambient loop?
        // User issue: "ahora el video loop original que sale oscurito ahora bien pero es una imagen estatica no se está reproduciendo el video en bucle como antes"
        // This means when NOT synced, it should loop.
        
        // HOWEVER, previously user asked: "si pauso el video preview y lo reanudo el video original se vuelve al princpio en vez de contnuar igual"
        // This implies: 
        // 1. If I PAUSE the clip -> Left video should PAUSE (static image) so it can resume.
        // 2. If I STOP (or it finishes? or just idle state?) -> It should LOOP.
        
        // The problem is we only have onPause from the clip.
        // Maybe we need to distinguish "Pause" vs "Idle/Stop".
        // But currently we just get `isSyncedPlaying = false`.
        
        // If the user wants "resume from where left off", it MUST be static (paused).
        // If the user wants "loop when not playing", it MUST play.
        // These are contradictory for the "Paused" state.
        
        // BUT, maybe the "loop original" refers to the initial state BEFORE any clip is played?
        // OR when the clip finishes?
        
        // Let's look at the logic:
        // isSyncedPlaying is true ONLY when a clip is playing.
        // When clip pauses, isSyncedPlaying becomes false.
        
        // If we want it to loop "como antes", we should set it to loop.
        // But then we lose the "resume" position because it starts looping.
        // Unless... we only loop if we haven't started syncing yet? Or if explicitly reset?
        
        // Wait, the user said: "ahora el video loop original que sale oscurito ahora bien pero es una imagen estatica"
        // This likely refers to the state AFTER analysis is complete but BEFORE (or after) playing a clip.
        // If I haven't touched a clip yet, `isSyncedPlaying` is false.
        // In that case, it SHOULD be looping.
        
        // My previous change removed the "else { play loop }" block entirely.
        // I need to restore the loop for the IDLE state, but keep the PAUSE for the "paused clip" state?
        // That requires knowing WHY `isSyncedPlaying` is false.
        
        // Actually, if `isSyncedPlaying` is false, it means no clip is controlling it.
        // If I want it to loop in the background, I can just let it loop.
        // BUT if I play a clip later, it will jump to the sync time anyway (handled by the `if (isSyncedPlaying)` block).
        // The only issue is if I PAUSE the clip, `isSyncedPlaying` becomes false, and if I immediately start looping,
        // visually it might jump or start moving when it should be "paused".
        
        // Let's try this:
        // If `syncedTime` is 0 (or we track if we ever started syncing?), we loop.
        // But `syncedTime` updates on play.
        
        // Alternative interpretation: The user sees it static because I removed `videoRef.current.play()` in the else block.
        // If I put it back, it fixes the "loop" issue.
        // Does it break the "resume" issue?
        // "si pauso el video preview y lo reanudo el video original se vuelve al princpio en vez de contnuar igual"
        // If I pause the clip -> `isSyncedPlaying` = false.
        // If logic says -> Loop from 0.
        // Then I resume -> `isSyncedPlaying` = true -> Jump to `syncedTime`.
        // This actually SHOULD work fine for "resume", because `syncedTime` comes from the clip's current time.
        // The only visual glitch is that while paused, the left video is looping instead of frozen on the frame.
        
        // If the user accepts that "Paused Clip" = "Background Loop", then we are good.
        // If the user wants "Paused Clip" = "Frozen Frame" AND "Idle" = "Background Loop", we need more state.
        // But typically "Idle" implies we aren't focusing on a clip.
        
        // Let's restore the loop behavior because "video loop original... es una imagen estatica" sounds like a bug to them.
        
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
            //  iframeWindow.postMessage(JSON.stringify({ event: 'command', func: 'pauseVideo', args: [] }), '*'); // Removed pause to allow loop if needed, but YT embeds are tricky with custom loops via API.
            // For now, let's just pause YouTube as complex looping is harder without state.
             iframeWindow.postMessage(JSON.stringify({ event: 'command', func: 'pauseVideo', args: [] }), '*');
        }
    }
  }, [syncedTime, isSyncedPlaying, isYouTube, videoSrc, syncTrigger]);


  const getYouTubeId = (url) => {
    const regExp = /^.*(youtu.be\/|v\/|u\/\w\/|embed\/|watch\?v=|&v=)([^#&?]*).*/;
    const match = url.match(regExp);
    return (match && match[2].length === 11) ? match[2] : null;
  };

  const containerClasses = `relative w-full aspect-video rounded-xl overflow-hidden bg-black border border-white/10 shadow-2xl mb-8 group animate-[fadeIn_0.5s_ease-out] transition-all duration-500 
    ${isComplete && !isSyncedPlaying ? 'grayscale brightness-50' : ''} 
    ${isSyncedPlaying ? 'ring-2 ring-primary ring-offset-2 ring-offset-black shadow-primary/20' : ''}`;

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
          <div className="w-full h-full flex items-center justify-center bg-zinc-900">
             <div className="w-16 h-16 border-4 border-zinc-700 border-t-zinc-500 rounded-full animate-spin"></div>
          </div>
        )}
      </div>

      {/* Overlays - Hide when synced playing so user sees clean video */}
      {!isSyncedPlaying && !isComplete && (
        <>
            <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.03)_1px,transparent_1px)] bg-[size:40px_40px] z-10 pointer-events-none"></div>
            <div className="absolute left-0 w-full h-[2px] bg-primary shadow-[0_0_15px_2px_rgba(59,130,246,0.5)] animate-[scan_2.5s_linear_infinite] z-20 pointer-events-none"></div>
            <div className="absolute left-0 w-full h-[15%] bg-gradient-to-b from-primary/0 via-primary/5 to-primary/0 animate-[scan-overlay_2.5s_linear_infinite] z-10 pointer-events-none"></div>
        </>
      )}

      {/* HUD Elements - Hide when synced playing */}
      {!isSyncedPlaying && (
          <div className={`absolute top-4 left-4 z-30 flex items-center gap-2 px-3 py-1.5 backdrop-blur-md rounded-lg border text-xs font-mono font-bold uppercase transition-all duration-500 ${isComplete ? 'bg-green-500/10 border-green-500/20 text-green-400' : 'bg-black/60 border-primary/30 text-primary animate-pulse'}`}>
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
          <div className="absolute top-4 right-4 z-30 flex items-center gap-2 px-3 py-1.5 bg-black/60 backdrop-blur-md rounded-lg border border-white/10 text-white/50 text-[10px] font-mono">
            AI_MODEL: GEMINI-2.5-PRO
          </div>
      )}
      
      {/* Visual Flair */}
      {!isSyncedPlaying && !isComplete && (
          <div className="absolute inset-0 pointer-events-none z-20 overflow-hidden">
             <div className="absolute top-0 bottom-0 left-[35%] w-[1px] bg-yellow-500/20 border-r border-dashed border-yellow-500/40"></div>
             <div className="absolute top-0 bottom-0 right-[35%] w-[1px] bg-yellow-500/20 border-l border-dashed border-yellow-500/40"></div>
             <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-12 h-12 border border-white/20 rounded-full flex items-center justify-center">
                <div className="w-1 h-1 bg-red-500 rounded-full animate-ping"></div>
             </div>
             <div className="absolute bottom-1/3 left-1/2 -translate-x-1/2 flex flex-col items-center justify-center gap-2 opacity-60">
                 <Scissors size={24} className="text-white/20" />
             </div>
          </div>
      )}

       {/* Synced Playing Indicator */}
       {isSyncedPlaying && (
           <div className="absolute top-4 right-4 z-30 flex items-center gap-2 px-3 py-1.5 bg-red-600/90 backdrop-blur text-white rounded-lg shadow-lg animate-pulse font-bold text-[10px] uppercase tracking-wider border border-white/20">
               <Activity size={12} /> Live Sync
           </div>
       )}
      
       {/* Bottom Info Bar */}
      {!isSyncedPlaying && !isComplete && (
          <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black/90 to-transparent z-30 flex justify-between items-end border-t border-white/5">
              <div className="font-mono text-[10px] text-primary/80 space-y-1">
                 <div className="flex items-center gap-2"><Activity size={10} className="animate-bounce" /> > ANALYSIS_THREAD_01: ACTIVE</div>
                 <div className="flex items-center gap-2"><Radio size={10} /> > AUDIO_TRANSCRIPT: PROCESSING</div>
              </div>
              <div className="flex gap-1">
                 <div className="w-1 h-3 bg-primary/40 animate-[pulse_0.5s_infinite]"></div>
                 <div className="w-1 h-5 bg-primary/60 animate-[pulse_0.7s_infinite]"></div>
                 <div className="w-1 h-2 bg-primary/30 animate-[pulse_0.4s_infinite]"></div>
                 <div className="w-1 h-4 bg-primary/80 animate-[pulse_0.6s_infinite]"></div>
                 <div className="w-1 h-3 bg-primary/50 animate-[pulse_0.5s_infinite]"></div>
              </div>
          </div>
      )}
    </div>
  );
};

export default ProcessingAnimation;