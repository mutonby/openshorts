import { useEffect, useRef, useState } from 'react';
import { Activity } from 'lucide-react';
import { getApiUrl } from '../config';
import { apiFetch } from '../lib/api';

const YOUTUBE_ORIGIN = 'https://www.youtube.com';

function youtubeId(url) {
  const match = url.match(/(?:youtu\.be\/|\/embed\/|\/shorts\/|[?&]v=)([\w-]{11})(?:[/?&#]|$)/);
  return match?.[1] || null;
}

export default function ProcessingAnimation({ media, syncedTime, isSyncedPlaying, syncTrigger }) {
  const [source, setSource] = useState(null);
  const videoRef = useRef(null);
  const iframeRef = useRef(null);
  const videoId = media.type === 'url' ? youtubeId(media.payload) : null;
  const videoSrc = source?.media === media ? source.url : null;

  useEffect(() => {
    if (media.type === 'file') {
      const url = URL.createObjectURL(media.payload);
      setSource({ media, url });
      return () => URL.revokeObjectURL(url);
    }
    if (media.type !== 'server') return undefined;

    // Restored sessions store the plain path. Get a fresh signed source URL.
    let cancelled = false;
    const jobId = media.payload.split('/').pop();
    apiFetch('/api/source-url/' + jobId)
      .then((res) => res.ok ? res.json() : null)
      .catch(() => null)
      .then((data) => {
        if (!cancelled) setSource({ media, url: getApiUrl(data?.url || media.payload) });
      });
    return () => { cancelled = true; };
  }, [media]);

  useEffect(() => {
    const playing = isSyncedPlaying && Number.isFinite(syncedTime);
    const video = videoRef.current;
    if (video) {
      if (playing) {
        video.currentTime = syncedTime;
        video.muted = true;
        video.play().catch(() => {});
      } else {
        video.pause();
      }
    }

    const frame = iframeRef.current?.contentWindow;
    if (frame) {
      const send = (func, args = []) => {
        frame.postMessage(JSON.stringify({ event: 'command', func, args }), YOUTUBE_ORIGIN);
      };
      if (playing) send('seekTo', [syncedTime, true]);
      send(playing ? 'playVideo' : 'pauseVideo');
    }
  }, [isSyncedPlaying, syncedTime, syncTrigger, videoId, videoSrc]);

  if (!videoId && !videoSrc) return null;

  return (
    <div className="relative w-full aspect-video rounded-card overflow-hidden bg-black border border-rule2 mb-4">
      {videoId ? (
        <iframe
          ref={iframeRef}
          className="w-full h-full border-0"
          src={YOUTUBE_ORIGIN + '/embed/' + videoId + '?mute=1&rel=0&enablejsapi=1'}
          title="Source video"
          allow="autoplay; encrypted-media; picture-in-picture"
        />
      ) : (
        <video
          ref={videoRef}
          src={videoSrc}
          className="w-full h-full object-contain"
          preload="metadata"
          controls={!isSyncedPlaying}
          muted
          playsInline
        />
      )}
      {isSyncedPlaying && (
        <div className="absolute top-2.5 right-2.5 badge-brass bg-black/70 pointer-events-none">
          <Activity size={12} aria-hidden="true" /> Synced to clip
        </div>
      )}
    </div>
  );
}
