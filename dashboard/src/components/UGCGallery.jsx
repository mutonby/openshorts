import React, { useState, useEffect, useRef } from 'react';
import { Film, Download, Copy, Check, ExternalLink, Loader2, Play, User } from 'lucide-react';
import { getApiUrl } from '../config';
import SegmentedControl from './ui/SegmentedControl';

export default function UGCGallery() {
  const [tab, setTab] = useState('videos');
  const [videos, setVideos] = useState([]);
  const [avatars, setAvatars] = useState([]);
  const [loading, setLoading] = useState(true);
  const [copied, setCopied] = useState('');

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(getApiUrl('/api/saasshorts/gallery?limit=100')).then(r => r.ok ? r.json() : { videos: [] }),
      fetch(getApiUrl('/api/saasshorts/actor-gallery')).then(r => r.ok ? r.json() : { images: [] }),
    ])
      .then(([vData, aData]) => {
        setVideos(vData.videos || []);
        setAvatars(aData.images || []);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  const handleCopy = (text, id) => {
    navigator.clipboard.writeText(text);
    setCopied(id);
    setTimeout(() => setCopied(''), 2000);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 size={24} className="animate-spin text-brass" />
        <span className="ml-2 text-muted lowercase">Loading gallery...</span>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="eyebrow mb-1">04 · UGC GALLERY</p>
          <h2 className="font-display lowercase text-2xl md:text-3xl text-ink">ugc gallery</h2>
          <p className="readout mt-2">{videos.length} videos · {avatars.length} avatars</p>
        </div>
        <a
          href={getApiUrl('/gallery')}
          target="_blank"
          rel="noopener noreferrer"
          className="btn-quiet text-xs shrink-0"
        >
          <ExternalLink size={13} /> public gallery
        </a>
      </div>

      {/* Tabs */}
      <div className="max-w-xs">
        <SegmentedControl
          size="sm"
          value={tab}
          onChange={setTab}
          options={[
            { value: 'videos', label: `Videos (${videos.length})`, icon: <Film size={14} /> },
            { value: 'avatars', label: `Avatars (${avatars.length})`, icon: <User size={14} /> },
          ]}
        />
      </div>

      {/* Videos Tab */}
      {tab === 'videos' && (
        videos.length === 0 ? (
          <div className="text-center py-16">
            <Film size={40} className="mx-auto text-muted opacity-40 mb-3" />
            <p className="text-sm text-muted lowercase">No videos yet. Generate one from AI Shorts.</p>
          </div>
        ) : (
          <div className="grid grid-cols-3 sm:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-3">
            {videos.map((video) => (
              <VideoCard key={video.video_id} video={video} copied={copied} onCopy={handleCopy} />
            ))}
          </div>
        )
      )}

      {/* Avatars Tab */}
      {tab === 'avatars' && (
        avatars.length === 0 ? (
          <div className="text-center py-16">
            <User size={40} className="mx-auto text-muted opacity-40 mb-3" />
            <p className="text-sm text-muted lowercase">No avatars yet. Generate actors from AI Shorts.</p>
          </div>
        ) : (
          <div className="grid grid-cols-3 sm:grid-cols-4 lg:grid-cols-6 gap-3">
            {avatars.map((avatar, i) => (
              <AvatarCard key={avatar.key || i} avatar={avatar} copied={copied} onCopy={handleCopy} />
            ))}
          </div>
        )
      )}
    </div>
  );
}

function AvatarCard({ avatar, copied, onCopy }) {
  return (
    <div className="group card overflow-hidden transition-colors hover:border-rule2">
      <div className="aspect-[3/4] bg-black">
        <img src={avatar.url} alt="Avatar" className="w-full h-full object-cover" />
      </div>
      <div className="p-2 space-y-1">
        {avatar.description ? (
          <div className="relative pr-4">
            <p className="text-micro text-muted line-clamp-2">{avatar.description}</p>
            <button
              onClick={() => onCopy(avatar.description, `avatar-${avatar.key}`)}
              className="absolute top-0 right-0 p-0.5 text-muted hover:text-brass transition-colors"
              title="Copy prompt"
            >
              {copied === `avatar-${avatar.key}` ? <Check size={10} className="text-ok" /> : <Copy size={10} />}
            </button>
          </div>
        ) : (
          <p className="text-micro text-muted opacity-60 lowercase">No description</p>
        )}
        <a
          href={avatar.url}
          download
          className="block text-center text-micro lowercase bg-paper3 hover:brightness-110 text-muted hover:text-ink2 py-1 rounded-full transition-all"
        >
          <Download size={10} className="inline mr-0.5" />Download
        </a>
      </div>
    </div>
  );
}

function VideoCard({ video, copied, onCopy }) {
  const videoRef = useRef(null);
  const [playing, setPlaying] = useState(false);

  const handleMouseEnter = () => {
    if (videoRef.current) {
      videoRef.current.play().catch(() => {});
      setPlaying(true);
    }
  };

  const handleMouseLeave = () => {
    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.currentTime = 0;
      setPlaying(false);
    }
  };

  const mode = video.video_mode;
  const caption = video.caption || '';
  const hashtags = (video.hashtags || []).join(' ');

  return (
    <div className="group card overflow-hidden transition-colors hover:border-rule2">
      <div
        className="relative aspect-[9/16] bg-black cursor-pointer"
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
      >
        <video
          ref={videoRef}
          src={video.video_url}
          poster={video.actor_url}
          muted
          playsInline
          preload="metadata"
          className="w-full h-full object-cover"
        />
        {!playing && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/20">
            <Play size={20} className="text-white/70" />
          </div>
        )}
        <div className="absolute top-1.5 right-1.5">
          <span className={`${mode === 'lowcost' ? 'badge-ok' : 'badge-brass'} bg-black/70`}>
            {mode === 'lowcost' ? 'LOW COST' : 'PREMIUM'}
          </span>
        </div>
      </div>

      <div className="p-2 space-y-1">
        <h3 className="text-xs font-semibold text-ink truncate">{video.title || 'Untitled'}</h3>
        <p className="readout">
          {video.duration?.toFixed(0)}s · ${video.cost_estimate?.total?.toFixed(2) || '?'}
        </p>
        {caption && (
          <div className="relative pr-4">
            <p className="text-micro text-muted line-clamp-2">{caption}</p>
            <button
              onClick={() => onCopy(`${caption}\n${hashtags}`, `caption-${video.video_id}`)}
              className="absolute top-0 right-0 p-0.5 text-muted hover:text-brass transition-colors"
              title="Copy caption"
            >
              {copied === `caption-${video.video_id}` ? <Check size={10} className="text-ok" /> : <Copy size={10} />}
            </button>
          </div>
        )}
        <div className="flex gap-1 pt-0.5">
          <a
            href={video.video_url}
            download
            className="flex-1 text-center text-micro lowercase bg-paper3 hover:brightness-110 text-muted hover:text-ink2 py-1 rounded-full transition-all"
          >
            <Download size={10} className="inline mr-0.5" />Download
          </a>
          <a
            href={getApiUrl(`/video/${video.video_id}`)}
            target="_blank"
            rel="noopener noreferrer"
            className="flex-1 text-center text-micro lowercase bg-paper3 hover:brightness-110 text-brass py-1 rounded-full transition-all"
          >
            <ExternalLink size={10} className="inline mr-0.5" />View
          </a>
        </div>
      </div>
    </div>
  );
}
