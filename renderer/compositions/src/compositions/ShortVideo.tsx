import React from "react";
import { AbsoluteFill } from "remotion";
import { Video } from "@remotion/media";
import type { ShortVideoProps } from "../lib/types";
import { Subtitles } from "./Subtitles";
import { HookOverlay } from "./HookOverlay";
import { VideoEffects } from "./VideoEffects";

/**
 * Main composition that layers all post-processing on top of the base video.
 * Uses @remotion/media Video for browser-side rendering compatibility.
 */
export const ShortVideo: React.FC<Record<string, unknown>> = (rawProps) => {
  const { videoUrl, subtitles, hook, effects } =
    rawProps as unknown as ShortVideoProps;
  return (
    <AbsoluteFill style={{ backgroundColor: "#000" }}>
      {/* Layer 1: Base video with optional zoom/color effects */}
      <VideoEffects config={effects}>
        <Video
          src={videoUrl}
          style={{ width: "100%", height: "100%", objectFit: "cover" }}
        />
      </VideoEffects>

      {/* Layer 2: Animated subtitles */}
      {subtitles && <Subtitles config={subtitles} />}

      {/* Layer 3: Hook text overlay */}
      {hook && <HookOverlay config={hook} />}
    </AbsoluteFill>
  );
};
