import React from "react";
import { useCurrentFrame, useVideoConfig, interpolate } from "remotion";
import type { EffectsConfig, EffectSegment } from "../lib/types";

interface VideoEffectsProps {
  config: EffectsConfig | null;
  children: React.ReactNode;
}

/**
 * Wraps children (typically <OffthreadVideo>) with dynamic CSS transforms and filters.
 * Interpolates smoothly between effect segments.
 */
export const VideoEffects: React.FC<VideoEffectsProps> = ({
  config,
  children,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  if (!config || config.segments.length === 0) {
    return <>{children}</>;
  }

  const currentTimeSec = frame / fps;
  const { zoom, centerX, centerY, brightness, contrast, saturate } =
    getInterpolatedValues(config.segments, currentTimeSec, frame, fps);

  const filterParts: string[] = [];
  if (brightness !== 1) filterParts.push(`brightness(${brightness})`);
  if (contrast !== 1) filterParts.push(`contrast(${contrast})`);
  if (saturate !== 1) filterParts.push(`saturate(${saturate})`);
  const filterStr = filterParts.length > 0 ? filterParts.join(" ") : "none";

  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        overflow: "hidden",
        position: "relative",
      }}
    >
      <div
        style={{
          width: "100%",
          height: "100%",
          transform: `scale(${zoom})`,
          transformOrigin: `${centerX * 100}% ${centerY * 100}%`,
          filter: filterStr,
        }}
      >
        {children}
      </div>
    </div>
  );
};

interface InterpolatedValues {
  zoom: number;
  centerX: number;
  centerY: number;
  brightness: number;
  contrast: number;
  saturate: number;
}

function getInterpolatedValues(
  segments: EffectSegment[],
  timeSec: number,
  frame: number,
  fps: number
): InterpolatedValues {
  // Default values (no effect)
  const defaults: InterpolatedValues = {
    zoom: 1,
    centerX: 0.5,
    centerY: 0.5,
    brightness: 1,
    contrast: 1,
    saturate: 1,
  };

  // Find active segment
  const active = segments.find(
    (s) => timeSec >= s.startSec && timeSec < s.endSec
  );

  if (!active) {
    // Check if we're transitioning between segments (smooth fade)
    const prev = segments.filter((s) => s.endSec <= timeSec).pop();
    const next = segments.find((s) => s.startSec > timeSec);

    if (prev && next) {
      const gap = next.startSec - prev.endSec;
      if (gap < 1.0) {
        // Short gap: interpolate between prev and next
        const progress = (timeSec - prev.endSec) / gap;
        return lerpSegments(prev, next, progress);
      }
    }

    // Transition out from previous segment
    if (prev) {
      const fadeOutDuration = 0.3; // seconds
      const elapsed = timeSec - prev.endSec;
      if (elapsed < fadeOutDuration) {
        const progress = elapsed / fadeOutDuration;
        return lerpToDefaults(prev, progress, defaults);
      }
    }

    // Transition into next segment
    if (next) {
      const fadeInDuration = 0.3;
      const remaining = next.startSec - timeSec;
      if (remaining < fadeInDuration) {
        const progress = 1 - remaining / fadeInDuration;
        return lerpFromDefaults(next, progress, defaults);
      }
    }

    return defaults;
  }

  // Inside active segment: smooth entrance/exit at edges
  const segDuration = active.endSec - active.startSec;
  const transitionSec = Math.min(0.3, segDuration * 0.15);

  const startFrame = Math.round(active.startSec * fps);
  const endFrame = Math.round(active.endSec * fps);
  const transitionFrames = Math.round(transitionSec * fps);

  // Entrance ease
  const entranceFactor = interpolate(
    frame,
    [startFrame, startFrame + transitionFrames],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Exit ease
  const exitFactor = interpolate(
    frame,
    [endFrame - transitionFrames, endFrame],
    [1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const factor = Math.min(entranceFactor, exitFactor);

  return {
    zoom: lerp(1, active.zoom, factor),
    centerX: lerp(0.5, active.zoomCenterX, factor),
    centerY: lerp(0.5, active.zoomCenterY, factor),
    brightness: lerp(1, active.brightness, factor),
    contrast: lerp(1, active.contrast, factor),
    saturate: lerp(1, active.saturate, factor),
  };
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function lerpSegments(
  a: EffectSegment,
  b: EffectSegment,
  t: number
): InterpolatedValues {
  return {
    zoom: lerp(a.zoom, b.zoom, t),
    centerX: lerp(a.zoomCenterX, b.zoomCenterX, t),
    centerY: lerp(a.zoomCenterY, b.zoomCenterY, t),
    brightness: lerp(a.brightness, b.brightness, t),
    contrast: lerp(a.contrast, b.contrast, t),
    saturate: lerp(a.saturate, b.saturate, t),
  };
}

function lerpToDefaults(
  seg: EffectSegment,
  t: number,
  defaults: InterpolatedValues
): InterpolatedValues {
  return {
    zoom: lerp(seg.zoom, defaults.zoom, t),
    centerX: lerp(seg.zoomCenterX, defaults.centerX, t),
    centerY: lerp(seg.zoomCenterY, defaults.centerY, t),
    brightness: lerp(seg.brightness, defaults.brightness, t),
    contrast: lerp(seg.contrast, defaults.contrast, t),
    saturate: lerp(seg.saturate, defaults.saturate, t),
  };
}

function lerpFromDefaults(
  seg: EffectSegment,
  t: number,
  defaults: InterpolatedValues
): InterpolatedValues {
  return {
    zoom: lerp(defaults.zoom, seg.zoom, t),
    centerX: lerp(defaults.centerX, seg.zoomCenterX, t),
    centerY: lerp(defaults.centerY, seg.zoomCenterY, t),
    brightness: lerp(defaults.brightness, seg.brightness, t),
    contrast: lerp(defaults.contrast, seg.contrast, t),
    saturate: lerp(defaults.saturate, seg.saturate, t),
  };
}
