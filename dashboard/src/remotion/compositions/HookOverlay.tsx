import React from "react";
import {
  AbsoluteFill,
  Sequence,
  useCurrentFrame,
  useVideoConfig,
  spring,
  interpolate,
} from "remotion";
import type { HookConfig } from "../lib/types";
import {
  notoSerifFontFace,
  montserratFontFace,
  antonFontFace,
  HOOK_FONTS,
} from "../lib/fonts";

interface HookOverlayProps {
  config: HookConfig;
}

const SIZE_SCALE: Record<string, number> = {
  S: 0.8,
  M: 1.0,
  L: 1.3,
};

// Percentages must match hooks.py's overlay_y math (top 20% / bottom 70%),
// or the preview drifts from what the server path renders.
const POSITION_STYLE: Record<string, React.CSSProperties> = {
  top: { top: "20%", bottom: "auto" },
  center: { top: "50%", bottom: "auto", transform: "translateY(-50%)" },
  bottom: { top: "70%", bottom: "auto" },
};

// Must mirror hooks.py HOOK_STYLES (the server-side FFmpeg fallback).
interface HookLook {
  box: string | null;
  text: string;
  outlinePx: number;
  shadow: boolean;
  /** One rounded box per line (hooks.py _draw_pills) instead of one card. */
  pills?: boolean;
}

const HOOK_LOOKS: Record<string, HookLook> = {
  pill: { box: "rgba(255, 255, 255, 0.98)", text: "#000000", outlinePx: 0, shadow: false, pills: true },
  classic: { box: "rgba(255, 255, 255, 0.94)", text: "#000000", outlinePx: 0, shadow: true },
  dark: { box: "rgba(18, 18, 20, 0.92)", text: "#FFFFFF", outlinePx: 0, shadow: true },
  yellow: { box: "rgba(255, 214, 0, 0.96)", text: "#000000", outlinePx: 0, shadow: true },
  red: { box: "rgba(220, 38, 38, 0.96)", text: "#FFFFFF", outlinePx: 0, shadow: true },
  outline: { box: null, text: "#FFFFFF", outlinePx: 8, shadow: false },
  outline_yellow: { box: null, text: "#FFD600", outlinePx: 8, shadow: false },
};

export const HookOverlay: React.FC<HookOverlayProps> = ({ config }) => {
  const { fps, durationInFrames } = useVideoConfig();
  const { cta } = config;
  const closingFrames = cta
    ? Math.min(durationInFrames, Math.max(1, Math.round(cta.displayDurationSec * fps)))
    : 0;
  const closingStart = durationInFrames - closingFrames;
  const hookFrames = Math.min(
    closingStart,
    config.displayDurationSec == null ? durationInFrames : Math.round(config.displayDurationSec * fps),
  );

  return (
    <AbsoluteFill>
      <style>{notoSerifFontFace + montserratFontFace + antonFontFace}</style>
      {config.text && hookFrames > 0 && (
        <Sequence from={0} durationInFrames={hookFrames} layout="none">
          <HookBox config={config} displayFrames={hookFrames} />
        </Sequence>
      )}
      {cta && (
        <Sequence from={closingStart} durationInFrames={closingFrames} layout="none">
          <HookBox config={{ ...config, ...cta, entranceAnimation: "none" }}
            displayFrames={closingFrames} />
        </Sequence>
      )}
    </AbsoluteFill>
  );
};

interface HookBoxProps {
  config: HookConfig;
  displayFrames: number;
}

const HookBox: React.FC<HookBoxProps> = ({ config, displayFrames }) => {
  const frame = useCurrentFrame();
  const { fps, width } = useVideoConfig();
  const scale = SIZE_SCALE[config.size] ?? 1.0;

  // Entrance animation
  let animOpacity = 1;
  let animScale = 1;
  let animTranslateY = 0;

  switch (config.entranceAnimation) {
    case "spring": {
      const prog = spring({
        frame,
        fps,
        config: { mass: 0.8, stiffness: 200, damping: 15 },
        durationInFrames: 20,
      });
      animScale = interpolate(prog, [0, 1], [0.7, 1]);
      animOpacity = interpolate(prog, [0, 1], [0, 1]);
      break;
    }
    case "fade": {
      animOpacity = interpolate(frame, [0, 15], [0, 1], {
        extrapolateRight: "clamp",
      });
      break;
    }
    case "slide-up": {
      const prog = spring({
        frame,
        fps,
        config: { mass: 1, stiffness: 150, damping: 18 },
        durationInFrames: 20,
      });
      animTranslateY = interpolate(prog, [0, 1], [60, 0]);
      animOpacity = interpolate(prog, [0, 1], [0, 1]);
      break;
    }
    default:
      break;
  }

  // Exit fade (last 15 frames)
  const fadeOutStart = Math.max(0, displayFrames - 15);
  if (config.entranceAnimation !== "none" && frame > fadeOutStart) {
    animOpacity *= interpolate(frame, [fadeOutStart, displayFrames], [1, 0], {
      extrapolateLeft: "clamp",
      extrapolateRight: "clamp",
    });
  }

  const positionStyle = POSITION_STYLE[config.position] ?? POSITION_STYLE.top;
  const look = HOOK_LOOKS[config.style ?? "pill"] ?? HOOK_LOOKS.pill;

  // Typeface and size as hooks.py: the chosen font, else the style's own;
  // font size = factor x the 90%-of-width box.
  const typeface = HOOK_FONTS[config.font ?? (look.pills ? "montserrat" : "serif")] ?? HOOK_FONTS.serif;
  const fontSize = Math.round(width * 0.9 * typeface.factor * scale);
  const outlinePx = Math.round(look.outlinePx * scale);

  if (look.pills) {
    // box-decoration-break: clone gives every wrapped line its own padded,
    // rounded box, the same stack hooks.py draws for the FFmpeg path.
    return (
      <div
        style={{
          position: "absolute",
          left: 0,
          right: 0,
          display: "flex",
          justifyContent: "center",
          ...positionStyle,
        }}
      >
        <div
          style={{
            opacity: animOpacity,
            transform: `scale(${animScale}) translateY(${animTranslateY}px)`,
            maxWidth: "90%",
            textAlign: "center",
            lineHeight: 1.62,
          }}
        >
          <span
            style={{
              fontFamily: typeface.family,
              fontSize,
              fontWeight: typeface.weight,
              color: look.text,
              backgroundColor: look.box ?? "transparent",
              padding: `${Math.round(fontSize * 0.2)}px ${Math.round(fontSize * 0.48)}px`,
              borderRadius: Math.round(fontSize * 0.36),
              WebkitBoxDecorationBreak: "clone",
              boxDecorationBreak: "clone",
              wordBreak: "break-word",
            }}
          >
            {config.text}
          </span>
        </div>
      </div>
    );
  }

  return (
    <div
      style={{
        position: "absolute",
        left: 0,
        right: 0,
        display: "flex",
        justifyContent: "center",
        ...positionStyle,
      }}
    >
      <div
        style={{
          opacity: animOpacity,
          transform: `scale(${animScale}) translateY(${animTranslateY}px)`,
          maxWidth: "90%",
          backgroundColor: look.box ?? "transparent",
          borderRadius: 20,
          padding: look.box ? `${25 * scale}px ${30 * scale}px` : 0,
          boxShadow: look.shadow ? "5px 5px 15px rgba(0, 0, 0, 0.25)" : "none",
          textAlign: "center",
        }}
      >
        <span
          style={{
            fontFamily: typeface.family,
            fontSize,
            fontWeight: typeface.weight,
            color: look.text,
            lineHeight: 1.4,
            wordBreak: "break-word",
            ...(outlinePx > 0
              ? {
                  WebkitTextStroke: `${outlinePx}px #000000`,
                  paintOrder: "stroke fill",
                }
              : {}),
          }}
        >
          {config.text}
        </span>
      </div>
    </div>
  );
};
