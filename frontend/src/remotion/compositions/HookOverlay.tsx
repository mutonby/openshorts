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
import { notoSerifFontFace, NOTO_SERIF_FONT_FAMILY } from "../lib/fonts";

interface HookOverlayProps {
  config: HookConfig;
}

const SIZE_SCALE: Record<string, number> = {
  S: 0.8,
  M: 1.0,
  L: 1.3,
};

const POSITION_STYLE: Record<string, React.CSSProperties> = {
  top: { top: "18%", bottom: "auto" },
  center: { top: "50%", bottom: "auto", transform: "translateY(-50%)" },
  bottom: { top: "68%", bottom: "auto" },
};

export const HookOverlay: React.FC<HookOverlayProps> = ({ config }) => {
  const { fps } = useVideoConfig();
  const displayFrames = Math.round(config.displayDurationSec * fps);

  return (
    <AbsoluteFill>
      <style>{notoSerifFontFace}</style>
      <Sequence from={0} durationInFrames={displayFrames} layout="none">
        <HookBox config={config} displayFrames={displayFrames} />
      </Sequence>
    </AbsoluteFill>
  );
};

interface HookBoxProps {
  config: HookConfig;
  displayFrames: number;
}

const HookBox: React.FC<HookBoxProps> = ({ config, displayFrames }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
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
  const fadeOutStart = displayFrames - 15;
  if (frame > fadeOutStart) {
    animOpacity *= interpolate(frame, [fadeOutStart, displayFrames], [1, 0], {
      extrapolateLeft: "clamp",
      extrapolateRight: "clamp",
    });
  }

  const positionStyle = POSITION_STYLE[config.position] ?? POSITION_STYLE.top;

  // Base font size: 5% of 1080 width (matches hooks.py logic)
  const baseFontSize = 1080 * 0.05;
  const fontSize = Math.round(baseFontSize * scale);

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
          backgroundColor: "rgba(255, 255, 255, 0.94)",
          borderRadius: 20,
          padding: `${25 * scale}px ${30 * scale}px`,
          boxShadow: "5px 5px 15px rgba(0, 0, 0, 0.25)",
          textAlign: "center",
        }}
      >
        <span
          style={{
            fontFamily: `'${NOTO_SERIF_FONT_FAMILY}', 'Noto Serif', Georgia, serif`,
            fontSize,
            fontWeight: 700,
            color: "#000000",
            lineHeight: 1.4,
            wordBreak: "break-word",
          }}
        >
          {config.text}
        </span>
      </div>
    </div>
  );
};
