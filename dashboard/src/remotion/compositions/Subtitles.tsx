import React from "react";
import {
  AbsoluteFill,
  Sequence,
  useCurrentFrame,
  useVideoConfig,
  spring,
  interpolate,
} from "remotion";
import type { SubtitleConfig } from "../lib/types";
import { groupCaptionsIntoBlocks, getActiveWordIndex } from "../lib/captions";
import { getFontStack, antonFontFace, montserratFontFace } from "../lib/fonts";

interface SubtitlesProps {
  config: SubtitleConfig;
}

// Hebrew, Arabic, Syriac, Thaana, NKo and their presentation forms. A block in
// one of these scripts sets direction: rtl so the words run right to left
// (github issue #89).
const RTL_CHARS = /[\u0590-\u08FF\uFB1D-\uFDFF\uFE70-\uFEFF]/;

const POSITION_MAP: Record<string, React.CSSProperties> = {
  top: { top: "12%", bottom: "auto" },
  middle: { top: "45%", bottom: "auto" },
  bottom: { bottom: "10%", top: "auto" },
};

export const Subtitles: React.FC<SubtitlesProps> = ({ config }) => {
  const { fps } = useVideoConfig();
  const blocks = groupCaptionsIntoBlocks(
    config.captions,
    config.maxChars ?? 20,
    config.maxDurationMs ?? 2000,
    config.maxWords
  );

  return (
    <AbsoluteFill>
      <style>{antonFontFace + montserratFontFace}</style>
      {blocks.map((block, i) => {
        const startFrame = Math.round((block.startMs / 1000) * fps);
        const durationFrames = Math.max(
          1,
          Math.round(((block.endMs - block.startMs) / 1000) * fps)
        );

        return (
          <Sequence
            key={i}
            from={startFrame}
            durationInFrames={durationFrames}
            layout="none"
          >
            <SubtitleBlock
              block={block}
              config={config}
              blockStartMs={block.startMs}
            />
          </Sequence>
        );
      })}
    </AbsoluteFill>
  );
};

interface SubtitleBlockProps {
  block: ReturnType<typeof groupCaptionsIntoBlocks>[number];
  config: SubtitleConfig;
  blockStartMs: number;
}

const SubtitleBlock: React.FC<SubtitleBlockProps> = ({
  block,
  config,
  blockStartMs,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const { style, position } = config;

  // Current time relative to composition start (sequence-relative frame)
  const currentTimeMs = blockStartMs + (frame / fps) * 1000;
  const activeIndex = getActiveWordIndex(block.words, currentTimeMs);

  const positionStyle = POSITION_MAP[position] ?? POSITION_MAP.bottom;
  const fontStack = getFontStack(style.fontFamily);

  // Background box style
  const hasBg = style.bgOpacity > 0;
  const bgStyle: React.CSSProperties = hasBg
    ? {
        backgroundColor: `${style.bgColor}${Math.round(style.bgOpacity * 255)
          .toString(16)
          .padStart(2, "0")}`,
        borderRadius: 8,
        padding: "8px 16px",
      }
    : {};

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
      {/* Words are joined with a real space in the caption font, as the burn
          does (subtitles.generate_ass joins them with " "). A fixed flex gap
          was 8px at a ~144px font, so the preview read "KNEWMELIKE" where the
          clip says "KNEW ME LIKE". The container carries the font so that
          space has the right width. */}
      <div
        style={{
          textAlign: "center",
          direction: block.words.some((w) => RTL_CHARS.test(w.text)) ? "rtl" : "ltr",
          fontFamily: fontStack,
          fontSize: style.fontSize,
          fontWeight: 700,
          maxWidth: "85%",
          ...bgStyle,
        }}
      >
        {block.words.map((word, i) => (
          <React.Fragment key={i}>
            {i > 0 && " "}
            <WordSpan
              word={word.text}
              isActive={i === activeIndex}
              hidden={!!style.reveal && activeIndex >= 0 && i > activeIndex}
              style={style}
              fontStack={fontStack}
              animation={style.animation}
              frame={frame}
              fps={fps}
              wordStartMs={word.startMs}
              blockStartMs={blockStartMs}
            />
          </React.Fragment>
        ))}
      </div>
    </div>
  );
};

interface WordSpanProps {
  word: string;
  isActive: boolean;
  hidden: boolean;
  style: SubtitleConfig["style"];
  fontStack: string;
  animation: SubtitleConfig["style"]["animation"];
  frame: number;
  fps: number;
  wordStartMs: number;
  blockStartMs: number;
}

const WordSpan: React.FC<WordSpanProps> = ({
  word,
  isActive,
  hidden,
  style,
  fontStack,
  animation,
  frame,
  fps,
  wordStartMs,
  blockStartMs,
}) => {
  const wordStartFrame = Math.round(
    ((wordStartMs - blockStartMs) / 1000) * fps
  );

  let transform = "";
  let color = style.fontColor;
  let extraStyle: React.CSSProperties = {};

  // Dim inactive words toward the backend's opaque scaled color (matches the
  // burned ASS look; not CSS opacity).
  if (!isActive && style.baseOpacity != null && style.baseOpacity < 1) {
    const m = /^#?([0-9a-fA-F]{6})$/.exec(style.fontColor || "#FFFFFF");
    if (m) {
      const scale = 0.35 + 0.65 * style.baseOpacity;
      const [r, g, b] = [0, 2, 4].map((i) =>
        Math.round(parseInt(m[1].slice(i, i + 2), 16) * scale)
      );
      color = `rgb(${r}, ${g}, ${b})`;
    }
  }

  if (isActive) {
    color = style.highlightColor;

    switch (animation) {
      case "pop": {
        const scale = spring({
          frame: frame - wordStartFrame,
          fps,
          config: { mass: 0.5, stiffness: 300, damping: 12 },
          durationInFrames: 10,
        });
        // Same range as the burned pop (\fscx90 -> \fscx108). A transform
        // takes no layout space, so the old 1.25 spilled over the space on
        // both sides and glued the active word to its neighbours.
        const scaleValue = interpolate(scale, [0, 1], [0.9, 1.08]);
        transform = `scale(${scaleValue})`;
        break;
      }
      case "karaoke": {
        extraStyle = {
          backgroundColor: style.highlightColor,
          color: style.highlightTextColor || style.bgColor || "#000000",
          borderRadius: 4,
          padding: "2px 10px",
        };
        break;
      }
      case "word-highlight": {
        extraStyle = {
          textShadow: `0 0 12px ${style.highlightColor}, 0 0 24px ${style.highlightColor}40`,
        };
        break;
      }
      default:
        break;
    }
  }

  // Text stroke via textShadow (CSS paint-order not reliable in Remotion)
  const strokeShadow =
    style.borderWidth > 0
      ? [
          `${style.borderWidth}px 0 0 ${style.borderColor}`,
          `-${style.borderWidth}px 0 0 ${style.borderColor}`,
          `0 ${style.borderWidth}px 0 ${style.borderColor}`,
          `0 -${style.borderWidth}px 0 ${style.borderColor}`,
        ].join(", ")
      : "none";
  // Burn units -> preview px, same scale as the font size (3.85 * 0.85).
  const dropShadow = style.shadow
    ? `${style.shadow * 3}px ${style.shadow * 3}px ${style.shadow * 2}px rgba(0,0,0,0.55)`
    : "";
  const outline = strokeShadow === "none" ? "" : strokeShadow;

  return (
    <span
      style={{
        fontFamily: fontStack,
        fontSize: style.fontSize,
        fontWeight: 700,
        color: animation === "karaoke" && isActive ? undefined : color,
        textShadow:
          [outline, dropShadow, animation !== "karaoke" ? extraStyle.textShadow : ""]
            .filter(Boolean)
            .join(", ") || "none",
        visibility: hidden ? "hidden" : "visible",
        transform,
        display: "inline-block",
        transition: "none",
        textTransform: style.uppercase ? "uppercase" : "none",
        ...extraStyle,
      }}
    >
      {word}
    </span>
  );
};
