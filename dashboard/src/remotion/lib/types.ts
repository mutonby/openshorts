import { z } from "zod";

// --- Word-level caption ---
export interface CaptionWord {
  text: string;
  startMs: number;
  endMs: number;
}

// --- Subtitle config ---
export type SubtitleAnimation = "none" | "word-highlight" | "pop" | "karaoke";
export type SubtitlePosition = "top" | "middle" | "bottom";

export interface SubtitleStyle {
  fontFamily: string;
  fontSize: number;
  fontColor: string;
  highlightColor: string;
  borderColor: string;
  borderWidth: number;
  bgColor: string;
  bgOpacity: number;
  animation: SubtitleAnimation;
  // Karaoke look: dim inactive words (0-1) and force uppercase.
  baseOpacity?: number;
  uppercase?: boolean;
  // Words not spoken yet stay invisible (keep their slot), as burned.
  reveal?: boolean;
  // Drop shadow depth in burn units (subtitles.generate_ass `shadow`).
  shadow?: number;
  // Text colour on the active-word box ("karaoke" animation).
  highlightTextColor?: string;
}

export interface SubtitleConfig {
  captions: CaptionWord[];
  position: SubtitlePosition;
  style: SubtitleStyle;
  // Line budget, mirrors the burn's max_chars / max_duration.
  maxChars?: number;
  maxWords?: number;
  maxDurationMs?: number;
}

// --- Hook config ---
// "auto": the server picks a spot off the faces and the captions (hook_placement.py);
// the browser preview and render, which cannot see faces, draw it at the top.
export type HookPosition = "auto" | "top" | "center" | "bottom";
export type HookSize = "S" | "M" | "L";
export type HookEntrance = "spring" | "fade" | "slide-up" | "none";
export type HookStyle =
  | "pill"
  | "classic"
  | "dark"
  | "yellow"
  | "red"
  | "outline"
  | "outline_yellow";

export type HookFont = "montserrat" | "anton" | "serif";

export interface HookDesign {
  shape: "pill" | "box" | "none";
  text_color: string;
  background_color: string;
  outline_color: string;
  outline_width: number;
  corner_radius: number;
  text_case: "as_written" | "uppercase";
}

export interface HookArtwork {
  src: string;
  width: number;
  height: number;
}

export interface HookConfig {
  text: string;
  position: HookPosition;
  size: HookSize;
  style?: HookStyle;
  /** montserrat | anton | serif; unset = the style's own typeface. */
  font?: HookFont;
  entranceAnimation: HookEntrance;
  displayDurationSec: number | null;
  design?: HookDesign;
  artwork?: { opening: HookArtwork | null; closing: HookArtwork | null };
  cta?: {
    text: string;
    position: HookPosition;
    size: HookSize;
    displayDurationSec: number;
  };
}

// --- Effects config ---
export interface EffectSegment {
  startSec: number;
  endSec: number;
  zoom: number;
  zoomCenterX: number;
  zoomCenterY: number;
  brightness: number;
  contrast: number;
  saturate: number;
}

export interface EffectsConfig {
  segments: EffectSegment[];
}

// --- Main composition props ---
export interface ShortVideoProps {
  videoUrl: string;
  durationInFrames: number;
  fps: number;
  width: number;
  height: number;
  subtitles: SubtitleConfig | null;
  hook: HookConfig | null;
  effects: EffectsConfig | null;
}

// --- Zod schemas for validation (used by render service) ---
export const captionWordSchema = z.object({
  text: z.string(),
  startMs: z.number(),
  endMs: z.number(),
});

export const subtitleStyleSchema = z.object({
  fontFamily: z.string(),
  fontSize: z.number(),
  fontColor: z.string(),
  highlightColor: z.string(),
  borderColor: z.string(),
  borderWidth: z.number(),
  bgColor: z.string(),
  bgOpacity: z.number().min(0).max(1),
  animation: z.enum(["none", "word-highlight", "pop", "karaoke"]),
});

export const subtitleConfigSchema = z.object({
  captions: z.array(captionWordSchema),
  position: z.enum(["top", "middle", "bottom"]),
  style: subtitleStyleSchema,
});

const hookArtworkSchema = z.object({
  src: z.string(),
  width: z.number().positive(),
  height: z.number().positive(),
});

const hookColor = z.string().regex(/^#[0-9a-f]{6}$/i);

export const hookConfigSchema = z.object({
  text: z.string(),
  position: z.enum(["auto", "top", "center", "bottom"]),
  size: z.enum(["S", "M", "L"]),
  style: z
    .enum(["pill", "classic", "dark", "yellow", "red", "outline", "outline_yellow"])
    .default("pill"),
  font: z.enum(["montserrat", "anton", "serif"]).optional(),
  entranceAnimation: z.enum(["spring", "fade", "slide-up", "none"]),
  displayDurationSec: z.number().positive().nullable(),
  design: z.object({
    shape: z.enum(["pill", "box", "none"]),
    text_color: hookColor,
    background_color: hookColor,
    outline_color: hookColor,
    outline_width: z.number().int().min(0).max(20),
    corner_radius: z.number().int().min(0).max(120),
    text_case: z.enum(["as_written", "uppercase"]),
  }).optional(),
  artwork: z.object({ opening: hookArtworkSchema.nullable(), closing: hookArtworkSchema.nullable() }).optional(),
  cta: z.object({
    text: z.string().min(1).max(180),
    position: z.enum(["auto", "top", "center", "bottom"]),
    size: z.enum(["S", "M", "L"]),
    displayDurationSec: z.number().positive().max(15),
  }).optional(),
});

export const effectSegmentSchema = z.object({
  startSec: z.number().min(0),
  endSec: z.number().positive(),
  zoom: z.number().min(0.5).max(3),
  zoomCenterX: z.number().min(0).max(1),
  zoomCenterY: z.number().min(0).max(1),
  brightness: z.number().min(0).max(3),
  contrast: z.number().min(0).max(3),
  saturate: z.number().min(0).max(3),
});

export const effectsConfigSchema = z.object({
  segments: z.array(effectSegmentSchema),
});

export const shortVideoPropsSchema = z.object({
  videoUrl: z.string(),
  durationInFrames: z.number().int().positive(),
  fps: z.number().positive(),
  width: z.number().int().positive(),
  height: z.number().int().positive(),
  subtitles: subtitleConfigSchema.nullable(),
  hook: hookConfigSchema.nullable(),
  effects: effectsConfigSchema.nullable(),
});
