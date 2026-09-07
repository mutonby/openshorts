import { staticFile } from "remotion";

/**
 * CSS @font-face declaration for NotoSerif-Bold (bundled locally).
 * Use in components via: <style>{notoSerifFontFace}</style>
 */
export const NOTO_SERIF_FONT_FAMILY = "NotoSerif-Bold";

export const notoSerifFontFace = `
@font-face {
  font-family: '${NOTO_SERIF_FONT_FAMILY}';
  src: url('${staticFile("fonts/NotoSerif-Bold.ttf")}') format('truetype');
  font-weight: 700;
  font-style: normal;
}
`;

/**
 * Map of subtitle font families to their CSS-safe names.
 * These match the options available in SubtitleModal.jsx.
 */
export const SUBTITLE_FONTS: Record<string, string> = {
  Verdana: "Verdana, Geneva, sans-serif",
  Arial: "Arial, Helvetica, sans-serif",
  Impact: "Impact, Haettenschweiler, sans-serif",
  Helvetica: "Helvetica, Arial, sans-serif",
  Georgia: "Georgia, 'Times New Roman', serif",
  "Courier New": "'Courier New', Courier, monospace",
  // Bundled caption-preset fonts (fonts/*.ttf); the dashboard loads the same
  // families from Google Fonts so the preview matches the burn.
  "Montserrat ExtraBold": "Montserrat, Arial, sans-serif",
  "Bebas Neue": "'Bebas Neue', Impact, sans-serif",
  Bangers: "Bangers, Impact, sans-serif",
};

export function getFontStack(fontFamily: string): string {
  return SUBTITLE_FONTS[fontFamily] ?? fontFamily;
}
