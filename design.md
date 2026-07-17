# Design — OpenShorts

A locked design system for this app (dashboard + landing). Every page redesign
reads this file before emitting code. Do not regenerate per page — extend or
amend this file when the system needs to grow.

Theme: **Lumen · Night Foundry** (Hallmark catalog, atmospheric genre).
Register: premium AI-tool instrument (Modal / Anthropic / ElevenLabs school).
The product's own numbers — timecodes, aspect ratios, clip counts, costs —
are the decoration: they render as UPPERCASE mono readouts.

## Genre

atmospheric

## Macrostructure family

- Marketing pages (Landing): **Marquee Hero, Lumen-canonical** — lowercase
  serif headline hero-left with one coral verb-landmark, hand-built CSS
  apparatus hero-right, meter strip below the fold, then sections with mono
  eyebrows. Nav: N9 edge-aligned minimal. Footer: Ft5 Statement.
- App pages (dashboard tabs, modals): **Workbench** — small functional
  headings, hairline cards lit from within, mono readouts, zero enrichment.
  Function carries the page.
- Content pages (Legal): **Long Document** — single column, 65ch measure,
  inline headings, typographic links.

## Theme tokens (Night Foundry)

- `--color-paper`      oklch(13% 0.014 265)   — page canvas, late-night violet
- `--color-paper-2`    oklch(16.5% 0.015 265) — elevated surface (cards, modals, sidebar)
- `--color-paper-3`    oklch(20% 0.016 265)   — hover surface / inset chips
- `--color-ink`        oklch(96% 0.006 262)   — headlines, near-white
- `--color-ink-2`      oklch(86% 0.01 262)    — body text
- `--color-muted`      oklch(64% 0.012 262)   — secondary text
- `--color-rule`       oklch(96% 0.006 262 / 0.08) — hairline borders
- `--color-rule-2`     oklch(96% 0.006 262 / 0.14) — stronger hairline
- `--rule-blueprint`   oklch(96% 0.006 262 / 0.04) — blueprint grid lines
- `--color-accent`     oklch(76% 0.17 50)     — molten brass (THE accent)
- `--color-accent-ink` oklch(17% 0.03 50)     — text on brass fills
- `--color-accent-2`   oklch(68% 0.16 18)     — coral chord (verb landmark ONLY + rare secondary)
- `--color-glow`       oklch(80% 0.16 50 / 0.42) — apparatus halo
- `--color-paper-emit` oklch(76% 0.17 50 / 0.04) — inner-emission wash
- `--color-focus`      oklch(76% 0.17 50)     — focus rings
- `--color-ok`         oklch(75% 0.11 150)    — success states
- `--color-warn`       oklch(78% 0.14 75)     — warnings (keys missing, low quota)
- `--color-danger`     oklch(66% 0.18 25)     — errors, destructive

Tailwind names: `paper, paper2, paper3, ink, ink2, muted, brass, brassink,
coral, ok, warn, danger` (+ legacy aliases `background→paper`, `surface→paper2`,
`primary→brass`, `accent→coral` so untouched files degrade gracefully).
Borders: use `border-rule` / `border-rule2` utilities (defined in index.css).

## Typography

- Display: **Instrument Serif 400**, `font-display`, always roman (never italic),
  **lowercase**, tracking -0.032em, line-height 1.02.
- Body: **Geist** 400/500/600, `font-body`.
- Mono: **JetBrains Mono** 400/500, `font-mono` — UPPERCASE labels only.
- Two-register rule: UI prose (headings, buttons, nav, empty states, helper
  text) renders **lowercase**; mono labels (eyebrows, badges, readouts, stat
  labels, keyboard hints) render **UPPERCASE** 10.5–11px tracking 0.10em at
  ~55% opacity. That contrast is the typographic signature.
- **NEVER lowercase user content**: video titles, captions, transcripts,
  descriptions, emails, API keys, log output, filenames, prices in legal copy.
  Apply `lowercase` class per-element on UI chrome only — no global
  text-transform on body.
- Numerals: `font-variant-numeric: tabular-nums` wherever numbers appear
  (`tabular-nums` utility). Big stats: Instrument Serif + tabular-nums.
- Type scale anchor: `--text-display: clamp(3rem, 6vw + 1rem, 6rem)` (landing
  hero only). App headings stay small: text-xl/2xl max, serif lowercase.

## Spacing

4-point scale via Tailwind defaults (p-2/3/4/6/8/12/16). No arbitrary
`text-[9px]`-style values — minimum readable label is 10.5px (`text-micro`
utility). Radii: `--radius-card: 10px` (rounded-card), `--radius-input: 8px`
(rounded-input), pills `rounded-full` for CTAs and status chips only.

## Motion

- Easings: `--ease-out: cubic-bezier(0.16, 1, 0.3, 1)` only. Never `ease`.
- Durations: 220ms UI state, 600ms section reveals, 4s apparatus pulse.
- Reveal pattern: fade only (`animate-fade`), optional 12px translateY on
  section entry. No slide-ins, no bounce, no scale-pops.
- Cards hover: `translateY(-4px)` + inner-glow brighten, 220ms.
- Reduced motion: everything collapses to instant final state.
- Focus ring: 2px brass, appears INSTANTLY (never animated).

## Microinteractions stance

- Silent success (inline check, quiet state change) — never celebratory.
- Buttons: `active:translate-y-px`, no scale tricks.
- Hover tooltips delay 800ms; focus tooltips 0ms.
- Loaders: `Loader2` lucide spin is fine, tinted `text-muted` or `text-brass`.

## CTA voice

- Primary: `.btn-primary` — brass fill, `text-brassink`, `rounded-full`,
  lowercase label, font-medium. One per view section.
- Secondary: `.btn-ghost` — hairline `border-rule2`, ink text, rounded-full.
- Tertiary/utility: `.btn-quiet` — `bg-paper3` fill, no border.
- Destructive: `.btn-danger` — danger outline, fills on hover.
- Labels are verbs, lowercase: "start clipping", "generate", "publish",
  "download". Never "Get Started!", never Title Case.

## Component recipes (shared — use these, don't reinvent)

- `.card` — hairline border + inner emission (radial brass 4–5%), rounded-card,
  bg-paper2. Hover variant `.card-hover` lifts and brightens. Replaces ALL
  glass-panel / bg-surface/50 / backdrop-blur cards.
- `.eyebrow` — mono UPPERCASE micro label. Section pattern: eyebrow stacked
  ABOVE the heading, same column (`01 · TOOLS` then the serif heading). Never
  tag-left/heading-right.
- `<Modal>` from `src/components/ui/Modal.jsx` — the ONLY modal shell
  (overlay plain `bg-black/70`, NO backdrop-blur; panel bg-paper2 hairline).
- `<SegmentedControl>` from `src/components/ui/SegmentedControl.jsx` — all
  option-button grids (position/size/animation pickers, platform toggles).
  Active state = brass hairline + paper3 fill + ink text. Never bg-white
  text-black, never per-feature colors.
- `<StepIndicator>` from `src/components/ui/StepIndicator.jsx` — the single
  wizard stepper (ThumbnailStudio + SaaShortsTab share it). Mono ordinals.
- `.input-field` — inset bg-paper, hairline, focus brass ring (redefined in
  index.css; existing class name kept).
- `.readout` — mono UPPERCASE value chip for machine data (timecodes, costs,
  ratios, model names): `00:42 · 9:16 · $0.65`.
- Status: `.badge-ok` / `.badge-warn` / `.badge-danger` — tinted 10% fills,
  mono uppercase.

## What pages MUST share

- Single accent: brass. Coral ONLY as the headline verb-landmark (landing) and
  at most one small secondary highlight per view. NOTHING ELSE gets a hue.
- Zero gradients (text or background). Zero glassmorphism / backdrop-blur
  panels. Zero per-feature identity colors (the old violet/emerald/amber/teal
  per-tab coding is retired — tools are differentiated by mono eyebrow ordinals
  `01 · CLIPS`, `02 · AI SHORTS`, not by hue).
- The two-register typography (lowercase prose / UPPERCASE mono labels).
- The CTA voice, radii, hairline card language.
- English copy everywhere (ScheduleWeekModal migrates from Spanish).
- Lucide icons at 16–18px, `text-muted` default, brass when active.

## What pages MAY differ on

- Landing may use enrichment: ONE hand-built CSS apparatus (filament chamber
  reading as a 9:16 clip instrument) + blueprint grid + meter strip. App pages
  get NONE of these (no grid bg, no apparatus, no meter) — except ProcessingAnimation,
  which keeps its scanner HUD but re-tinted to brass/ink tokens.
- Legal is typography-only.

## Hard bans (Lumen)

No italics anywhere. No gradient text. No gradient buttons. No backdrop-blur.
No glowing orbs. No invented metrics (existing cited stats/prices are the
brief's real copy — keep them). No emoji as icons in chrome (existing emoji
option labels in SaaShorts pickers may stay as content). No rounded-2xl
soup — use rounded-card/rounded-input/rounded-full deliberately. No
`bg-[#121214]`-style hardcoded colors — tokens only. No Title Case UI copy.

## Functional contract (NEVER break)

Hash routing (`#app`, `#/pricing`, `#/account`, `#legal`, `#features`,
`#how-it-works`, `#pricing`, `#comparison`, `#faq` anchors), `billingEnabled` /
`isManaged` / `isSignedIn` gating, all handlers and props, localStorage keys
(`gemini_key`, `uploadPostKey_v3`, `elevenLabsKey_v1`, `falKey_v1`,
`uploadUserId`, `openshorts_session`, `openshorts_auth`,
`openshorts_skip_landing`), API calls and BYOK headers, QuotaError flows,
Remotion preview/render wiring. Redesign is classes + markup structure only.
