import { createElement } from 'react';
import { Video, Instagram, Youtube, Facebook, Linkedin, AtSign, Cloud } from 'lucide-react';

// The networks a clip can be posted to, mirroring social_platforms.PLATFORMS on
// the server. The first three are the ones picked by default when the
// connected accounts are unknown (self-hosted, BYOK).
export const SOCIAL_PLATFORMS = [
  { value: 'tiktok', label: 'TikTok', Icon: Video, byDefault: true },
  { value: 'instagram', label: 'Instagram', Icon: Instagram, byDefault: true },
  { value: 'youtube', label: 'YouTube', Icon: Youtube, byDefault: true },
  { value: 'facebook', label: 'Facebook', Icon: Facebook },
  { value: 'linkedin', label: 'LinkedIn', Icon: Linkedin },
  { value: 'threads', label: 'Threads', Icon: AtSign },
  { value: 'bluesky', label: 'Bluesky', Icon: Cloud },
];

export const PLATFORM_VALUES = SOCIAL_PLATFORMS.map((p) => p.value);

export const platformLabel = (value) =>
  (SOCIAL_PLATFORMS.find((p) => p.value === value) || {}).label || value;

// SegmentedControl options with icons; `lower` keeps the lower-case UI style.
export const platformOptions = ({ lower = false, size = 16 } = {}) =>
  SOCIAL_PLATFORMS.map(({ value, label, Icon }) => ({
    value,
    label: lower ? label.toLowerCase() : label,
    icon: createElement(Icon, { size }),
  }));

// { tiktok: bool, ... } from a predicate over the platform value.
export const platformFlags = (pick) =>
  Object.fromEntries(PLATFORM_VALUES.map((v) => [v, Boolean(pick(v))]));

export const defaultPlatformFlags = () =>
  platformFlags((v) => SOCIAL_PLATFORMS.find((p) => p.value === v).byDefault);

export const flagsFromList = (arr) => platformFlags((v) => (arr || []).includes(v));
