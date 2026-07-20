import { Github } from 'lucide-react';

export const REPO_URL = 'https://github.com/mutonby/openshorts';

// Small "star us" ask, placed at high-delight moments (clips just rendered)
// and at the free-plan ceiling (out-of-minutes modal). No incentive attached.
export default function StarBanner({ message = 'Enjoying OpenShorts?' }) {
  return (
    <a
      href={REPO_URL}
      target="_blank"
      rel="noopener noreferrer"
      className="flex items-center gap-2 px-3 py-2 rounded-input bg-paper3 border border-rule text-sm text-muted hover:text-ink transition-colors"
    >
      <Github size={14} className="shrink-0" />
      <span>{message} <span className="text-brass">Star us on GitHub ⭐</span></span>
    </a>
  );
}
