from typing import List, Dict
import json


class FewShotBuilder:
    def build_examples(self, clips: List[Dict], current_transcript: str = "") -> str:
        if not clips:
            return ""

        success = [c for c in clips if c.get("metadata", {}).get("is_success") == True]
        failure = [c for c in clips if c.get("metadata", {}).get("is_success") == False]

        success.sort(
            key=lambda c: float(c.get("metadata", {}).get("engagement_rate", 0)), reverse=True
        )
        failure.sort(
            key=lambda c: float(c.get("metadata", {}).get("engagement_rate", 0)), reverse=True
        )

        top_success = success[:3]
        top_failures = failure[:2]
        top_all = top_success + top_failures

        if not top_all:
            return ""

        lines = ["# LEARNED PATTERNS FROM PAST PERFORMANCE"]

        for clip in top_all:
            meta = clip.get("metadata", {})
            is_success = meta.get("is_success", False)
            views = int(meta.get("views", 0))
            likes = int(meta.get("likes", 0))
            shares = int(meta.get("shares", 0))
            engagement = float(meta.get("engagement_rate", 0))

            label = "HIGH PERFORMANCE" if is_success else "LOW PERFORMANCE"
            stats = (
                f"({views:,} views, {likes:,} likes, {shares:,} shares, "
                f"{engagement:.1%} engagement)"
            )

            lines.append(f"\n## {label} {stats}")

            hook = meta.get("hook_text", "")
            if hook:
                lines.append(f"- Hook: \"{hook}\"")

            pattern = meta.get("viral_pattern", "")
            if pattern:
                lines.append(f"- Viral Pattern: {pattern}")

            segment = meta.get("transcript_segment", "")
            if segment:
                truncated = segment[:300]
                lines.append(f"- Transcript excerpt: \"{truncated}\"")

        lines.append(
            "\n# APPLY THESE LEARNINGS\n"
            "- Use similar hook styles and patterns to the HIGH PERFORMANCE examples\n"
            "- Avoid approaches shown in LOW PERFORMANCE examples\n"
            "- Prioritize viral patterns that have proven engagement in similar content"
        )

        return "\n".join(lines)
