# Blind Hunter Review Prompt

Use the `bmad-review-adversarial-general` skill.

You receive only this diff summary and must review it without project context. Find likely bugs, regressions, unsafe assumptions, and missing tests. Return findings only, ordered by severity, with a short rationale and suggested fix.

## Diff Output

```diff
diff --git a/main.py b/main.py
@@
+def _clean_text(value, fallback=""):
+    """Normalize generated metadata text without inventing new topic details."""
+    if not isinstance(value, str):
+        return fallback
+    text = re.sub(r"\s+", " ", value).strip()
+    return text or fallback
+
+def _clean_hook_text(value, fallback="VIRAL SHORT", max_words=6):
+    """Keep hook overlays short, readable, and stable for downstream UI."""
+    text = _clean_text(value, fallback)
+    words = text.split()
+    if len(words) > max_words:
+        text = " ".join(words[:max_words])
+    return text.upper()
+
+def _clip_text_metadata(clip):
+    """Populate backward-compatible text fields with normalized values."""
+    hook = _clean_hook_text(clip.get("viral_hook_text"))
+    caption = _clean_text(clip.get("social_caption"))
+    title = _clean_text(clip.get("video_title_for_youtube_short"))
+    tiktok = _clean_text(clip.get("video_description_for_tiktok"))
+    instagram = _clean_text(clip.get("video_description_for_instagram"))
+    clip["viral_hook_text"] = hook
+    clip["social_caption"] = caption
+    clip["video_title_for_youtube_short"] = title or hook[:100]
+    clip["video_description_for_tiktok"] = tiktok or caption
+    clip["video_description_for_instagram"] = instagram or caption
@@
-        hook = clip.get("viral_hook_text") or "Viral Short"
-        caption = clip.get("social_caption") or ""
-        clip.setdefault("video_title_for_youtube_short", hook[:100])
-        clip.setdefault("video_description_for_tiktok", caption)
-        clip.setdefault("video_description_for_instagram", caption)
+        _clip_text_metadata(clip)
@@
+    def _word_start(word):
+        return word.get("s", word.get("start"))
+
+    def _word_end(word):
+        return word.get("e", word.get("end"))
+
+    def _prefer_nearby_scene_boundary(t):
+        for sp in scene_points:
+            if abs(sp - t) <= snap_padding:
+                return sp
+        return t
@@
-        # Check if t falls inside a word — snap to word end
+        # If t falls inside a word, expand backward to preserve the word.
         for w in words:
-            if w["s"] <= t <= w["e"]:
-                return w["e"] + snap_padding
+            w_start = _word_start(w)
+            w_end = _word_end(w)
+            if w_start is None or w_end is None:
+                continue
+            if w_start <= t <= w_end:
+                return _prefer_nearby_scene_boundary(w_start - snap_padding)
@@
-        # Check if t falls inside a word — snap to word start
+        # If t falls inside a word, expand forward to preserve the word.
         for w in words:
-            if w["s"] <= t <= w["e"]:
-                return w["s"] - snap_padding
+            w_start = _word_start(w)
+            w_end = _word_end(w)
+            if w_start is None or w_end is None:
+                continue
+            if w_start <= t <= w_end:
+                return w_end
```

New tests in `tests/test_clip_quality.py` cover:
- strongest near-duplicate retained and hook/title/caption normalized
- best fallback retained when all candidates are below confidence threshold
- clip start/end inside words expand to preserve speech
- original boundaries retained when snapping would make the clip too short
