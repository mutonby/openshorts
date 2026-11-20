"""File-backed review queue: nothing is published until a human approves it."""
import json
import os

PENDING, APPROVED, PUBLISHED = "pending", "approved", "published"


class Queue:
    def __init__(self, path):
        self.path = path

    def load(self):
        if not os.path.exists(self.path):
            return []
        with open(self.path, encoding="utf-8") as f:
            return json.load(f)

    def save(self, items):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.path)

    def add(self, clip):
        items = self.load()
        if any(i["clip_path"] == clip["clip_path"] for i in items):
            return
        items.append({**clip, "status": PENDING, "published": {}})
        self.save(items)

    def approve(self, clip_path):
        items = self.load()
        hit = False
        for i in items:
            if i["clip_path"] == clip_path and i["status"] == PENDING:
                i["status"], hit = APPROVED, True
        self.save(items)
        return hit
