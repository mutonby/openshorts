import os
import json
import re
import uuid
from datetime import datetime
from typing import List, Optional

from learning.models import LearningProfile
from learning.vector_store import VectorStore


class ProfileManager:
    def __init__(self, vector_store: VectorStore, profiles_file: str = None):
        from learning.config import PROFILES_FILE

        self.store = vector_store
        self.profiles_file = profiles_file or PROFILES_FILE
        self._profiles: dict = {}
        self._load()

    def _load(self):
        if os.path.exists(self.profiles_file):
            try:
                with open(self.profiles_file, "r") as f:
                    data = json.load(f)
                    self._profiles = {p["profile_id"]: LearningProfile(**p) for p in data}
            except Exception as e:
                print(f"  [ProfileManager] Failed to load profiles: {e}")
                self._profiles = {}
        else:
            self._profiles = {}

    def _save(self):
        os.makedirs(os.path.dirname(self.profiles_file), exist_ok=True)
        with open(self.profiles_file, "w") as f:
            json.dump(
                [p.model_dump(mode="json") if hasattr(p, "model_dump") else p.__dict__ for p in self._profiles.values()],
                f,
                indent=2,
            )

    def _slugify(self, name: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        return slug or "profile"

    def create_profile(self, name: str, base_category: str, description: str = "") -> LearningProfile:
        profile_id = self._slugify(name)
        if profile_id in self._profiles:
            profile_id = f"{profile_id}_{uuid.uuid4().hex[:6]}"

        profile = LearningProfile(
            profile_id=profile_id,
            name=name,
            base_category=base_category,
            description=description,
        )
        self._profiles[profile_id] = profile
        self._save()
        return profile

    def get_profile(self, profile_id: str) -> Optional[LearningProfile]:
        profile = self._profiles.get(profile_id)
        if profile:
            stats = self.store.get_profile_stats(profile_id)
            profile.clip_count = stats["clip_count"]
            profile.avg_engagement = stats["avg_engagement"]
        return profile

    def list_profiles(self) -> List[LearningProfile]:
        profiles = []
        for p in self._profiles.values():
            stats = self.store.get_profile_stats(p.profile_id)
            p.clip_count = stats["clip_count"]
            p.avg_engagement = stats["avg_engagement"]
            profiles.append(p)
        return profiles

    def update_profile(self, profile_id: str, updates: dict) -> Optional[LearningProfile]:
        if profile_id not in self._profiles:
            return None
        profile = self._profiles[profile_id]
        for key, value in updates.items():
            if hasattr(profile, key):
                setattr(profile, key, value)
        self._save()
        return profile

    def delete_profile(self, profile_id: str, delete_clips: bool = True):
        if delete_clips:
            clips = self.store.get_profile_clips(profile_id)
            for c in clips:
                self.store.delete_clip(c["clip_id"])
        self._profiles.pop(profile_id, None)
        self._save()
